"""Black-box parity tests against the original v4 C++ runtime."""
import os
from pathlib import Path
import secrets
import subprocess
import sys

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE))
from jx_accel import Archive, NeuralCodec, Volume, read_bounded, SIDE
from service import create_app

torch.set_num_threads(1)

@pytest.fixture(scope='session')
def runtime(tmp_path_factory):
    dest = tmp_path_factory.mktemp('parity')
    cpp = Path(os.environ.get('JX_CPP_BIN','')).resolve()
    if not cpp.is_file():
        pytest.skip('Set JX_CPP_BIN to the compiled v4 C++ runtime')
    vol = dest/'input.jxvol'
    model = dest/'model.jxmodel'
    subprocess.run([str(cpp),'generate',str(vol),'16','9'],check=True,capture_output=True)
    subprocess.run([str(cpp),'train',str(vol),str(model),'12','0.004'],check=True,capture_output=True)
    engine = NeuralCodec(model.read_bytes(),batch_blocks=64)
    return cpp, model, vol, engine, dest


def cpp_cmd(runtime, *args):
    p=subprocess.run([str(runtime[0]),*map(str,args)],capture_output=True,text=True)
    if p.returncode!=0:
        raise RuntimeError(f'cpp command failed: {p.stderr} {p.stdout}')
    return p.stdout


def test_cpp_model_format_and_crc(runtime):
    raw=runtime[1].read_bytes()
    assert len(raw)==40054
    assert runtime[3].model_hash>0
    bad=bytearray(raw);bad[47]^=1
    with pytest.raises(ValueError,match='CRC32'):NeuralCodec(bytes(bad))
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError,match='CUDA'):
            NeuralCodec(raw,'cuda')


def test_exact_torch_to_cpp(runtime):
    _,model,vol,codec,dir=runtime
    v=Volume.parse(vol.read_bytes())
    packet=codec.encode(v,True,(SIDE-16,1234567,42)).serialize()
    p=dir/'torch_exact.jxpkt';out=dir/'cpp_restore.jxvol';p.write_bytes(packet)
    cpp_cmd(runtime,'decode',p,model,out)
    assert out.read_bytes()==vol.read_bytes()
    decoded=codec.decode(Archive.parse(packet)).serialize()
    assert decoded==vol.read_bytes()


def test_cpp_to_torch_lossy_and_exact_parity(runtime):
    _,model,vol,codec,dir=runtime
    v=Volume.parse(vol.read_bytes())
    for exact in (False,True):
        mode='exact' if exact else 'lossy'
        packet=dir/f'cpp_{mode}.jxpkt';out=dir/f'cpp_{mode}.jxvol'
        cpp_cmd(runtime,'encode',vol,model,packet,mode)
        cpp_cmd(runtime,'decode',packet,model,out)
        parsed=Archive.parse(packet.read_bytes())
        assert parsed.model_hash==codec.model_hash
        assert codec.decode(parsed).serialize()==out.read_bytes()
        # For this synthetic volume, q and prediction should agree byte-for-byte.
        ours=codec.encode(v,exact).serialize()
        assert ours==packet.read_bytes()


def test_random_bytes_parity(runtime):
    _,model,vol,codec,dir=runtime
    rng=np.random.default_rng(816)
    for n in (8,16):
        inp=dir/f'random_{n}.raw'
        inp.write_bytes(rng.integers(0,256,size=n**3,dtype=np.uint8).tobytes())
        wrapped=dir/f'random_{n}.jxvol'
        cpp_cmd(runtime,'import-raw',inp,n,n,n,wrapped)
        v=Volume.parse(wrapped.read_bytes())
        for exact in (False,True):
            p=dir/f'random_{n}_{exact}.jxpkt';out=dir/f'random_{n}_{exact}.jxvol'
            cpp_cmd(runtime,'encode',wrapped,model,p,'exact' if exact else 'lossy')
            cpp_cmd(runtime,'decode',p,model,out)
            a=codec.encode(v,exact).serialize()
            assert a==p.read_bytes()
            assert codec.decode(Archive.parse(a)).serialize()==out.read_bytes()
            if exact:assert out.read_bytes()==wrapped.read_bytes()


def test_corruption_and_model_fingerprint(runtime):
    _,model,vol,codec,dir=runtime
    packet=codec.encode(Volume.parse(vol.read_bytes()),True).serialize()
    bad=bytearray(packet);bad[len(bad)//2]^=1
    with pytest.raises(ValueError,match='CRC32'):Archive.parse(bytes(bad))
    with pytest.raises(ValueError,match='archive payload length'):Archive.parse(packet[:-1])
    with pytest.raises(ValueError,match='model fingerprint'):
        arc=Archive.parse(packet)
        from dataclasses import replace
        codec.decode(replace(arc,model_hash=arc.model_hash^1))


def test_virtual_origin_and_query(runtime):
    _,model,vol,codec,dir=runtime
    v=Volume.parse(vol.read_bytes())
    origin=(SIDE-16,SIDE-16,SIDE-16)
    packet=codec.encode(v,True,origin).serialize()
    a=Archive.parse(packet)
    for xyz in ((0,0,0),(15,15,15),(1,2,3)):
        pos=tuple(x+y for x,y in zip(xyz,origin))
        r=codec.probe(a,pos)
        assert r['decoded']==int(v.voxels[xyz[2],xyz[1],xyz[0]])
    with pytest.raises(ValueError,match='out of bounds'):codec.probe(a,(0,0,0))
    with pytest.raises(ValueError,match='outside'):codec.encode(v,False,(SIDE-15,0,0))


def test_api_end_to_end(runtime):
    _,model,vol,codec,dir=runtime
    app=create_app(str(model),'secrets-only-valid-123456',max_body=8192)
    client=TestClient(app)
    r=client.get('/healthz')
    assert r.status_code==200 and r.json()['device']=='cpu'
    original=vol.read_bytes()
    assert client.post('/v1/encode',content=original).status_code==401
    headers={'Authorization':'Bearer secrets-only-valid-123456'}
    bad={'Authorization':'Bearer guess'}
    assert client.post('/v1/encode',content=original,headers=bad).status_code==401
    r=client.post('/v1/encode?exact=true&ox=999999999984&oy=0&oz=123',content=original,headers=headers)
    assert r.status_code==200, r.text[:200]
    p=Archive.parse(r.content);assert p.exact
    q=client.post('/v1/query?x=999999999985&y=2&z=126',content=r.content,headers=headers)
    assert q.status_code==200 and q.json()['exact']
    d=client.post('/v1/decode',content=r.content,headers=headers)
    assert d.status_code==200 and d.content==original
    invalid=client.post('/v1/decode',content=r.content[:-1],headers=headers)
    assert invalid.status_code==422
    huge=client.post('/v1/encode',content=b'X'*8200,headers=headers)
    assert huge.status_code==413
    out_of_bounds=client.post('/v1/encode?ox=999999999999',content=original,headers=headers)
    assert out_of_bounds.status_code==422


def test_input_limits(runtime):
    with pytest.raises(ValueError,match='dimensions'):Volume.from_blocks((8,6,8),np.zeros((6,64),dtype=np.uint8))
    with pytest.raises(ValueError,match='bad JXVOL4'):Volume.parse(b'hello')
    with pytest.raises(ValueError,match='bad JXPKT4'):Archive.parse(b'hello')
