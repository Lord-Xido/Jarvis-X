from __future__ import annotations

import base64
import math

import pytest
from fastapi import HTTPException

from jarvisx.hyper3d_api import Attachment, ExecuteRequest, execute as api_execute, get_capabilities, healthz, index
from jarvisx.hyper3d_cli import parser as cli_parser
from jarvisx.hyper3d_runtime import (
    ABITS, AXIS, CH, DEFAULT_PROGRAM, Address128, Hyper3DRuntime,
    Instruction64, Modality, Opcode, compile_program, decode_block, encode_block,
)

def test_address_128_round_trip():
    a=Address128.from_hierarchy(6399,2,3,4,5,6,7,8,9)
    assert ABITS*3==114
    assert Address128.unpack(a.pack())==a
    assert a.hierarchy()['x']==(6399,2,3)
    assert a.x<AXIS and a.y<AXIS and a.z<AXIS

def test_instruction_64_round_trip():
    i=Instruction64(5,Opcode.FOLD,0xA55,37,-2,7,-1)
    assert Instruction64.decode(i.encode())==i
    assert i.hex().startswith('0x')

def test_eight_haar_channels_are_orthonormal():
    for k in range(CH):
        z=[0.0]*CH; z[k]=1.0
        recovered=encode_block(decode_block(z))
        for c,v in enumerate(recovered):
            assert math.isclose(v,1.0 if c==k else 0.0,abs_tol=1e-7)

def test_program_compiles_to_64_bit_words():
    p=compile_program(DEFAULT_PROGRAM)
    assert p[-1].opcode is Opcode.HALT
    assert any(x.opcode is Opcode.FEEDBACK for x in p)
    assert all(0<=x.encode()<=0xffffffffffffffff for x in p)

def run_once():
    r=Hyper3DRuntime(64)
    r.ingest(Modality.TEXT,'Jarvis X echo through','prompt.txt')
    r.ingest(Modality.IMAGE,bytes(range(256))*4,'image.raw')
    return r.execute(DEFAULT_PROGRAM)

def test_end_to_end_is_deterministic_and_bounded():
    a=run_once(); b=run_once()
    assert a['telemetry']['active_nodes']<=64
    assert a['telemetry']['state_sha256']==b['telemetry']['state_sha256']
    assert a['architecture']['logical_cells']==str(6400**9)
    assert a['architecture']['execution_model']=='sparse-active-set'
    assert a['architecture']['block_compression_ratio']==8.0
    assert a['telemetry']['feedback_passes']==1
    assert a['telemetry']['fusion_passes']==1
    assert a['points'] and a['reconstructed']

def test_bitwise_and_geometric_telemetry():
    r=run_once(); p=r['points'][0]
    assert len(p['position'])==3
    assert p['control_word'].startswith('0x')
    assert len(p['address'])==34
    assert any(x['opcode']=='BITMIX' for x in r['trace'])
    assert all(x['word'].startswith('0x') for x in r['trace'])


def test_api_surface_and_cli_parser():
    req=ExecuteRequest(
        text='api-text',
        attachments=[Attachment(modality=Modality.AUDIO,name='audio.raw',data_base64=base64.b64encode(b'abc'*20).decode())],
        max_active_nodes=32,
    )
    result=api_execute(req)
    assert result['telemetry']['active_nodes']>0
    assert get_capabilities()['instruction_bits']==64
    assert healthz()=={'status':'ok'}
    assert index().path.endswith('index.html')
    args=cli_parser().parse_args(['--host','127.0.0.1','--port','9001'])
    assert args.port==9001

def test_api_rejects_bad_base64():
    req=ExecuteRequest(attachments=[Attachment(modality=Modality.GENERIC,name='bad.bin',data_base64='***')])
    with pytest.raises(HTTPException):
        api_execute(req)
