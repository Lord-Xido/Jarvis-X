from __future__ import annotations

import math

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
