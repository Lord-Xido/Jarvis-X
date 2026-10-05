#!/usr/bin/env python3
"""Assemble the canonical DM3D 64-bit bootstrap bytecode."""
from pathlib import Path
import struct
import sys

OPS = {
    "NOP": 0x00, "BOOT": 0x01, "CONFIG_FABRIC": 0x02, "OPEN_STREAM": 0x03,
    "NEXT_WINDOW": 0x04, "DECODE_MEDIA": 0x05, "ENCODE_MODAL": 0x06,
    "FUSE_3D": 0x07, "FOLD_INWARD": 0x08, "DECODE_MODAL": 0x09,
    "RESIDUAL": 0x0A, "UPDATE_OMEGA": 0x0B, "FIXPOINT_CHECK": 0x0C,
    "BRANCH_IF_NOT_CONVERGED": 0x0D, "EMIT": 0x0E,
    "CLOSE_WINDOW": 0x0F, "JUMP": 0x10, "HALT": 0xFF,
}

def enc(op, dst=0, a=0, b=0, imm=0):
    return ((OPS[op] & 0xff) << 56) | ((dst & 0xff) << 48) | \
           ((a & 0xff) << 40) | ((b & 0xff) << 32) | (imm & 0xffffffff)

p = []
def e(*args):
    p.append(enc(*args))

e("BOOT")
e("CONFIG_FABRIC", 1, 0, 0, 1000)       # 1000 KB -> 1,000,000 logical coords/axis
e("OPEN_STREAM", 2, 0, 0, 0b111111)     # video/audio/image/text/depth/metadata

window = len(p)
e("NEXT_WINDOW", 2, 2, 0, 32)
e("DECODE_MEDIA", 3, 2, 0, 0b111111)
e("ENCODE_MODAL", 4, 3, 0, 64)
e("FUSE_3D", 5, 4, 9, 0)

fold = len(p)
e("FOLD_INWARD", 6, 5, 9, 985000)       # lambda = 0.985
e("DECODE_MODAL", 7, 6, 0, 0b111111)
e("RESIDUAL", 8, 3, 7, 0)
e("UPDATE_OMEGA", 9, 9, 8, 880000)      # beta = 0.88
e("FIXPOINT_CHECK", 0, 8, 10, 1000)     # epsilon = 0.001

pc = len(p)
e("BRANCH_IF_NOT_CONVERGED", 0, 11, 0, fold - (pc + 1))
e("EMIT", 12, 7, 6, 0b111111)
e("CLOSE_WINDOW", 0, 2, 0, 0)

pc = len(p)
e("JUMP", 0, 0, 0, window - (pc + 1))
e("HALT")

out = Path(sys.argv[1] if len(sys.argv) > 1 else "dm3d_system.bc")
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("wb") as f:
    for word in p:
        f.write(struct.pack(">Q", word))

print(f"wrote {len(p)} instructions / {len(p) * 8} bytes -> {out}")
