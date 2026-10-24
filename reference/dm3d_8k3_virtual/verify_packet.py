#!/usr/bin/env python3
"""Independent stdlib-only JX8KT1 decoder; verifies finite bytes, not cognition."""
import argparse
import hashlib
import json
import struct
from pathlib import Path

WORLD, TILE, RAW_TILE_BYTES = 8192, 16, 1024
MAGIC = b"JX8KT1\0\0"


def morton3(x: int, y: int, z: int) -> int:
    if any(not 0 <= a < WORLD for a in (x, y, z)):
        raise ValueError("Morton coordinate outside 8192^3")
    return sum(((a >> i) & 1) << (3*i + k) for i in range(13)
               for k, a in enumerate((x, y, z)))


def get2(buf: bytes, i: int) -> int:
    return (buf[i >> 2] >> (2*(i & 3))) & 3


def set2(buf: bytearray, i: int, value: int) -> None:
    p, shift = i >> 2, 2*(i & 3)
    buf[p] = (buf[p] & ~(3 << shift)) | ((value & 3) << shift)


def index(x: int, y: int, z: int) -> int:
    return (z*TILE + y)*TILE + x


def decode(payload: bytes) -> bytes:
    if not payload:
        raise ValueError("empty payload")
    mode = payload[0]
    if mode == 0:
        if len(payload) != RAW_TILE_BYTES + 1:
            raise ValueError("raw payload length mismatch")
        return payload[1:]
    if mode not in (2,4,8,16) or len(payload) < 3:
        raise ValueError("invalid block mode or header")
    count = struct.unpack_from("<H", payload, 1)[0]
    side = TILE // mode
    n_prediction_bytes = (side**3 + 3) // 4
    if count > TILE**3 or len(payload) != 3 + n_prediction_bytes + count*3:
        raise ValueError("invalid correction count or length")
    predictors = payload[3:3+n_prediction_bytes]
    result = bytearray(RAW_TILE_BYTES)
    for z in range(TILE):
        for y in range(TILE):
            for x in range(TILE):
                coarse = ((z//mode)*side + y//mode)*side + x//mode
                set2(result, index(x,y,z), get2(predictors,coarse))
    seen = set()
    for j in range(count):
        local, value = struct.unpack_from("<HB", payload, 3+n_prediction_bytes+3*j)
        if local >= TILE**3 or value > 3 or local in seen:
            raise ValueError("invalid or duplicate exception")
        seen.add(local)
        set2(result,local,value)
    return bytes(result)


def fnv1a(raw: bytes) -> str:
    h = 0xcbf29ce484222325
    for v in raw:
        h = ((h ^ v)*0x100000001b3) & ((1 << 64)-1)
    return f"{h:016x}"


def verify(report: dict, packet: bytes, raw: bytes) -> dict:
    if (report.get("world_side") != WORLD or report.get("tile_side") != TILE
            or report.get("voxel_bits") != 2 or report.get("voxel_address_bits") != 39):
        raise ValueError("incompatible geometry profile")
    observations = report["observations"]
    expected = report["tiles_verified"]
    if len(observations) != expected or len(raw) != expected*RAW_TILE_BYTES:
        raise ValueError("reference length/observation count mismatch")
    if not packet.startswith(MAGIC):
        raise ValueError("packet magic mismatch")
    pos = 8
    sum_payload = 0
    visited = set()
    for i, obs in enumerate(observations):
        if pos+8 > len(packet):
            raise ValueError("truncated record")
        x,y,z,n = struct.unpack_from("<HHHH",packet,pos)
        pos += 8
        if n == 0 or pos+n > len(packet):
            raise ValueError("invalid/truncated payload")
        content = packet[pos:pos+n]
        pos += n
        if not all(0 <= v < 512 for v in (x,y,z)):
            raise ValueError("tile address outside [0,512)^3")
        if (x,y,z) in visited:
            raise ValueError("duplicate tile")
        visited.add((x,y,z))
        if (x,y,z) != (obs["x"],obs["y"],obs["z"]):
            raise ValueError("tile metadata mismatch")
        if morton3(x*TILE,y*TILE,z*TILE) != obs["morton"]:
            raise ValueError("Morton mismatch")
        reconstruction = decode(content)
        reference = raw[i*RAW_TILE_BYTES:(i+1)*RAW_TILE_BYTES]
        if reconstruction != reference or fnv1a(reconstruction) != obs["fnv1a_hex"]:
            raise ValueError("lossless round-trip failed")
        if obs["compressed_bytes"] != n or obs["block_mode"] != content[0]:
            raise ValueError("mode/length inconsistency")
        sum_payload += n
    if pos != len(packet):
        raise ValueError("trailing packet bytes")
    if (sum_payload != report["encoded_bytes"]
            or len(packet) != report["packet_file_bytes"]
            or len(raw) != report["materialized_raw_bytes"]
            or report["tiles_rejected"] != 0):
        raise ValueError("reported accounting inconsistent")
    return {
        "schema":"JX8K-CTR-1", "status":"PASS",
        "authority":"provisional", "commit_eligible":False,
        "verified_tiles":expected,"virtual_voxels":WORLD**3,
        "physical_raw_bytes":len(raw),"encoded_packet_bytes":len(packet),
        "verified_packet_ratio":round(len(raw)/len(packet),6),
        "packet_sha256":hashlib.sha256(packet).hexdigest(),
        "raw_sha256":hashlib.sha256(raw).hexdigest(),
        "checkpoints":["size","version","coordinates","morton39","decode2","bit_exact","fnv","metadata"],
        "external_accuracy":"not evaluated","ann_training":"not implemented",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report",required=True)
    p.add_argument("--encoded",required=True)
    p.add_argument("--raw",required=True)
    p.add_argument("--receipt")
    a = p.parse_args()
    receipt = verify(json.loads(Path(a.report).read_text()),
                     Path(a.encoded).read_bytes(),Path(a.raw).read_bytes())
    data = json.dumps(receipt,indent=2,sort_keys=True)+"\n"
    if a.receipt:
        Path(a.receipt).write_text(data)
    print(data,end="")


if __name__=="__main__":
    main()
