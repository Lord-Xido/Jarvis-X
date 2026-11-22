#!/usr/bin/env python3
"""Independent output audit: metrics, 3D volume, header and exact recovered bytes."""
import csv
import hashlib
import math
from pathlib import Path
import struct
import sys
import zlib

def audit(prefix):
    p=Path(prefix)
    with p.with_name(p.name+"_metrics.csv").open(newline="") as f:
        rows=list(csv.DictReader(f))
    assert [r["dataset"] for r in rows]==[
        "smooth_holdout_A","smooth_holdout_B","OOD_blocks","OOD_random"]
    for r in rows:
        sizes=[int(r[k]) for k in (
            "raw_packet","zstd_raw_packet","mean_residual_packet","learned_residual_packet")]
        assert int(r["selected_packet"])==min(sizes)
        for field in ("nearest_rmse","trilinear_rmse","untrained_rmse","joint_trained_rmse","feedback6_rmse"):
            assert math.isfinite(float(r[field])) and float(r[field])>=0
    with p.with_name(p.name+"_heldout.csv").open(newline="") as f:
        voxels=list(csv.DictReader(f))
    assert len(voxels)==4096
    coords={(int(v["x"]),int(v["y"]),int(v["z"])) for v in voxels}
    assert len(coords)==4096 and all(0<=axis<16 for point in coords for axis in point)
    rmse=math.sqrt(sum((float(v["learned_prediction"])-int(v["source"])/255.)**2 for v in voxels)/4096)
    assert abs(rmse-float(rows[0]["joint_trained_rmse"]))<3e-7
    raw=p.with_name(p.name+"_original.raw").read_bytes()
    decoded=p.with_name(p.name+"_recovered.raw").read_bytes()
    packet=p.with_name(p.name+"_encoded.jx3d").read_bytes()
    assert len(raw)==len(decoded)==4096 and raw==decoded
    assert len(packet)==int(rows[0]["selected_packet"])
    magic,version,method,flags,n,size,checksum=struct.unpack_from("<4sBBHIII",packet)
    assert magic==b"JX3D" and version==1 and flags==0 and n==4096
    assert method==2 and size==len(packet)-20
    assert checksum==zlib.crc32(raw)
    assert float(rows[0]["joint_trained_rmse"])<float(rows[0]["trilinear_rmse"])
    print("AUDIT_PASS",len(raw),"voxels",len(packet),"encoded bytes","SHA256",hashlib.sha256(raw).hexdigest())

if __name__=="__main__":
    if len(sys.argv)!=2:sys.exit("usage: audit.py PREFIX")
    audit(sys.argv[1])
