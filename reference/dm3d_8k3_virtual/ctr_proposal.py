#!/usr/bin/env python3
"""Map JX8KT1 evidence to ADR-016-shaped *provisional* transaction metadata."""
import argparse
import json
from pathlib import Path

REQUIRED = {"size","version","coordinates","morton39","decode2","bit_exact","fnv","metadata"}


def build_proposal(receipt: dict, report: dict) -> dict:
    if receipt.get("schema") != "JX8K-CTR-1" or receipt.get("status") != "PASS":
        raise ValueError("unverified receipt")
    if not REQUIRED.issubset(set(receipt.get("checkpoints",[]))):
        raise ValueError("missing verification checkpoints")
    if receipt.get("authority") != "provisional" or receipt.get("commit_eligible") is not False:
        raise ValueError("non-provisional input is prohibited")
    for name in ("packet_sha256","raw_sha256"):
        digest=receipt.get(name,"")
        if len(digest)!=64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("invalid digest: "+name)
    if (report.get("world_side")!=8192 or report.get("tile_side")!=16
            or report.get("voxel_bits")!=2 or report.get("tiles_rejected")!=0
            or receipt.get("verified_tiles")!=report.get("tiles_verified")
            or receipt.get("physical_raw_bytes")!=report.get("materialized_raw_bytes")
            or receipt.get("encoded_packet_bytes")!=report.get("packet_file_bytes")):
        raise ValueError("receipt/report mismatch")
    return {
        "schema":"JX8K-TXPROPOSAL-1",
        "authority":"provisional",
        "commit_eligible":False,
        "production_state_modified":False,
        "geometry_profile":{"world_side":8192,"address_bits":39,"tile_side":16,
                            "value_bits":2,"wire_format":"JX8KT1"},
        "typed_state":{
            "spatial":{"selected_tiles":receipt["verified_tiles"],"physical_raw_bytes":receipt["physical_raw_bytes"]},
            "prediction":{"decode_exact":True},
            "residual":{"hamming_nonzero_after_correction":0},
            "evidence":{"receipt_schema":receipt["schema"],"packet_sha256":receipt["packet_sha256"]},
            "audit":{"raw_sha256":receipt["raw_sha256"]},
        },
        "gates":{"type":"pass","geometry_address":"pass","reconstruction":"pass",
                 "integrity_against_reference":"pass","resource":"bounded_sample_only",
                 "fixed_point":"not_evaluated","model_generalization":"not_implemented",
                 "external_world_correspondence":"not_evaluated","production_policy":"not_evaluated"},
        "next_action":"Review evidence; production authority requires explicit integration and all applicable gates."
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--receipt",required=True)
    p.add_argument("--report",required=True)
    p.add_argument("--out")
    a=p.parse_args()
    proposal=build_proposal(json.loads(Path(a.receipt).read_text()),
                            json.loads(Path(a.report).read_text()))
    data=json.dumps(proposal,indent=2,sort_keys=True)+"\n"
    if a.out:Path(a.out).write_text(data)
    print(data,end="")


if __name__=="__main__":
    main()
