#!/usr/bin/env python3
"""Map JX8KT1 evidence to ADR-016-shaped *provisional* transaction metadata."""
import argparse
import json
from pathlib import Path

REQUIRED = {"size","version","coordinates","morton39","decode2","bit_exact","fnv","metadata"}


def build_proposal(receipt: dict, report: dict, vm_receipt: dict | None = None) -> dict:
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
    if vm_receipt is not None:
        import hashlib
        from tile_vm_adapter import default_bytecode

        count = report["tiles_verified"]
        if (vm_receipt.get("schema") != "JX8K-TILEVM-1"
                or vm_receipt.get("status") != "PASS"
                or vm_receipt.get("authority") != "provisional"
                or vm_receipt.get("commit_eligible") is not False
                or vm_receipt.get("production_state_modified") is not False
                or vm_receipt.get("packet_sha256") != receipt["packet_sha256"]
                or vm_receipt.get("raw_sha256") != receipt["raw_sha256"]
                or vm_receipt.get("program_sha256") != hashlib.sha256(default_bytecode()).hexdigest()
                or vm_receipt.get("tiles_processed") != count
                or vm_receipt.get("physical_tensor_values") != count * 4096
                or vm_receipt.get("instruction_steps") != count * 6 + 1):
            raise ValueError("VM staging evidence does not match verified tile packet")
        observations = vm_receipt.get("observations")
        if (not isinstance(observations, list) or len(observations) != count
                or any(not x.get("bit_exact", False) or sum(x.get("counts", [])) != 4096
                       or x.get("tensor_dtype") != "float32"
                       or x.get("tensor_shape") != [16, 16, 16]
                       or x.get("tile_coord") != [report["observations"][i][a] for a in ("x", "y", "z")]
                       or x.get("morton39") != report["observations"][i]["morton"]
                       for i, x in enumerate(observations))):
            raise ValueError("VM per-tile geometric observations are invalid")
    proposal = {
        "schema": "JX8K-TXPROPOSAL-1",
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
    proposal["gates"]["tile_vm_staging"] = "pass" if vm_receipt is not None else "not_evaluated"
    if vm_receipt is not None:
        proposal["typed_state"]["vm_staging"] = {
            "schema": vm_receipt["schema"],
            "program_sha256": vm_receipt["program_sha256"],
            "tiles_processed": vm_receipt["tiles_processed"],
            "tensor_dtype": "float32",
            "voxel_values_per_tile": 4096,
        }
    return proposal


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--receipt",required=True)
    p.add_argument("--report",required=True)
    p.add_argument("--out")
    p.add_argument("--vm-receipt")
    a=p.parse_args()
    proposal=build_proposal(json.loads(Path(a.receipt).read_text()),
                            json.loads(Path(a.report).read_text()),
                            json.loads(Path(a.vm_receipt).read_text()) if a.vm_receipt else None)
    data=json.dumps(proposal,indent=2,sort_keys=True)+"\n"
    if a.out:Path(a.out).write_text(data)
    print(data,end="")


if __name__=="__main__":
    main()
