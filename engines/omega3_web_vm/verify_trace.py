"""Validate the archived or freshly generated 1000-cycle trace."""
import json
import math
import sys
from pathlib import Path

def require(ok, msg):
    if not ok:
        raise ValueError(msg)

def main(path):
    p = Path(path)
    s = json.loads((p / "summary.json").read_text())
    rows = [json.loads(line) for line in (p / "trace.jsonl").read_text().splitlines()]
    require(len(rows) == s["iterations"] == 1000, "iteration count")
    require(s["bit_errors"] == 0 and s["restart_verified"], "integrity")
    require(s["self_source_roundtrip"] and s["pdf_core_source_roundtrip"], "source recovery")
    require(s["corruption_detected"] and s["continuation_test"], "fault/address tests")
    require(s["accepted"] == 999 and s["rolled_back"] == 1, "commit counts")
    prior_loss = s["initial_loss"]
    prior_hash = None
    seal = rows[0]["corpus_seal"]
    for i, row in enumerate(rows, 1):
        require(row["iteration"] == i, "sequence")
        require(math.isfinite(row["loss"]) and row["loss"] <= prior_loss + 1e-12, "loss regression")
        require(row["corpus_seal"] == seal and row["probe_hits"] == 2, "seal/probes")
        require(row["accepted"] == (i != 500), "unexpected commit/rollback")
        if not row["accepted"]:
            require(row["model_sha256"] == prior_hash and row["loss"] == prior_loss, "rollback changed state")
        prior_loss, prior_hash = row["loss"], row["model_sha256"]
    require(math.isclose(prior_loss, s["final_loss"], abs_tol=1e-12), "final loss")
    print("PASS: 1000 cycles, 999 commits, 1 rollback, zero reported binary errors")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "test_output")
