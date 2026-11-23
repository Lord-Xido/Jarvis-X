"""Integration regression: one-shot Riemannian optimization, cache, semantic checks, provenance and rollback."""
from __future__ import annotations
import json
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "one_shot.py"


def check(*args: str, success: bool = True) -> dict | None:
    cp = subprocess.run([sys.executable, str(SCRIPT), *args], text=True, capture_output=True)
    if success and cp.returncode != 0:
        raise AssertionError(f"expected success: {cp.stdout}\n{cp.stderr}")
    if not success and cp.returncode == 0:
        raise AssertionError(f"expected rejection: {cp.stdout}\n{cp.stderr}")
    if success:
        assert "ACCEPTED:" in cp.stdout, cp.stdout
    return None


def run(compiler: str = "g++") -> None:
    with tempfile.TemporaryDirectory(prefix="nano3d-one-shot-test-") as td:
        root = Path(td) / "out"
        base = ["--root", str(root), "--files", "2", "--lines", "24", "--steps", "1000000", "--compiler", compiler]
        check(*base, "--release", "v1")
        a = json.loads((root / "releases" / "v1" / "receipt.json").read_text())
        assert a["geometry"]["iterations"] == 1000000
        assert a["geometry"]["verified"]
        assert a["geometry"]["final_loss"] <= 1e-6
        assert a["checked_runtime_cases"] == 2*24*4
        assert a["cache_hit"] is False
        check(*base, "--release", "cache")
        b = json.loads((root / "releases" / "cache" / "receipt.json").read_text())
        assert b["cache_hit"] is True
        assert a["sha256"] == b["sha256"]
        previous = (root / "latest.json").read_bytes()
        check(*base, "--release", "invalid", "--inject-invalid", success=False)
        assert (root / "latest.json").read_bytes() == previous
        assert not (root / "releases" / "invalid").exists()
        check(*base, "--release", "new-context", "--task-version", "different-v2")
        changed = json.loads((root / "releases" / "new-context" / "receipt.json").read_text())
        assert changed["cache_hit"] is False
        assert changed["cache_key"] != a["cache_key"]
        # Corruption invalidates cached data, never bypasses source/behavior checks.
        source0 = root / "prediction-cache" / a["cache_key"] / "source" / "generated_00000.cpp"
        source0.write_text("// tampered\n")
        check(*base, "--release", "stale")
        stale = json.loads((root / "releases" / "stale" / "receipt.json").read_text())
        assert stale["cache_hit"] is False
        assert stale["sha256"] == a["sha256"]
        # Negative expected net reuse gain forbids the cache despite a valid entry.
        check(*base, "--release", "no-reuse", "--task-version", "different-v2", "--expected-reuses", "0")
        no_reuse = json.loads((root / "releases" / "no-reuse" / "receipt.json").read_text())
        assert not no_reuse["cache_hit"]
        for release, extra in (("bad-steps", ("--steps", "1000001")),
                               ("bad-name!", ()), ("too-many", ("--lines", "900"))):
            check(*base, "--release", release, *extra, success=False)
    print("PASS: 1,000,000 geometric iterations; 3D predictor; runtime oracle; cache reuse;")
    print("      context invalidation; tamper miss; negative ROI; rejected-code rollback; bounded CLI")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "g++")
