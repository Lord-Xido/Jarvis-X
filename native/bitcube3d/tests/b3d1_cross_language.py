#!/usr/bin/env python3
"""Independent Python ↔ native C++ B3D1 binary and GF(2) conformance test."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from jarvisx.bitcube3d_b3d1 import deserialize, from_points  # noqa: E402

VECTORS = ROOT / "native/bitcube3d/tests/fixtures/b3d1-v1.json"


def check(native_executable: Path) -> None:
    vectors = json.loads(VECTORS.read_text(encoding="utf-8"))["vectors"]
    with tempfile.TemporaryDirectory(prefix="b3d1-interoperability-") as tmp:
        folder = Path(tmp)

        def run_script(name: str, lines: list[str]) -> None:
            script = folder / (name + ".b3ds")
            script.write_text("\n".join(lines) + "\n", encoding="utf-8")
            proc = subprocess.run(
                [str(native_executable), "run", str(script)],
                cwd=folder, capture_output=True, text=True, timeout=30, check=False
            )
            if proc.returncode:
                raise AssertionError(
                    f"native execution failed ({name}): {proc.stdout}\n{proc.stderr}"
                )

        for vector in vectors:
            name = vector["name"]
            bits = [tuple(values) for values in vector["bits"]]
            rounds = vector["rounds"]
            python_bytes = from_points(bits, rounds=rounds).serialize()
            python_path = folder / (name + "-python.b3d")
            cpp_path = folder / (name + "-cpp.b3d")
            restored_path = folder / (name + "-restored.b3d")
            python_path.write_bytes(python_bytes)

            script = ["RESET"]
            script += [f"SET {x} {y} {z} 1" for x, y, z in bits]
            if rounds:
                script.append(f"ENCODE {rounds}")
            script.append(f"SAVE {cpp_path}")
            run_script(name + "-cpp-writer", script)

            native_bytes = cpp_path.read_bytes()
            assert native_bytes == python_bytes, f"native/Python B3D1 mismatch: {name}"
            digest = hashlib.sha256(native_bytes).hexdigest()
            assert digest == vector["sha256"], f"golden digest mismatch: {name}"
            assert len(native_bytes) == vector["expected_bytes"]
            parsed = deserialize(native_bytes)
            assert parsed.active_points(inverse_rounds=True) == sorted(set(bits))

            run_script(
                name + "-cpp-reader",
                [
                    f"LOAD {python_path}",
                    f"SAVE {restored_path}",
                ],
            )
            assert restored_path.read_bytes() == python_bytes, f"native load/save mismatch: {name}"

            # Native inverse rounds must recover original occupancy, including sparse edges.
            steps = [f"LOAD {cpp_path}"]
            if rounds:
                steps.append("DECODE")
            steps += [f"ASSERT {x} {y} {z} 1" for x, y, z in bits]
            steps.append("ASSERT 1 1 1 0")
            run_script(name + "-native-inverse", steps)
            print(f"native↔Python B3D1 {name}: PASS bytes={len(native_bytes)} sha256={digest}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: b3d1_cross_language.py PATH_TO_NATIVE_CLI")
    check(Path(sys.argv[1]).resolve())
