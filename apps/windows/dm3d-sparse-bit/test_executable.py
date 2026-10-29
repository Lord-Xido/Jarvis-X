"""Independent process acceptance tests; user state is never modified."""

import argparse
import hashlib
import json
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--expected-size", type=int)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="dm3d-acceptance-") as temporary:
        base = Path(temporary)
        exe = base / "DM3D_JARVIS_X.exe"
        shutil.copyfile(args.exe.resolve(), exe)
        data = base / "DM3D_Data"
        data.mkdir()
        examples = Path(__file__).resolve().parent / "examples"
        shutil.copyfile(examples / "demo.asm", data / "demo.asm")
        shutil.copyfile(examples / "demo.txt", base / "DM3D_Demo.txt")
        scratch = base / "work"
        scratch.mkdir()
        checks = []
        created = []

        def check(name, ok):
            if not ok:
                raise AssertionError(name)
            checks.append(name)

        def invoke(*args, stdin=None):
            return subprocess.run(
                [str(exe), *args],
                input=stdin,
                text=True,
                capture_output=True,
                timeout=45,
            )

        def snapshot(active, axis=8_000_000_000_000, iteration=0):
            return (
                b"DM3D\1"
                + axis.to_bytes(6, "big")
                + iteration.to_bytes(4, "big")
                + len(active).to_bytes(8, "big")
                + b"".join(v.to_bytes(6, "big") for p in sorted(active) for v in p)
            )

        def put(name, blob):
            p = data / name
            p.write_bytes(blob)
            created.append(p)
            return p

        check("Windows PE image", exe.open("rb").read(2) == b"MZ")
        if args.expected_size is not None:
            check(
                "Exact requested file length", exe.stat().st_size == args.expected_size
            )
        r = invoke("--self-test")
        check(
            "Packaged executable internal tests",
            r.returncode == 0 and "PASS 3112" in r.stdout,
        )
        r = invoke("--batch", str(base / "DM3D_Demo.txt"))
        check(
            "Full assembly execution snapshot demo",
            r.returncode == 0 and "HALT instructions=8 active=7" in r.stdout,
        )
        expected = {
            (2, 2, 2),
            (1, 2, 2),
            (3, 2, 2),
            (2, 1, 2),
            (2, 3, 2),
            (2, 2, 1),
            (2, 2, 3),
        }
        expected_blob = snapshot(expected, iteration=1)
        check(
            "Canonical bytes match independent Python reference",
            (data / "demo.dm3d").read_bytes() == expected_blob,
        )
        check(
            "SHA256 matches independent Python reference",
            hashlib.sha256(expected_blob).hexdigest() in r.stdout,
        )
        check(
            "Maximum-coordinate Morton exceeds 128-bit range",
            str(
                sum(
                    ((7_999_999_999_999 >> j) & 1) << (3 * j + k)
                    for j in range(43)
                    for k in range(3)
                )
            )
            in r.stdout,
        )

        for name, blob in [
            ("bad_magic", b"xxxxx" + expected_blob[5:]),
            ("truncated", expected_blob[:-1]),
            ("trailing", expected_blob + b"x"),
            ("duplicate", snapshot([(1, 1, 1), (1, 1, 1)])),
            ("axis", snapshot([], axis=8_000_000_000_001)),
            ("iterations", snapshot([], iteration=101)),
        ]:
            p = put("test_" + name + ".dm3d", blob)
            r = invoke("load", p.name)
            check("Reject snapshot " + name, r.returncode == 1)

        for name, payload in [
            ("unknown", b"D3VM" + struct.pack("<i", 1) + bytes([99]) + bytes(31)),
            (
                "reserved",
                b"D3VM" + struct.pack("<i", 1) + bytes([8, 0, 0, 0, 1]) + bytes(27),
            ),
            ("truncated", b"D3VM" + struct.pack("<i", 1) + bytes(12)),
        ]:
            p = put("test_" + name + ".d3b", payload)
            r = invoke("run", p.name)
            check("Reject bytecode " + name, r.returncode == 1)

        for command in [
            ("save", "../escape.dm3d"),
            ("load", "audit.log"),
            ("write", "8000000000000", "0", "0", "1"),
            ("write", "0", "0", "0", "2"),
        ]:
            r = invoke(*command)
            check("Reject invalid operation " + " ".join(command), r.returncode == 1)

        p = put("test_loop.asm", b"MOV R0 1\nJNZ R0 1\nHALT\n")
        r = invoke("assemble", p.name, "test_loop.d3b")
        created.append(data / "test_loop.d3b")
        check("Assemble bounded loop", r.returncode == 0)
        r = invoke("run", "test_loop.d3b")
        check(
            "Loop stopped by instruction quota",
            r.returncode == 1 and "Instruction quota" in r.stderr,
        )

        # Each initial cell has a disjoint seven-cell neighborhood.
        large = snapshot({(i * 4 + 2, 2, 2) for i in range(15000)})
        put("test_quota.dm3d", large)
        r = invoke(stdin="load test_quota.dm3d\nstep\nsave test_after.dm3d\nquit\n")
        created.append(data / "test_after.dm3d")
        check(
            "STEP growth quota rejects before commit",
            "Active cell quota" in r.stdout
            and (data / "test_after.dm3d").read_bytes() == large,
        )

        badbatch = scratch / "invalid_batch.txt"
        badbatch.write_text("write 0 0 0 1\nunknown\nsave should_not_exist.dm3d\n")
        r = invoke("--batch", str(badbatch))
        check(
            "Batch stops on first error",
            r.returncode == 1 and not (data / "should_not_exist.dm3d").exists(),
        )
        check(
            "Audit records success and failure",
            " FAILED " in (data / "audit.log").read_text()
            and " OK " in (data / "audit.log").read_text(),
        )
        for p in created:
            if p.parent == data and p.name.startswith("test_"):
                p.unlink(missing_ok=True)
        report = {
            "passed": len(checks),
            "checks": checks,
            "internal_checks": 3112,
            "exe_bytes": exe.stat().st_size,
            "snapshot_sha256": hashlib.sha256(expected_blob).hexdigest(),
        }
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(
                json.dumps(report, indent=2) + "\n", encoding="utf-8"
            )
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
