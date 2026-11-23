#!/usr/bin/env python3
"""Jarvis-X Nano3D one-shot: geometric precomputation, source, independent semantic tests, atomic admission.

Bounded synthetic proof. Never execute untrusted C++ or treat template tests as general-purpose correctness.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
VERSION = "jx-nano3d-one-shot.v1"
UINT32 = (1 << 32) - 1
EXPR = ("a + b", "a ^ b", "a | b", "a & b", "a - b", "a * 3u + b", "(a << 1u) ^ b", "(a >> 1u) + b")
FUNC = re.compile(r"constexpr unsigned jx_(\d+)_(\d+)\(unsigned a,unsigned b\) noexcept \{ return (.+); \}\Z")
TEST_INPUTS = ((0, 0), (1, UINT32), (0xDEADBEEF, 0xC0FFEE03), (UINT32, UINT32))


def digest(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def load_digest(path: Path) -> str:
    return digest(path.read_bytes())


def run_command(argv: list[str], timeout: int = 120, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode:
        raise RuntimeError(f"command failed ({proc.returncode}): {argv[0]}\n{proc.stderr[-2000:]}\n{proc.stdout[-300:]}")
    return proc


def xorshift(x: int) -> int:
    x ^= (x << 13) & UINT32
    x ^= x >> 17
    x ^= (x << 5) & UINT32
    return x & UINT32


def expected_class(f: int, line: int) -> int:
    state = (0x9E3779B9 ^ ((f * 131071 + line * 8191) & UINT32)) & UINT32
    coords = []
    for _ in range(3):
        state = xorshift(state)
        coords.append(state & 31)
    return ((coords[0] >= 16) << 2) | ((coords[1] >= 16) << 1) | (coords[2] >= 16)


def expected_value(k: int, a: int, b: int) -> int:
    v = (a+b, a^b, a|b, a&b, a-b, a*3+b, (a << 1)^b, (a >> 1)+b)[k]
    return v & UINT32


def audit_and_make_driver(stage: Path, files: int, lines: int) -> tuple[dict[str, str], int]:
    source = stage / "source"
    driver = stage / "verify_runtime.cpp"
    parts = ["// Independent C++ behavioral check against Python uint32 oracle.", "#include <cstdint>"]
    for f in range(files):
        parts.append(f'#include "source/generated_{f:05d}.cpp"')
    parts.append("int main() {")
    digests: dict[str, str] = {}
    checked = 0
    for f in range(files):
        path = source / f"generated_{f:05d}.cpp"
        b = path.read_bytes()
        digests[path.name] = digest(b)
        text = b.decode("utf-8")
        entries = text.splitlines()
        if len(entries) != lines or b.count(b"\n") != lines:
            raise ValueError(f"line count mismatch in {path.name}")
        for j, s in enumerate(entries):
            m = FUNC.fullmatch(s)
            if not m or int(m.group(1)) != f or int(m.group(2)) != j:
                raise ValueError(f"candidate format/symbol rejected at {f}:{j}")
            k = expected_class(f, j)
            if m.group(3) != EXPR[k]:
                raise ValueError(f"ANN prediction vs independent oracle mismatch at {f}:{j}")
            for a, b_value in TEST_INPUTS:
                # volatile inputs force runtime evaluation instead of solely constexpr checks.
                parts.append(f"  {{ volatile unsigned a=0x{a:08x}u, b=0x{b_value:08x}u; ")
                parts.append(f"    if (jx_{f}_{j}(a,b) != 0x{expected_value(k,a,b_value):08x}u) return 1; }}")
                checked += 1
    parts += ["  return 0;", "}"]
    driver.write_text("\n".join(parts) + "\n", encoding="utf-8")
    return digests, checked


def cache_copy(cache_path: Path, source: Path, files: int) -> bool:
    try:
        manifest = json.loads((cache_path / "manifest.json").read_text())
        names = [f"generated_{i:05d}.cpp" for i in range(files)]
        if set(manifest["sha256"]) != set(names):
            return False
        for n in names:
            if not (cache_path / "source" / n).is_file() or load_digest(cache_path / "source" / n) != manifest["sha256"][n]:
                return False
        shutil.copytree(cache_path / "source", source)
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def publish_cache(root: Path, cache_key: str, sources: Path, hashes: dict[str, str]) -> None:
    cache_dir = root / "prediction-cache"
    cache_dir.mkdir(exist_ok=True)
    published = cache_dir / cache_key
    if published.exists():
        return
    stage = Path(tempfile.mkdtemp(dir=cache_dir, prefix=".pending_"))
    try:
        shutil.copytree(sources, stage / "source")
        (stage / "manifest.json").write_text(json.dumps({"sha256": hashes}, sort_keys=True) + "\n")
        os.replace(stage, published)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def one_shot(*, root: Path, release: str, task_version: str, files: int, lines: int,
             steps: int, compiler: str, inject_invalid: bool, expected_reuses: int,
             valid_probability: float, fresh_ms: float, lookup_ms: float,
             precompute_ms: float, invalidation_ms: float) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", release):
        raise ValueError("invalid release identifier")
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", task_version):
        raise ValueError("invalid task version")
    if not(1 <= files <= 40 and 1 <= lines <= 850 and files * lines <= 8000):
        raise ValueError("demo limit 40 files, 850 lines/file, 8000 LOC total")
    if not(1 <= steps <= 1000000) or expected_reuses < 0:
        raise ValueError("iteration/reuse bounds violated")
    if not(0 <= valid_probability <= 1) or min(fresh_ms, lookup_ms, precompute_ms, invalidation_ms) < 0:
        raise ValueError("invalid cost scenario")
    os.umask(0o077)
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = root / ".one_shot.lock"
    with lock.open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        published = root / "releases" / release
        if published.exists():
            raise FileExistsError("release already exists")
        (root / "releases").mkdir(exist_ok=True)
        start = time.perf_counter()
        stage = Path(tempfile.mkdtemp(prefix=".candidate_", dir=root))
        try:
            engine_src = HERE / "nano3d.cpp"
            geometry_src = HERE / "sphere_optimizer.cpp"
            engine_hash = load_digest(engine_src)
            geometry_hash = load_digest(geometry_src)
            compiler_version = run_command([compiler, "--version"], timeout=10).stdout.splitlines()[0]
            manifest = {"schema": VERSION, "engine_sha256": engine_hash,
                        "geometry_sha256": geometry_hash, "compiler_version": compiler_version,
                        "task_version": task_version, "files": files, "lines": lines,
                        "steps": steps, "test_inputs": TEST_INPUTS}
            cache_key = digest(json.dumps(manifest, sort_keys=True).encode())
            quality_gain_ms = expected_reuses * (valid_probability * (fresh_ms - lookup_ms)
                    - (1-valid_probability) * invalidation_ms) - precompute_ms
            allow_reuse = quality_gain_ms > 0
            cpp_flags = ["-O2", "-std=c++17", "-Wall", "-Wextra", "-Wpedantic",
                         "-Wconversion", "-Wshadow", "-Werror"]
            engine = stage / "nano3d"
            geom = stage / "geometry"
            run_command([compiler, *cpp_flags, str(engine_src), "-o", str(engine)])
            run_command([compiler, *cpp_flags, str(geometry_src), "-o", str(geom)])
            geometry = json.loads(run_command([str(geom), str(steps)], timeout=120).stdout)
            if not geometry["verified"] or geometry["iterations"] != steps:
                raise RuntimeError("geometry verification failed")
            source = stage / "source"
            reused = allow_reuse and cache_copy(root / "prediction-cache" / cache_key, source, files)
            if not reused:
                output = run_command([str(engine), "--files", str(files), "--lines", str(lines),
                                      "--loops", "10000", "--out", str(source)], timeout=120)
                if "trained_correct=32768/32768" not in output.stdout:
                    raise RuntimeError("incomplete synthetic precomputed lattice")
            if inject_invalid:
                file0 = source / "generated_00000.cpp"
                s = file0.read_text()
                file0.write_text("int deliberately invalid C++;\n" + s.split("\n", 1)[1])
            hashes, behavior_count = audit_and_make_driver(stage, files, lines)
            for f in range(files):
                run_command([compiler, "-std=c++17", "-fsyntax-only", str(source / f"generated_{f:05d}.cpp")], timeout=120)
            binary = stage / "verify_runtime"
            run_command([compiler, "-std=c++17", "-O0", str(stage / "verify_runtime.cpp"), "-o", str(binary)], timeout=120)
            run_command([str(binary)], timeout=120)
            elapsed = time.perf_counter() - start
            receipt = {"schema": VERSION, "release": release, "task_version": task_version,
                       "cache_key": cache_key, "cache_hit": reused, "estimated_reuse_gain_ms": quality_gain_ms,
                       "cost_scenario_is_measured": False,
                       "scope": "synthetic octant classifier, typed function templates, independent oracle, compiler syntax and sampled runtime tests",
                       "geometry": geometry, "files": files, "lines": files * lines,
                       "checked_runtime_cases": behavior_count, "sha256": hashes,
                       "compiler": compiler_version, "elapsed_seconds_to_acceptance": elapsed}
            (stage / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
            binary.unlink()
            (stage / "verify_runtime.cpp").unlink()
            engine.unlink()
            geom.unlink()
            os.replace(stage, published)
            pointer = root / f".latest_{release}.json"
            pointer.write_text(json.dumps({"release": release,
                                           "receipt_sha256": load_digest(published / "receipt.json")}, sort_keys=True) + "\n")
            os.replace(pointer, root / "latest.json")
            # Cache is a hint; independent syntax and behavioral checks are repeated on every hit.
            publish_cache(root, cache_key, published / "source", hashes)
            return receipt
        except BaseException:
            if stage.exists():
                shutil.rmtree(stage)
            raise


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("nano3d-output"))
    p.add_argument("--release", required=True)
    p.add_argument("--task-version", default="demo-v1")
    p.add_argument("--files", type=int, default=4)
    p.add_argument("--lines", type=int, default=128)
    p.add_argument("--steps", type=int, default=1000000)
    p.add_argument("--compiler", default="g++")
    p.add_argument("--inject-invalid", action="store_true")
    p.add_argument("--expected-reuses", type=int, default=5)
    p.add_argument("--valid-probability", type=float, default=0.95)
    p.add_argument("--fresh-ms", type=float, default=100.0)
    p.add_argument("--lookup-ms", type=float, default=1.0)
    p.add_argument("--precompute-ms", type=float, default=300.0)
    p.add_argument("--invalidation-ms", type=float, default=5.0)
    a = p.parse_args()
    try:
        result = one_shot(root=a.root, release=a.release, task_version=a.task_version,
                          files=a.files, lines=a.lines, steps=a.steps, compiler=a.compiler,
                          inject_invalid=a.inject_invalid, expected_reuses=a.expected_reuses,
                          valid_probability=a.valid_probability, fresh_ms=a.fresh_ms,
                          lookup_ms=a.lookup_ms, precompute_ms=a.precompute_ms,
                          invalidation_ms=a.invalidation_ms)
        print(json.dumps({k: result[k] for k in ("release", "lines", "checked_runtime_cases", "cache_hit", "estimated_reuse_gain_ms", "elapsed_seconds_to_acceptance")}, indent=2))
        print("ACCEPTED: independent source oracle, full syntax, behavioral execution, receipt and commit")
        return 0
    except Exception as e:
        print("REJECTED:", e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
