"""Read-only verifier and explicit external runner for DM3D v2 DOCX archives.

Public interoperability adapter. Does NOT embed, import, or execute DOCX code.
Private runtimes are supplied as separate, user-approved local files.
Python >= 3.10; standard library only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import zipfile

MAX_ENTRIES = 3000
MAX_COMPRESSED = 64 * 1024 * 1024
MAX_MEMBER = 32 * 1024 * 1024
MAX_TOTAL_UNCOMPRESSED = 256 * 1024 * 1024
CORE = (
    "dm3d/runtime.py", "dm3d/runtime.sha256", "dm3d/programs.json",
    "dm3d/model.json", "dm3d/volumes.json", "dm3d/assets.json",
)
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
KEY_RE = re.compile(r"^[0-9a-f]{30}$")


class InvalidPackage(ValueError):
    """The document failed the bounded format/integrity contract."""


def inspect_document(document: str | Path) -> dict:
    """Inspect archive metadata with strict bounds; never run embedded source."""
    path = Path(document)
    if not path.is_file() or path.stat().st_size > MAX_COMPRESSED:
        raise InvalidPackage("DOCX absent or exceeds size cap")
    try:
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            if len(entries) > MAX_ENTRIES:
                raise InvalidPackage("too many ZIP entries")
            seen = set()
            total = 0
            for entry in entries:
                name = entry.filename
                pure = PurePosixPath(name)
                if (name in seen or name.startswith("/") or "\\" in name
                        or ".." in pure.parts or not name or "\x00" in name):
                    raise InvalidPackage("duplicate or invalid ZIP path")
                seen.add(name)
                if entry.file_size > MAX_MEMBER or entry.compress_size > MAX_COMPRESSED:
                    raise InvalidPackage("ZIP member exceeds size cap")
                if entry.file_size and entry.compress_size == 0:
                    raise InvalidPackage("invalid compressed-size declaration")
                if entry.compress_size and entry.file_size > 1000 * entry.compress_size:
                    raise InvalidPackage("excessive ZIP compression ratio")
                total += entry.file_size
                if total > MAX_TOTAL_UNCOMPRESSED:
                    raise InvalidPackage("ZIP expanded size cap exceeded")
            if not set(CORE).issubset(seen):
                raise InvalidPackage("missing mandatory DM3D v2 members")

            source = archive.read("dm3d/runtime.py")
            digest = archive.read("dm3d/runtime.sha256").decode("ascii").strip()
            if not SHA_RE.fullmatch(digest) or hashlib.sha256(source).hexdigest() != digest:
                raise InvalidPackage("embedded runtime digest mismatch")

            def read_json(name: str):
                value = json.loads(archive.read(name))
                if not isinstance(value, dict):
                    raise InvalidPackage("invalid JSON root: " + name)
                return value

            programs = read_json("dm3d/programs.json")
            model = read_json("dm3d/model.json")
            volumes = read_json("dm3d/volumes.json")
            assets = read_json("dm3d/assets.json")
            if not programs or not all(isinstance(k, str) and isinstance(v, list)
                                       for k, v in programs.items()):
                raise InvalidPackage("invalid program manifest")
            arch = model.get("architecture")
            if (not isinstance(arch, list) or len(arch) != 3 or
                    any(type(n) is not int or n <= 0 for n in arch)):
                raise InvalidPackage("invalid ANN architecture")
            ram, rom = volumes.get("ram"), volumes.get("rom")
            if not isinstance(ram, dict) or not isinstance(rom, dict):
                raise InvalidPackage("invalid RAM/ROM structure")
            if len(ram) > 1024 or len(rom) > 1024:
                raise InvalidPackage("too many materialized tiles")
            if not all(KEY_RE.fullmatch(k) for k in (*ram, *rom)):
                raise InvalidPackage("invalid Morton key")
            if type(volumes.get("rom_sealed")) is not bool:
                raise InvalidPackage("ROM seal state missing")
            objects = assets.get("objects", {})
            if not isinstance(objects, dict):
                raise InvalidPackage("invalid asset manifest")
            return {
                "format": "dm3d-docx-v2", "source_sha256": digest,
                "programs": sorted(programs), "ann_architecture": arch,
                "ram_tiles": len(ram), "rom_tiles": len(rom),
                "rom_sealed": volumes["rom_sealed"],
                "asset_count": len(objects), "validated_zip_entries": len(entries),
            }
    except (zipfile.BadZipFile, UnicodeError, json.JSONDecodeError, KeyError,
            OSError) as exc:
        raise InvalidPackage("unable to validate DOCX: " + str(exc)) from exc


def run_approved_runtime(document: str | Path, runtime: str | Path,
                         program: str, output: str | Path, *,
                         approved: bool = False, mp4: bool = False,
                         timeout_seconds: int = 120) -> subprocess.CompletedProcess[str]:
    """Run only an explicit trusted *external* Python file, never DOCX code."""
    if not approved:
        raise PermissionError("explicit --approve-external-runtime required")
    if not isinstance(program, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", program):
        raise ValueError("unsafe program name")
    summary = inspect_document(document)
    if program not in summary["programs"]:
        raise ValueError("program not present in DOCX manifest")
    runner = Path(runtime).resolve(strict=True)
    if not runner.is_file() or runner.suffix != ".py":
        raise ValueError("runner must be a local Python source file")
    cmd = [sys.executable, str(runner), "run", str(Path(document).resolve()),
           program, "--out", str(Path(output).resolve())]
    if mp4:
        cmd.append("--mp4")
    return subprocess.run(cmd, capture_output=True, text=True,
                          check=False, timeout=timeout_seconds)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    inspect = sub.add_parser("inspect", help="validate a v2 DOCX without executing it")
    inspect.add_argument("document")
    run = sub.add_parser("run", help="execute only an approved external runtime")
    run.add_argument("document")
    run.add_argument("program")
    run.add_argument("--runtime", required=True)
    run.add_argument("--out", required=True)
    run.add_argument("--approve-external-runtime", action="store_true")
    run.add_argument("--mp4", action="store_true")
    opts = parser.parse_args(argv)
    try:
        if opts.action == "inspect":
            print(json.dumps(inspect_document(opts.document), sort_keys=True, indent=2))
            return 0
        proc = run_approved_runtime(opts.document, opts.runtime, opts.program,
                                    opts.out, approved=opts.approve_external_runtime,
                                    mp4=opts.mp4)
        sys.stdout.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        return proc.returncode
    except (InvalidPackage, PermissionError, ValueError, FileNotFoundError,
            subprocess.TimeoutExpired) as exc:
        parser.exit(2, "DM3D adapter: " + str(exc) + "\n")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
