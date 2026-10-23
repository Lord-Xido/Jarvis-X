#!/usr/bin/env python3
"""Public, read-only interoperability bridge for an external DM3D 1000-op DOCX VM.

Does not bundle the proprietary encoder, decoder, document or sample payload.
Explicitly approved execution delegates to a separately provided trusted local runtime.
"""
from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import struct
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile

PART = "customXml/item2.xml"
REQUIRED_OOXML = frozenset({"[Content_Types].xml", "_rels/.rels", "word/document.xml"})
NS = "urn:dm-pcs:3d-docx-1000bytecode:v2"
MAX_ZIP = 16 * 1024 * 1024
MAX_XML = 2 * 1024 * 1024
MAX_ENTRY = 8 * 1024 * 1024
MAX_EXPANDED = 32 * 1024 * 1024
SHA = re.compile(r"[0-9a-f]{64}\Z")


class InvalidPackage(ValueError):
    """Malformed or unsupported DOCX container."""


def inspect_document(path: str | Path) -> dict:
    """Validate bounded layout, bytecode frame checksums and declared digests.

    Checks are for corruption, not authentication of a maliciously rewritten file.
    Neither embedded payload nor bytecode instructions are executed.
    """
    path = Path(path)
    if not path.is_file() or path.stat().st_size > MAX_ZIP:
        raise InvalidPackage("missing or oversized DOCX")
    try:
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            if len(entries) > 2048:
                raise InvalidPackage("too many ZIP entries")
            seen = set()
            total = 0
            for entry in entries:
                name = entry.filename
                posix = PurePosixPath(name)
                if (not name or name in seen or name.startswith("/") or "\\" in name
                        or ".." in posix.parts or "\x00" in name or entry.flag_bits & 1):
                    raise InvalidPackage("invalid, duplicate or encrypted ZIP entry")
                seen.add(name)
                if entry.file_size > MAX_ENTRY or entry.compress_size > MAX_ZIP:
                    raise InvalidPackage("ZIP member exceeds limits")
                if entry.file_size and (not entry.compress_size
                                        or entry.file_size > 1000 * entry.compress_size):
                    raise InvalidPackage("ZIP compression ratio exceeds limit")
                total += entry.file_size
                if total > MAX_EXPANDED:
                    raise InvalidPackage("ZIP expansion exceeds limit")
            if not REQUIRED_OOXML.issubset(seen):
                raise InvalidPackage("missing required Word OOXML parts")
            if PART not in seen or archive.getinfo(PART).file_size > MAX_XML:
                raise InvalidPackage("missing or oversized VM custom XML")
            xml = archive.read(PART)
            if b"<!DOCTYPE" in xml or b"<!ENTITY" in xml:
                raise InvalidPackage("XML document type or entity declaration rejected")
            root = ET.fromstring(xml)
        tag = lambda name: f"{{{NS}}}{name}"
        if root.tag != tag("machine") or root.attrib != {"version": "2.0"}:
            raise InvalidPackage("unrecognized VM namespace/version")
        if [child.tag for child in root] != [tag("manifest"), tag("program"), tag("payload")]:
            raise InvalidPackage("VM XML must contain exactly one ordered manifest/program/payload")
        manifest_node = root.find(tag("manifest"))
        program_node = root.find(tag("program"))
        payload_node = root.find(tag("payload"))
        if any(x is None for x in (manifest_node, program_node, payload_node)):
            raise InvalidPackage("incomplete VM metadata")
        manifest = {}
        for field in manifest_node:
            if field.tag != tag("field") or set(field.attrib) != {"name", "value"}:
                raise InvalidPackage("invalid manifest entry")
            key = field.attrib["name"]
            if key in manifest:
                raise InvalidPackage("duplicate manifest key")
            manifest[key] = field.attrib["value"]
        if (manifest.get("format") != "DM3D-1000-VM"
                or manifest.get("version") != "2.0"
                or manifest.get("instructions") != "1000"
                or manifest.get("bytecode_bytes") != "8000"
                or manifest.get("tile_bytes") != "512"):
            raise InvalidPackage("unsupported VM manifest")
        def bounded_number(name: str, limit: int) -> int:
            s = manifest.get(name, "")
            if not re.fullmatch(r"0|[1-9][0-9]{0,9}", s) or int(s) > limit:
                raise InvalidPackage("invalid " + name)
            return int(s)
        source_bytes = bounded_number("length", 1024 * 1024)
        tiles = bounded_number("tiles", 2048)
        side = bounded_number("grid_side", 16)
        packed_len = bounded_number("input_size", 2 * 1024 * 1024)
        if tiles != (source_bytes + 511) // 512 or (tiles == 0 and side != 0):
            raise InvalidPackage("invalid tile geometry")
        if tiles and (side < 1 or side & (side - 1) or not ((side // 2) ** 3 < tiles <= side ** 3)):
            raise InvalidPackage("invalid grid geometry")
        for name in ("source_sha256", "input_sha256", "program_sha256"):
            if not SHA.fullmatch(manifest.get(name, "")):
                raise InvalidPackage("invalid digest " + name)
        if program_node.attrib != {"encoding": "hex", "record": "8-byte-pc-op-count-index-crc16"}:
            raise InvalidPackage("unsupported program encoding")
        code = program_node.text or ""
        if len(code) != 16000 or not re.fullmatch(r"[0-9a-f]{16000}", code):
            raise InvalidPackage("invalid 1000-instruction program length")
        machine = bytes.fromhex(code)
        if hashlib.sha256(machine).hexdigest() != manifest["program_sha256"]:
            raise InvalidPackage("program digest mismatch")
        for pc in range(1000):
            rec = machine[pc * 8:(pc + 1) * 8]
            index, _, _, _, crc = struct.unpack(">HBBHH", rec)
            if index != pc or binascii.crc_hqx(rec[:6], 0xffff) != crc:
                raise InvalidPackage("instruction PC/CRC mismatch")
        if payload_node.attrib != {"encoding": "base64+zlib"}:
            raise InvalidPackage("unsupported payload encoding")
        encoded = payload_node.text or ""
        if len(encoded) > 3 * 1024 * 1024:
            raise InvalidPackage("oversized payload text")
        try:
            packed = base64.b64decode(encoded, validate=True)
        except binascii.Error as exc:
            raise InvalidPackage("invalid base64 payload") from exc
        if len(packed) != packed_len or hashlib.sha256(packed).hexdigest() != manifest["input_sha256"]:
            raise InvalidPackage("payload size/digest mismatch")
        return {"format": "DM3D-1000-VM", "instructions": 1000, "bytecode_bytes": 8000,
                "source_bytes": source_bytes, "tiles": tiles, "grid_side": side,
                "program_sha256": manifest["program_sha256"],
                "source_sha256_claimed": manifest["source_sha256"],
                "payload_integrity": "matched stored digest; not authenticated or decoded"}
    except (OSError, zipfile.BadZipFile, ET.ParseError, KeyError) as exc:
        raise InvalidPackage("unable to inspect DOCX: " + str(exc)) from exc


def run_approved_runtime(document: str | Path, runtime: str | Path, output: str | Path,
                         *, approved: bool = False, runtime_sha256: str | None = None,
                         timeout_seconds: int = 120) -> subprocess.CompletedProcess[str]:
    if not approved:
        raise PermissionError("external runtime approval required")
    inspect_document(document)
    executable = Path(runtime).resolve(strict=True)
    if not executable.is_file() or executable.suffix != ".py":
        raise ValueError("runtime must be a local Python source file")
    if runtime_sha256 is not None:
        if not SHA.fullmatch(runtime_sha256) or hashlib.sha256(executable.read_bytes()).hexdigest() != runtime_sha256:
            raise ValueError("external runtime SHA-256 mismatch")
    cmd = [sys.executable, str(executable), "run", str(Path(document).resolve()),
           "--output", str(Path(output).resolve())]
    return subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=timeout_seconds)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    sub.add_parser("inspect").add_argument("document")
    run = sub.add_parser("run")
    run.add_argument("document")
    run.add_argument("--runtime", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--runtime-sha256")
    run.add_argument("--approve-external-runtime", action="store_true")
    a = p.parse_args(argv)
    try:
        if a.action == "inspect":
            print(json.dumps(inspect_document(a.document), sort_keys=True, indent=2))
            return 0
        result = run_approved_runtime(a.document, a.runtime, a.output,
                        approved=a.approve_external_runtime, runtime_sha256=a.runtime_sha256)
        sys.stdout.write(result.stdout)
        sys.stderr.write(result.stderr)
        return result.returncode
    except (InvalidPackage, PermissionError, ValueError, FileNotFoundError,
            subprocess.TimeoutExpired) as exc:
        p.exit(2, "DM3D 1000 adapter: " + str(exc) + "\n")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
