"""CLI for the Dr Moagi verified 1 GiB firmware container."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from .dr_moagi_firmware import (
    FirmwareBuilder,
    FirmwareImage,
    build_reference_riscv_elf,
    generate_aes256_key,
    generate_ed25519_keypair,
    inspect_riscv_elf,
)
from .dr_moagi_os import demo_field


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jarvisx-dr-moagi-firmware",
        description=(
            "Build, inspect, verify and boot the exact-size DMLAMBDA 1 GiB firmware container."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    keys = sub.add_parser("keygen", help="Generate Ed25519 signing and AES-256 encryption keys")
    keys.add_argument("prefix", type=Path)
    keys.add_argument("--pretty", action="store_true")

    build_demo = sub.add_parser("build-demo", help="Build a signed/encrypted demo firmware image")
    build_demo.add_argument("output", type=Path)
    build_demo.add_argument("--side", type=int, default=16)
    _add_build_keys(build_demo)
    build_demo.add_argument("--supervisor", type=Path)
    build_demo.add_argument("--pretty", action="store_true")

    build = sub.add_parser("build", help="Build firmware from a sparse state JSON file")
    build.add_argument("state", type=Path)
    build.add_argument("output", type=Path)
    build.add_argument("--side", type=int, required=True)
    build.add_argument("--metric", type=Path)
    build.add_argument("--supervisor", type=Path)
    _add_build_keys(build)
    build.add_argument("--pretty", action="store_true")

    inspect = sub.add_parser("inspect", help="Inspect container metadata without booting it")
    inspect.add_argument("image", type=Path)
    inspect.add_argument("--pretty", action="store_true")

    verify = sub.add_parser("verify", help="Verify manifest, signature, encryption and sections")
    verify.add_argument("image", type=Path)
    _add_verify_keys(verify)
    verify.add_argument("--pretty", action="store_true")

    run = sub.add_parser("run", help="Verified-boot the image and execute bounded autonomic cycles")
    run.add_argument("image", type=Path)
    run.add_argument("--cycles", type=int, default=1)
    run.add_argument("--max-active-cells", type=int, default=50_000)
    _add_verify_keys(run)
    run.add_argument("--pretty", action="store_true")

    serve = sub.add_parser("serve", help="Serve a verified firmware image through FastAPI")
    serve.add_argument("image", type=Path)
    serve.add_argument("--public-key", type=Path)
    serve.add_argument("--encryption-key", type=Path)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=10002)

    elf = sub.add_parser("reference-elf", help="Write the valid RV64 reference monitor ELF")
    elf.add_argument("output", type=Path)
    elf.add_argument("--pretty", action="store_true")
    return parser


def _add_build_keys(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--signing-private-key", type=Path)
    parser.add_argument("--encryption-key", type=Path)


def _add_verify_keys(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--public-key", type=Path)
    parser.add_argument("--encryption-key", type=Path)


def _read_optional(path: Path | None) -> bytes | None:
    return None if path is None else path.read_bytes()


def _load_field(path: Path) -> dict[tuple[int, int, int], float]:
    raw: Any = json.loads(path.read_text(encoding="utf-8"))
    rows = raw.get("field") if isinstance(raw, dict) else raw
    if not isinstance(rows, list) or not rows:
        raise ValueError("state JSON must contain a non-empty list or {'field': [...]} object")
    result: dict[tuple[int, int, int], float] = {}
    for row in rows:
        if isinstance(row, dict):
            coordinate = int(row["x"]), int(row["y"]), int(row["z"])
            value = float(row["value"])
        elif isinstance(row, list) and len(row) == 4:
            coordinate = int(row[0]), int(row[1]), int(row[2])
            value = float(row[3])
        else:
            raise ValueError("state rows must be {x,y,z,value} or [x,y,z,value]")
        if coordinate in result:
            raise ValueError(f"duplicate state coordinate: {coordinate}")
        result[coordinate] = value
    return result


def _load_metric(path: Path) -> dict[tuple[int, int, int], tuple[float, ...]]:
    raw: Any = json.loads(path.read_text(encoding="utf-8"))
    rows = raw.get("metric") if isinstance(raw, dict) else raw
    if not isinstance(rows, list):
        raise ValueError("metric JSON must contain a list or {'metric': [...]} object")
    result: dict[tuple[int, int, int], tuple[float, ...]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("metric rows must be objects")
        coordinate = int(row["x"]), int(row["y"]), int(row["z"])
        components = tuple(
            float(row[name]) for name in ("gxx", "gyy", "gzz", "gxy", "gxz", "gyz")
        )
        if coordinate in result:
            raise ValueError(f"duplicate metric coordinate: {coordinate}")
        result[coordinate] = components
    return result


def _print(payload: object, pretty: bool) -> None:
    print(json.dumps(payload, indent=2 if pretty else None, sort_keys=True))


def main() -> int:
    args = _parser().parse_args()

    if args.command == "keygen":
        private, public = generate_ed25519_keypair()
        aes = generate_aes256_key()
        args.prefix.parent.mkdir(parents=True, exist_ok=True)
        private_path = Path(str(args.prefix) + ".ed25519.private")
        public_path = Path(str(args.prefix) + ".ed25519.public")
        aes_path = Path(str(args.prefix) + ".aes256")
        private_path.write_bytes(private)
        public_path.write_bytes(public)
        aes_path.write_bytes(aes)
        _print(
            {
                "private_key": str(private_path),
                "public_key": str(public_path),
                "encryption_key": str(aes_path),
                "warning": "keep private and AES keys outside firmware images and source control",
            },
            args.pretty,
        )
        return 0

    if args.command == "reference-elf":
        payload = build_reference_riscv_elf()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(payload)
        _print(
            {"path": str(args.output), "bytes": len(payload), **inspect_riscv_elf(payload)},
            args.pretty,
        )
        return 0

    if args.command in ("build", "build-demo"):
        state = demo_field(args.side) if args.command == "build-demo" else _load_field(args.state)
        metric = None
        if args.command == "build" and args.metric is not None:
            metric = _load_metric(args.metric)
        supervisor = _read_optional(args.supervisor)
        report = FirmwareBuilder().build(
            args.output,
            state=state,
            side=args.side,
            metric=metric,
            supervisor_elf=supervisor,
            signing_private_key=_read_optional(args.signing_private_key),
            encryption_key=_read_optional(args.encryption_key),
        )
        _print(report, args.pretty)
        return 0

    if args.command == "serve":
        os.environ["JARVISX_FIRMWARE_IMAGE"] = str(args.image)
        if args.public_key is not None:
            os.environ["JARVISX_FIRMWARE_PUBLIC_KEY"] = str(args.public_key)
        if args.encryption_key is not None:
            os.environ["JARVISX_FIRMWARE_ENCRYPTION_KEY"] = str(args.encryption_key)
        import uvicorn

        uvicorn.run("jarvisx.dr_moagi_firmware_api:app", host=args.host, port=args.port)
        return 0

    image = FirmwareImage(args.image)
    if args.command == "inspect":
        _print(
            {
                "path": str(args.image),
                "logical_size_bytes": args.image.stat().st_size,
                "signed": image.signed,
                "encrypted": image.encrypted,
                "manifest": image.manifest,
            },
            args.pretty,
        )
        return 0

    public = _read_optional(args.public_key)
    encryption = _read_optional(args.encryption_key)
    if args.command == "verify":
        _print(image.verify(public_key=public, encryption_key=encryption).as_dict(), args.pretty)
        return 0
    if args.command == "run":
        session = image.boot(
            public_key=public,
            encryption_key=encryption,
            max_active_cells=args.max_active_cells,
        )
        report = session.run(args.cycles)
        _print(
            {
                "run": report.as_dict(),
                "status": session.architecture.status(),
                "trace_head": session.trace_head,
            },
            args.pretty,
        )
        return 0
    raise AssertionError("unreachable command")


if __name__ == "__main__":
    raise SystemExit(main())
