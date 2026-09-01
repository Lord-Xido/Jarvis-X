"""Command-line interface for the Jarvis-X universal bitcode runtime."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

from .universal_bitcode import (
    DEFAULT_CHUNK_SIZE,
    BitcodeError,
    MediaKind,
    RepresentationContract,
    UniversalBitcodeRuntime,
    detect_contract,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jarvisx-universal-bitcode",
        description="Compile arbitrary typed bytes into deterministic Jarvis-X bitcode IR.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    encode = subcommands.add_parser("encode", help="encode a file into a .jxbi container")
    _add_input_output(encode)
    _add_contract_options(encode)

    cycle = subcommands.add_parser(
        "cycle", help="encode, decode, verify, and confirm the canonical fixed point"
    )
    _add_input_output(cycle)
    _add_contract_options(cycle)

    decode = subcommands.add_parser("decode", help="decode and verify a .jxbi container")
    _add_input_output(decode)

    inspect = subcommands.add_parser(
        "inspect", help="validate and print the typed manifest without decompression"
    )
    inspect.add_argument("input", type=Path)

    verify = subcommands.add_parser("verify", help="fully reconstruct and verify a container")
    verify.add_argument("input", type=Path)
    return parser


def _add_input_output(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--force", action="store_true", help="replace an existing output atomically"
    )


def _add_contract_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--source-name")
    parser.add_argument("--media-kind", choices=[kind.value for kind in MediaKind])
    parser.add_argument("--media-type")
    parser.add_argument("--format-name")
    parser.add_argument("--schema")
    parser.add_argument(
        "--metadata",
        default="{}",
        help="JSON object merged into inferred detection metadata",
    )


def _read_bounded(path: Path, runtime: UniversalBitcodeRuntime, *, container: bool) -> bytes:
    maximum = runtime.budget.max_input_bytes
    if container:
        maximum += runtime.budget.max_manifest_bytes + 1024
    size = path.stat().st_size
    if size > maximum:
        raise ValueError(f"input file size {size} exceeds CLI budget {maximum}")
    data = path.read_bytes()
    if len(data) != size:
        raise OSError("input file changed while it was being read")
    return data


def _metadata(value: str) -> dict[str, Any]:
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("--metadata must decode to a JSON object")
    return parsed


def _contract(args: argparse.Namespace, data: bytes) -> RepresentationContract:
    source_name = args.source_name if args.source_name is not None else args.input.name
    inferred = detect_contract(data, source_name=source_name)
    metadata = dict(inferred.metadata)
    metadata.update(_metadata(args.metadata))
    return RepresentationContract(
        media_kind=MediaKind(args.media_kind) if args.media_kind else inferred.media_kind,
        media_type=args.media_type or inferred.media_type,
        format_name=args.format_name or inferred.format_name,
        source_name=source_name,
        schema=args.schema,
        metadata=metadata,
    )


def _atomic_write(path: Path, data: bytes, *, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"output exists (use --force to replace): {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass


def _distinct_paths(input_path: Path, output_path: Path) -> None:
    if input_path.resolve() == output_path.resolve():
        raise ValueError("input and output paths must be different")


def _emit(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _run(args: argparse.Namespace, runtime: UniversalBitcodeRuntime) -> int:
    if args.command in {"encode", "cycle"}:
        _distinct_paths(args.input, args.output)
        data = _read_bounded(args.input, runtime, container=False)
        contract = _contract(args, data)
        if args.command == "encode":
            container = runtime.encode(
                data,
                contract=contract,
                chunk_size=args.chunk_size,
            )
            _atomic_write(args.output, container, force=args.force)
            report = runtime.verify(container).as_dict()
            report["output"] = str(args.output)
            _emit(report)
            return 0
        receipt = runtime.close_loop(
            data,
            contract=contract,
            chunk_size=args.chunk_size,
        )
        _atomic_write(args.output, receipt.container, force=args.force)
        payload = receipt.as_dict()
        payload["output"] = str(args.output)
        _emit(payload)
        return 0

    container = _read_bounded(args.input, runtime, container=True)
    if args.command == "inspect":
        _emit(runtime.inspect(container).as_dict())
        return 0
    if args.command == "verify":
        _emit(runtime.verify(container).as_dict())
        return 0
    if args.command == "decode":
        _distinct_paths(args.input, args.output)
        decoded = runtime.decode(container)
        _atomic_write(args.output, decoded.data, force=args.force)
        report = runtime.verify(container).as_dict()
        report["output"] = str(args.output)
        _emit(report)
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        return _run(args, UniversalBitcodeRuntime())
    except (BitcodeError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
