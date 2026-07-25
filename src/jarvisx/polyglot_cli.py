"""Command-line interface for Jarvis-X polyglot ROM compilation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from .polyglot import PolyglotCompiler


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m jarvisx.polyglot_cli")
    subparsers = parser.add_subparsers(dest="command", required=True)

    compile_parser = subparsers.add_parser("compile", help="compile source to JXROM")
    compile_parser.add_argument("--language", required=True)
    compile_parser.add_argument("--input", required=True, type=Path)
    compile_parser.add_argument("--output", required=True, type=Path)

    decode_parser = subparsers.add_parser("decode", help="decode JXROM to assembly")
    decode_parser.add_argument("--input", required=True, type=Path)
    decode_parser.add_argument("--output", type=Path)

    inspect_parser = subparsers.add_parser("inspect", help="inspect JXROM metadata")
    inspect_parser.add_argument("--input", required=True, type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    compiler = PolyglotCompiler()

    if args.command == "compile":
        source = args.input.read_text(encoding="utf-8")
        compilation = compiler.compile(source, args.language, {"source_path": str(args.input)})
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(compilation.rom)
        print("compiled %d instructions to %s" % (len(compilation.words), args.output))
        return 0

    rom = args.input.read_bytes()
    if args.command == "decode":
        assembly = compiler.decode_to_assembly(rom)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(assembly + "\n", encoding="utf-8")
        else:
            print(assembly)
        return 0

    if args.command == "inspect":
        image = compiler.decode(rom)
        print(
            json.dumps(
                {
                    "language": image.language,
                    "version": image.version,
                    "instruction_count": len(image.words),
                    "source_sha256": image.source_sha256,
                    "payload_sha256": image.payload_sha256,
                    "metadata": image.metadata,
                },
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            )
        )
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
