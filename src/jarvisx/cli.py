import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence, Tuple

from .api import start_api
from .assembler import Assembler
from .chrysalis_rom import (
    ChrysalisROM,
    ROMError,
    mutate_immediate,
    rom_from_source,
    run_rom,
    words_from_rom,
)
from .core import CodexVM
from .node import CodexNode
from .parser import Parser
from .web import start_web


def _grid_shape(value: str) -> Tuple[int, int, int]:
    normalized = value.lower().replace(",", "x")
    parts = normalized.split("x")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("grid must be formatted as XxYxZ")
    try:
        shape = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("grid dimensions must be integers") from exc
    if any(dimension < 1 or dimension > 0xFFFF for dimension in shape):
        raise argparse.ArgumentTypeError("grid dimensions must be in the range 1..65535")
    return shape[0], shape[1], shape[2]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jarvisx",
        description="Jarvis-X assembly VM and deterministic Chrysalis 3D byte-ROM runtime.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    run_command = commands.add_parser("run", help="assemble and run a Jarvis-X source file")
    run_command.add_argument("file", type=Path)

    commands.add_parser("api", help="start the API service")
    commands.add_parser("web", help="start the web service")
    commands.add_parser("node", help="start a Jarvis-X node")

    rom_command = commands.add_parser("rom", help="operate the Chrysalis 3D byte-ROM")
    rom_commands = rom_command.add_subparsers(dest="rom_command", required=True)

    encode = rom_commands.add_parser("encode", help="assemble source into a ROM image")
    encode.add_argument("source", type=Path)
    encode.add_argument("output", type=Path)
    encode.add_argument("--engines", type=int, default=1)
    encode.add_argument("--grid", type=_grid_shape, default=(4, 4, 4), metavar="XxYxZ")
    encode.add_argument("--cell-bytes", type=int)
    encode.add_argument("--compress", action="store_true")

    inspect = rom_commands.add_parser("inspect", help="verify and describe a ROM image")
    inspect.add_argument("rom", type=Path)

    verify = rom_commands.add_parser("verify", help="verify ROM and bytecode integrity")
    verify.add_argument("rom", type=Path)

    extract = rom_commands.add_parser("extract", help="extract the packed bytecode image")
    extract.add_argument("rom", type=Path)
    extract.add_argument("output", type=Path)

    execute = rom_commands.add_parser("run", help="verify and execute a ROM image")
    execute.add_argument("rom", type=Path)

    mutate = rom_commands.add_parser(
        "mutate",
        help="create a bounded candidate by changing one SET immediate",
    )
    mutate.add_argument("rom", type=Path)
    mutate.add_argument("output", type=Path)
    mutate.add_argument("--word-index", type=int, required=True)
    mutate.add_argument("--delta", type=int, required=True)

    return parser


def _run_source(path: Path) -> CodexVM:
    source = path.read_text(encoding="utf-8")
    ast = Parser().parse(source)
    bytecode = Assembler().assemble(ast)
    vm = CodexVM()
    vm.load(bytecode)
    vm.run()
    return vm


def _print_rom_stats(rom: ChrysalisROM, word_count: Optional[int] = None) -> None:
    stats = rom.statistics()
    if word_count is not None:
        stats["bytecode_words"] = word_count
    print(json.dumps(stats, indent=2, sort_keys=True))


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "run":
            vm = _run_source(args.file)
            print("Registers:", vm.regs.snapshot())
            return 0

        if args.command == "api":
            start_api()
            return 0

        if args.command == "web":
            start_web()
            return 0

        if args.command == "node":
            CodexNode().start()
            return 0

        if args.rom_command == "encode":
            source = args.source.read_text(encoding="utf-8")
            rom = rom_from_source(
                source,
                engines=args.engines,
                grid_shape=args.grid,
                cell_bytes=args.cell_bytes,
                compress=args.compress,
            )
            rom.write(args.output)
            _print_rom_stats(rom, word_count=len(words_from_rom(rom)))
            return 0

        rom = ChrysalisROM.read(args.rom)

        if args.rom_command == "inspect":
            words = words_from_rom(rom)
            _print_rom_stats(rom, word_count=len(words))
            return 0

        if args.rom_command == "verify":
            words = words_from_rom(rom)
            print("verified: {} bytecode words".format(len(words)))
            return 0

        if args.rom_command == "extract":
            payload = rom.payload(verify=True)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_bytes(payload)
            print("extracted {} bytes to {}".format(len(payload), args.output))
            return 0

        if args.rom_command == "run":
            vm = run_rom(rom)
            print("Registers:", vm.regs.snapshot())
            return 0

        if args.rom_command == "mutate":
            candidate = mutate_immediate(
                rom,
                word_index=args.word_index,
                delta=args.delta,
            )
            candidate.write(args.output)
            _print_rom_stats(candidate, word_count=len(words_from_rom(candidate)))
            return 0

        raise RuntimeError("unreachable command state")
    except (OSError, ROMError, ValueError, IndexError, TypeError) as exc:
        print("jarvisx: {}".format(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
