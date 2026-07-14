"""Jarvis-X command-line interface."""

import argparse
import json
from pathlib import Path

from .assembler import Assembler
from .core import CodexVM
from .parser import Parser

ANN30_SOURCE = """LOAD30
ENCODE30
PLACE30
FIELD30
PREDICT30
COMPARE30
UPDATE_MEMORY30
PROJECT30
DECODE30
HALT30"""


def _load_vector(value):
    candidate = Path(value)
    raw = candidate.read_text(encoding="utf-8") if candidate.exists() else value
    parsed = json.loads(raw)
    if not isinstance(parsed, list):
        raise ValueError("ANN input must be a JSON array")
    return parsed


def build_parser():
    parser = argparse.ArgumentParser(prog="jarvisx")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="execute an assembly file")
    run.add_argument("file")
    run.add_argument("--ann-input")
    run.add_argument("--ann-target", type=float, default=0.0)
    run.add_argument("--ledger")

    ann = sub.add_parser("ann30d", help="run the unified 30D ANN pipeline")
    ann.add_argument("input", help="JSON array or path to a JSON array")
    ann.add_argument("--target", type=float, default=0.0)

    api = sub.add_parser("api")
    api.add_argument("--host", default="127.0.0.1")
    api.add_argument("--port", type=int, default=8080)

    web = sub.add_parser("web")
    web.add_argument("--host", default="127.0.0.1")
    web.add_argument("--port", type=int, default=5000)

    node = sub.add_parser("node")
    node.add_argument("--host", default="127.0.0.1")
    node.add_argument("--port", type=int, default=9000)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "run":
        source = Path(args.file).read_text(encoding="utf-8")
        bytecode = Assembler().assemble(Parser().parse(source))
        ann_input = _load_vector(args.ann_input) if args.ann_input else None
        vm = CodexVM(ledger_path=args.ledger)
        vm.load(bytecode, ann_input=ann_input, ann_target=args.ann_target)
        print(json.dumps(vm.run(), indent=2, sort_keys=True))
    elif args.command == "ann30d":
        bytecode = Assembler().assemble(Parser().parse(ANN30_SOURCE))
        vm = CodexVM()
        vm.load(bytecode, ann_input=_load_vector(args.input), ann_target=args.target)
        print(json.dumps(vm.run(), indent=2, sort_keys=True))
    elif args.command == "api":
        from .api import start_api

        start_api(args.host, args.port)
    elif args.command == "web":
        from .web import start_web

        start_web(args.host, args.port)
    elif args.command == "node":
        from .node import CodexNode

        CodexNode(args.host, args.port).start()


if __name__ == "__main__":
    main()
