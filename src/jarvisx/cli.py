import argparse
import json

from .api import start_api
from .assembler import Assembler
from .core import CodexVM
from .ledger_store import PersistentLedger
from .node import CodexNode
from .parser import Parser
from .web import start_web


def _run_file(path, ledger_path=None):
    with open(path, encoding="utf-8") as handle:
        source = handle.read()
    bytecode = Assembler().assemble(Parser().parse(source))
    ledger = PersistentLedger(ledger_path) if ledger_path else None
    vm = CodexVM(ledger=ledger)
    vm.load(bytecode)
    result = vm.run()
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))


def build_parser():
    parser = argparse.ArgumentParser(
        prog="jarvisx",
        description="Jarvis-X deterministic transactional VM",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="assemble and execute a program")
    run_parser.add_argument("file", help="path to a Jarvis-X assembly program")
    run_parser.add_argument(
        "--ledger",
        help="optional path for the persistent deterministic Omega ledger",
    )

    subparsers.add_parser("api", help="start the local FastAPI service")
    subparsers.add_parser("web", help="start the local browser control panel")

    node_parser = subparsers.add_parser("node", help="start the local TCP node")
    node_parser.add_argument("--host", default="127.0.0.1")
    node_parser.add_argument("--port", type=int, default=9000)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "run":
        _run_file(args.file, ledger_path=args.ledger)
    elif args.command == "api":
        start_api()
    elif args.command == "web":
        start_web()
    elif args.command == "node":
        CodexNode(host=args.host, port=args.port).start()


if __name__ == "__main__":
    main()
