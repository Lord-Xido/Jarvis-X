import json

from jarvisx.assembler import Assembler
from jarvisx.core import CodexVM
from jarvisx.ledger_store import PersistentLedger
from jarvisx.parser import Parser


def bytecode():
    return Assembler().assemble(
        Parser().parse("SET A 2\nSET B 3\nMUL C A B\nHALT")
    )


def test_replay_produces_identical_ledger_hashes():
    first = CodexVM()
    second = CodexVM()
    first.load(bytecode())
    second.load(bytecode())
    first.run()
    second.run()
    assert [entry["hash"] for entry in first.ledger.chain] == [
        entry["hash"] for entry in second.ledger.chain
    ]


def test_persistent_ledger_is_json_safe_and_valid(tmp_path):
    path = tmp_path / "omega.json"
    ledger = PersistentLedger(str(path))
    vm = CodexVM(ledger=ledger)
    vm.load(bytecode())
    vm.run()
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(loaded, list)
    assert loaded[-1]["logical_time"] == vm.cycles
    assert PersistentLedger(str(path)).verify()
