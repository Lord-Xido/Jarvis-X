import json
from collections.abc import Callable

import pytest

from jarvisx.ledger import GENESIS_HASH, OmegaLedger
from jarvisx.ledger_store import PersistentLedger


def clock(*values: int) -> Callable[[], int]:
    sequence = iter(values)
    return lambda: next(sequence)


def test_ledger_is_json_native_and_hash_chained() -> None:
    ledger = OmegaLedger(clock_ns=clock(100, 200))

    first = ledger.log({"A": 1}, 1)
    second = ledger.log({"A": 2}, 3)

    assert first["previous_hash"] == GENESIS_HASH
    assert second["previous_hash"] == first["hash"]
    assert ledger.verify()
    assert json.loads(json.dumps(ledger.chain)) == ledger.chain


def test_ledger_detects_tampering() -> None:
    ledger = OmegaLedger(clock_ns=clock(100))
    ledger.log({"A": 1}, 1)

    ledger.chain[0]["state"]["A"] = 999

    assert not ledger.verify()


def test_persistent_ledger_round_trip(tmp_path) -> None:
    path = tmp_path / "state" / "omega.json"
    ledger = PersistentLedger(path, clock_ns=clock(123))
    ledger.log({"Ψ": 10, "Φ": 20}, 3)

    restored = PersistentLedger(path)

    assert restored.chain == ledger.chain
    assert restored.verify()
    assert not path.with_name(f".{path.name}.tmp").exists()


def test_persistent_ledger_rejects_corruption(tmp_path) -> None:
    path = tmp_path / "omega.json"
    path.write_text('[{"hash": "invalid"}]', encoding="utf-8")

    with pytest.raises(ValueError, match="integrity verification failed"):
        PersistentLedger(path)
