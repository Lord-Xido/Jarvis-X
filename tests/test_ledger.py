import json

from jarvisx.ledger import OmegaLedger
from jarvisx.ledger_store import PersistentLedger


def test_omega_ledger_is_deterministic_and_json_serializable():
    left = OmegaLedger()
    right = OmegaLedger()
    state = {"A": 30, "IP": 2, "Ω": 0}

    left_record = left.log(state, 3)
    right_record = right.log(state, 3)

    assert left_record == right_record
    assert left_record["payload"]["sequence"] == 0
    assert json.loads(json.dumps(left.chain, ensure_ascii=False)) == left.chain


def test_persistent_ledger_round_trips_atomically(tmp_path):
    path = tmp_path / "omega.json"
    ledger = PersistentLedger(path)
    record = ledger.log({"A": 7, "IP": 1}, 1)

    assert path.exists()
    assert not (tmp_path / "omega.json.tmp").exists()
    assert json.loads(path.read_text(encoding="utf-8")) == ledger.chain

    restored = PersistentLedger(path)
    assert restored.chain == [record]
