import json

from jarvisx.ledger import OmegaLedger
from jarvisx.ledger_store import PersistentLedger


def test_ledger_is_json_serializable_and_verifiable(tmp_path):
    path = tmp_path / "omega.json"
    ledger = PersistentLedger(str(path))
    ledger.log({"A": 1}, 1, timestamp_ns=100)
    ledger.log({"A": 2}, 3, timestamp_ns=101)
    assert ledger.verify()
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded[0]["state"] == {"A": 1}
    assert PersistentLedger(str(path)).verify()


def test_ledger_detects_tampering():
    ledger = OmegaLedger()
    ledger.log({"A": 1}, 1, timestamp_ns=100)
    ledger.chain[0]["state"]["A"] = 999
    assert ledger.verify() is False
