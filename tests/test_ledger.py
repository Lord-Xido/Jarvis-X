import json

from jarvisx.ledger import OmegaLedger
from jarvisx.ledger_store import PersistentLedger


def test_omega_ledger_is_json_serializable_and_verifiable():
    ledger = OmegaLedger()
    ledger.log({"A": 1, "Ψ": 2}, "SET")
    json.dumps(ledger.chain)
    assert ledger.verify()


def test_persistent_ledger_round_trip(tmp_path):
    path = tmp_path / "omega.json"
    ledger = PersistentLedger(str(path))
    ledger.log({"A": 30}, "ADD")

    restored = PersistentLedger(str(path))
    assert restored.verify()
    assert restored.chain == ledger.chain
