import json

from jarvisx.ledger_store import PersistentLedger


def test_persistent_ledger_writes_and_reloads_json(tmp_path):
    path = tmp_path / "omega_ledger.json"
    ledger = PersistentLedger(path)

    ledger.log({"A": 1}, 0x03)

    persisted = json.loads(path.read_text())
    assert len(persisted) == 1
    assert isinstance(persisted[0]["payload"], str)

    reloaded = PersistentLedger(path)
    reloaded.log({"A": 2}, 0x04)

    assert len(reloaded.chain) == 2
    assert reloaded.chain[0]["hash"] == persisted[0]["hash"]
