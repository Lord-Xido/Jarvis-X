import json

from jarvisx.ledger_store import PersistentLedger


def test_persistent_ledger_writes_json_safe_hash_chain(tmp_path):
    path = tmp_path / "omega_ledger.json"
    ledger = PersistentLedger(str(path))

    ledger.log({"A": 30, "blob": b"\x00\xff"}, b"\x01")

    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert len(persisted) == 1
    assert isinstance(persisted[0]["payload"], str)
    assert persisted[0]["payload_encoding"] == "utf-8"
    assert len(persisted[0]["hash"]) == 64

    reloaded = PersistentLedger(str(path))
    assert reloaded.chain == ledger.chain
