import hashlib
import json

from jarvisx.ledger_store import PersistentLedger


def test_persistent_ledger_is_json_serializable_and_hash_linked(tmp_path):
    path = tmp_path / "omega_ledger.json"
    ledger = PersistentLedger(path=str(path))

    ledger.log({"A": 1}, 0x01)
    ledger.log({"A": 2}, 0x03)

    with path.open(encoding="utf-8") as handle:
        persisted = json.load(handle)

    assert len(persisted) == 2
    assert isinstance(persisted[0]["payload"], str)
    assert isinstance(persisted[1]["payload"], str)

    first_expected = hashlib.sha256(
        persisted[0]["payload"].encode("utf-8")
    ).hexdigest()
    second_expected = hashlib.sha256(
        (persisted[0]["hash"] + persisted[1]["payload"]).encode("utf-8")
    ).hexdigest()

    assert persisted[0]["hash"] == first_expected
    assert persisted[1]["hash"] == second_expected

    reloaded = PersistentLedger(path=str(path))
    assert reloaded.chain == persisted
