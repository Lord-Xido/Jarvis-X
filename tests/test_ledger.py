import json

from jarvisx.ledger import OmegaLedger
from jarvisx.ledger_store import PersistentLedger


def test_omega_ledger_is_deterministic_and_verifiable():
    state = {"Ψ": 7, "Φ": 3}
    left = OmegaLedger()
    right = OmegaLedger()

    left_entry = left.log(state, 1)
    right_entry = right.log(state, 1)

    assert left_entry == right_entry
    assert left.verify()
    assert right.verify()


def test_persistent_ledger_is_json_serializable_and_reloadable(tmp_path):
    path = tmp_path / "omega.json"
    ledger = PersistentLedger(str(path))
    ledger.log({"Ψ": 10, "Φ": 20}, 1)
    ledger.log({"A": 30}, 3)

    with path.open(encoding="utf-8") as source:
        raw = json.load(source)

    assert raw == ledger.chain
    assert ledger.verify()

    reloaded = PersistentLedger(str(path))
    assert reloaded.chain == ledger.chain
    assert reloaded.verify()


def test_persistent_ledger_rejects_tampering(tmp_path):
    path = tmp_path / "omega.json"
    ledger = PersistentLedger(str(path))
    ledger.log({"A": 1}, 1)

    data = ledger.chain
    data[0]["payload"]["state"]["A"] = 99
    path.write_text(json.dumps(data), encoding="utf-8")

    try:
        PersistentLedger(str(path))
    except RuntimeError as exc:
        assert "hash chain" in str(exc)
    else:
        raise AssertionError("tampered ledger was accepted")
