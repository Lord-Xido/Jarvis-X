from datetime import datetime, timezone

from jarvisx.transaction_fabric import OmegaChain, execute_transaction

NOW = datetime(2026, 8, 13, 12, 0, tzinfo=timezone.utc)


def sample_record():
    return {
        "structure": {"tag": "article"},
        "semantics": {"entities": ["Jarvis-X"], "topics": ["runtime"]},
        "provenance_url": "https://example.com/source",
        "observed_at": "2026-08-13T11:59:00+00:00",
        "trust": 0.9,
        "payload": {"text": "bounded observation"},
        "compliance": {
            "authorized": True,
            "robots_permitted": True,
            "terms_permitted": True,
            "restricted_data": False,
            "permission_basis": "fixture",
        },
    }


class AcceptedAdapter:
    name = "reference-adapter"
    version = "1"

    def transform(self, state):
        candidate = dict(state)
        candidate["derived"] = {"stable": True, "score": 1.0}
        return candidate

    def validate(self, before, after):
        return ()


def test_end_to_end_commit_and_omega_chain():
    omega = OmegaChain()
    receipt = execute_transaction([sample_record()], adapters=[AcceptedAdapter()], omega=omega, now=NOW)
    assert receipt.committed is True
    assert receipt.accepted_records == 1
    assert receipt.decisions[0].accepted is True
    assert receipt.authoritative_state["derived"]["stable"] is True
    assert omega.verify() is True
