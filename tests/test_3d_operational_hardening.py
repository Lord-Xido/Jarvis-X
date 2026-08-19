from __future__ import annotations

import pytest

from jarvisx.run_store import RunArtifactStore
from jarvisx.spatial_codec_3d import MortonQuantizedFieldCodec3D, SIGNED_INT32_MAX


def test_spatial_codec_rejects_values_that_cannot_fit_packed_int32():
    codec = MortonQuantizedFieldCodec3D(step=1e-12, side=8)

    with pytest.raises(ValueError, match="signed 32-bit"):
        codec.encode({(1, 1, 1): 1.0})


def test_spatial_codec_accepts_signed_int32_boundary():
    codec = MortonQuantizedFieldCodec3D(step=1.0 / SIGNED_INT32_MAX, side=8)

    latent = codec.encode({(1, 1, 1): 1.0})

    assert next(iter(latent.entries.values())) == SIGNED_INT32_MAX


def test_run_store_persists_summary_and_verifies_omega_head(tmp_path):
    store = RunArtifactStore(tmp_path)
    run_id = store.new_run_id()
    ledger = store.ledger(run_id)
    ledger.log({"event": "test", "state_digest": "abc"}, 0xA1)
    head = ledger.chain[-1]["hash"]
    summary = {
        "journal_head_hash": head,
        "final_state_digest": "state-digest",
        "latent_digest": "latent-digest",
    }

    store.write_summary(run_id, summary)

    assert store.read_summary(run_id) == summary
    verification = store.verify(run_id)
    assert verification["verified"] is True
    assert verification["journal_head_hash"] == head
    assert verification["final_state_digest"] == "state-digest"


def test_run_store_detects_summary_to_journal_head_mismatch(tmp_path):
    store = RunArtifactStore(tmp_path)
    run_id = store.new_run_id()
    ledger = store.ledger(run_id)
    ledger.log({"event": "test"}, 0xA1)
    store.write_summary(run_id, {"journal_head_hash": "0" * 64})

    verification = store.verify(run_id)

    assert verification["journal_verified"] is True
    assert verification["head_matches_summary"] is False
    assert verification["verified"] is False


def test_run_store_rejects_path_like_run_identifier(tmp_path):
    store = RunArtifactStore(tmp_path)

    with pytest.raises(ValueError, match="32-character"):
        store.read_summary("../escape")
