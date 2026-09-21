from __future__ import annotations

import time

from jarvisx.cloud import ShardCoordinate, SQLiteStateStore


def test_active_owner_can_renew_job_lease(tmp_path) -> None:
    state = SQLiteStateStore(str(tmp_path / "state.db"), str(tmp_path / "objects"))
    queued = state.create_job("demo", "chat", {"prompt": "long inference"})
    claimed = state.claim_next_job(
        worker_id="worker-owner",
        coordinate=ShardCoordinate(0, 0, 0),
        operations=("chat",),
        lease_seconds=0.05,
    )
    assert claimed is not None and claimed.id == queued.id

    time.sleep(0.01)
    assert state.renew_job_lease(
        queued.id,
        worker_id="worker-owner",
        lease_seconds=0.20,
    ) is True

    # Past the original lease, but safely inside the renewed lease.
    time.sleep(0.06)
    assert state.requeue_expired_leases() == 0
    running = state.get_job(queued.id)
    assert running is not None
    assert running.status == "running"
    assert running.lease_owner == "worker-owner"

    assert state.complete_job(
        queued.id,
        {"text": "done"},
        worker_id="worker-owner",
    ) is True
    state.close()


def test_non_owner_cannot_renew_job_lease(tmp_path) -> None:
    state = SQLiteStateStore(str(tmp_path / "state.db"), str(tmp_path / "objects"))
    queued = state.create_job("demo", "chat", {"prompt": "fenced"})
    claimed = state.claim_next_job(
        worker_id="worker-owner",
        coordinate=ShardCoordinate(0, 0, 0),
        operations=("chat",),
        lease_seconds=1.0,
    )
    assert claimed is not None

    assert state.renew_job_lease(
        queued.id,
        worker_id="worker-stale",
        lease_seconds=1.0,
    ) is False
    state.close()
