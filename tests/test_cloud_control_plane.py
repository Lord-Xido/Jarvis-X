from jarvisx.cloud_control_plane import CloudControlPlane, TileTask, WorkerState


def worker(
    worker_id: str,
    *,
    latency_ms: float,
    resident_tiles: frozenset[str] = frozenset(),
    healthy: bool = True,
    free_units: float = 12.0,
    queue_depth: int = 0,
) -> WorkerState:
    return WorkerState(
        worker_id=worker_id,
        region="test-region",
        kind="gpu",
        capacity_units=16.0,
        free_units=free_units,
        queue_depth=queue_depth,
        latency_ms=latency_ms,
        resident_tiles=resident_tiles,
        healthy=healthy,
    )


def task(task_id: str = "task-1", parent_version: int = 0) -> TileTask:
    return TileTask(
        task_id=task_id,
        tile_id="0:0:0",
        operation="AUTO_ENCODE",
        parent_version=parent_version,
        bytes_estimate=4_000_000_000,
        preferred_kind="gpu",
    )


def test_router_prefers_data_local_worker_when_transfer_cost_dominates():
    plane = CloudControlPlane(
        {
            "local": worker(
                "local",
                latency_ms=18.0,
                resident_tiles=frozenset({"0:0:0"}),
                queue_depth=4,
            ),
            "remote": worker("remote", latency_ms=2.0),
        }
    )
    plane.register_tile("0:0:0", {"value": 1})

    route = plane.route(task())

    assert route.worker_id == "local"
    assert route.transfer_cost == 0.0


def test_router_excludes_unhealthy_workers():
    plane = CloudControlPlane(
        {
            "local": worker(
                "local",
                latency_ms=1.0,
                resident_tiles=frozenset({"0:0:0"}),
                healthy=False,
            ),
            "remote": worker("remote", latency_ms=10.0),
        }
    )
    plane.register_tile("0:0:0", {"value": 1})

    assert plane.route(task()).worker_id == "remote"


def test_verified_candidate_commits_and_retry_is_idempotent():
    plane = CloudControlPlane({"gpu-0": worker("gpu-0", latency_ms=3.0)})
    plane.register_tile("0:0:0", {"value": 1})
    calls = {"executor": 0}

    def executor(current_task, worker_id, parent):
        calls["executor"] += 1
        assert current_task.operation == "AUTO_ENCODE"
        assert worker_id == "gpu-0"
        return {"value": parent.value["value"] + 1}

    first = plane.dispatch(task(), executor=executor, verifier=lambda *_: True)
    second = plane.dispatch(task(), executor=executor, verifier=lambda *_: True)

    assert first.receipt.decision == "commit"
    assert first.receipt.resulting_version == 1
    assert plane.tile_state("0:0:0").value == {"value": 2}
    assert second.replayed is True
    assert second.receipt.receipt_hash == first.receipt.receipt_hash
    assert plane.tile_state("0:0:0").version == 1
    assert calls["executor"] == 1


def test_ctr_rejection_preserves_authoritative_state():
    plane = CloudControlPlane({"gpu-0": worker("gpu-0", latency_ms=3.0)})
    initial = plane.register_tile("0:0:0", {"value": 1})

    result = plane.dispatch(
        task(),
        executor=lambda *_: {"value": 999},
        verifier=lambda *_: False,
    )

    assert result.receipt.decision == "reject"
    assert result.receipt.reason == "ctr-verification-failed"
    assert plane.tile_state("0:0:0") == initial


def test_stale_parent_is_rejected_before_execution():
    plane = CloudControlPlane({"gpu-0": worker("gpu-0", latency_ms=3.0)})
    plane.register_tile("0:0:0", {"value": 1})

    plane.dispatch(
        task("first", 0),
        executor=lambda _task, _worker, parent: {"value": parent.value["value"] + 1},
        verifier=lambda *_: True,
    )

    calls = {"executor": 0}

    def stale_executor(*_args):
        calls["executor"] += 1
        return {"value": 123}

    stale = plane.dispatch(
        task("stale", 0),
        executor=stale_executor,
        verifier=lambda *_: True,
    )

    assert stale.receipt.decision == "reject"
    assert stale.receipt.reason == "stale-parent-version"
    assert calls["executor"] == 0
    assert plane.tile_state("0:0:0").version == 1


def test_no_schedulable_worker_fails_closed():
    plane = CloudControlPlane(
        {
            "down": worker(
                "down",
                latency_ms=1.0,
                healthy=False,
                free_units=0.0,
            )
        }
    )
    plane.register_tile("0:0:0", {"value": 1})

    try:
        plane.route(task())
    except RuntimeError as exc:
        assert "no healthy cloud worker" in str(exc)
    else:
        raise AssertionError("routing should fail when no worker is schedulable")
