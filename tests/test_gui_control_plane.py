from __future__ import annotations

import pytest

from jarvisx.gui_control_plane import (
    CommandKind,
    DuplicateCommandError,
    EngineSnapshot,
    GUICommand,
    GUIControlError,
    GUIControlPlane,
    LOD,
    PanelContext,
    PanelVisibility,
    QueueFullError,
    StaleSnapshotError,
    snapshots_are_monotonic,
)


def snapshot(epoch: int = 7) -> EngineSnapshot:
    return EngineSnapshot(
        epoch=epoch,
        state_hash=f"sha-{epoch}",
        logical_cells=1_000_000,
        active_cells=10_000,
        residual_norm=0.012,
        fixed_point_delta=0.001,
        lifecycle="ready",
        changed_bits=512,
        logical_bits=1_048_576,
    )


def test_engine_snapshot_exposes_measured_sparse_fractions() -> None:
    snap = snapshot()

    assert snap.active_fraction == pytest.approx(0.01)
    assert snap.changed_fraction == pytest.approx(512 / 1_048_576)


def test_render_only_command_never_requests_authoritative_commit() -> None:
    plane = GUIControlPlane()
    snap = snapshot()
    command = GUICommand.from_mapping(
        "view-1",
        CommandKind.SET_VIEW,
        snap.epoch,
        {"view": "entropy"},
    )

    proposal = plane.propose(snap, command)

    assert proposal.render_only is True
    assert proposal.requires_pi_lambda is False
    assert proposal.required_capability is None


def test_runtime_command_remains_a_pi_lambda_candidate() -> None:
    plane = GUIControlPlane()
    snap = snapshot()
    command = GUICommand.from_mapping(
        "step-1",
        CommandKind.EXECUTE_STEP,
        snap.epoch,
    )

    proposal = plane.propose(snap, command)

    assert proposal.render_only is False
    assert proposal.requires_pi_lambda is True
    assert proposal.required_capability == "os.execute"


def test_stale_gui_command_fails_closed() -> None:
    plane = GUIControlPlane()
    command = GUICommand.from_mapping(
        "step-stale",
        CommandKind.EXECUTE_STEP,
        6,
    )

    with pytest.raises(StaleSnapshotError):
        plane.propose(snapshot(epoch=7), command)


def test_batch_execution_is_bounded_before_runtime_dispatch() -> None:
    plane = GUIControlPlane(max_batch_steps=8)
    snap = snapshot()

    accepted = GUICommand.from_mapping(
        "batch-ok",
        CommandKind.RUN_BATCH,
        snap.epoch,
        {"steps": 8},
    )
    assert plane.propose(snap, accepted).requires_pi_lambda

    rejected = GUICommand.from_mapping(
        "batch-too-large",
        CommandKind.RUN_BATCH,
        snap.epoch,
        {"steps": 9},
    )
    with pytest.raises(GUIControlError, match="within"):
        plane.propose(snap, rejected)


def test_command_queue_is_bounded_and_ids_are_idempotence_keys() -> None:
    plane = GUIControlPlane(max_queue=2)
    one = GUICommand.from_mapping("one", CommandKind.SET_VIEW, 1)
    two = GUICommand.from_mapping("two", CommandKind.SET_LOD, 1)
    three = GUICommand.from_mapping("three", CommandKind.SET_VIEW, 1)

    plane.submit(one)
    with pytest.raises(DuplicateCommandError):
        plane.submit(one)

    plane.submit(two)
    with pytest.raises(QueueFullError):
        plane.submit(three)

    assert plane.drain(limit=1) == (one,)
    assert plane.drain() == (two,)


@pytest.mark.parametrize(
    ("visibility", "last_render_ms", "now_ms", "expected"),
    [
        (PanelVisibility.FOREGROUND, 0, 16, True),
        (PanelVisibility.FOREGROUND, 0, 15, False),
        (PanelVisibility.VISIBLE, 0, 100, True),
        (PanelVisibility.INACTIVE, 0, 999, False),
        (PanelVisibility.INACTIVE, 0, 1000, True),
        (PanelVisibility.OFFSCREEN, None, 10_000, False),
    ],
)
def test_sparse_panel_scheduler(
    visibility: PanelVisibility,
    last_render_ms: int | None,
    now_ms: int,
    expected: bool,
) -> None:
    panel = PanelContext(
        panel_id="panel",
        visibility=visibility,
        camera_distance=3.0,
        last_render_ms=last_render_ms,
    )

    assert GUIControlPlane.should_render(panel, now_ms=now_ms) is expected


def test_lod_projects_global_state_without_materializing_every_bit() -> None:
    plane = GUIControlPlane()
    snap = snapshot()

    close = plane.project(
        snap,
        PanelContext("close", PanelVisibility.FOREGROUND, camera_distance=1.0),
    )
    middle = plane.project(
        snap,
        PanelContext("middle", PanelVisibility.VISIBLE, camera_distance=8.0),
    )
    far = plane.project(
        snap,
        PanelContext("far", PanelVisibility.VISIBLE, camera_distance=50.0),
    )

    assert close.lod is LOD.BIT
    assert middle.lod is LOD.TILE
    assert far.lod is LOD.REGION
    assert close.state_hash == snap.state_hash


def test_snapshot_stream_must_not_move_backwards_in_epoch() -> None:
    assert snapshots_are_monotonic([snapshot(1), snapshot(1), snapshot(2)])
    assert not snapshots_are_monotonic([snapshot(2), snapshot(1)])
