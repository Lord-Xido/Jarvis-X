import math

import pytest

from jarvisx.kinetic3d import Kinetic3DRuntime, KineticOp, compile_kinetic_ir


def test_exact_reconstruction_with_zero_refine_threshold() -> None:
    runtime = Kinetic3DRuntime()
    values = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]

    result = runtime.execute(
        values,
        (2, 2, 2),
        active_threshold=0.0,
        coarse_factor=2,
        refine_threshold=0.0,
        tolerance=0.0,
    )

    assert result.reconstructed == tuple(values)
    assert result.verification.passed
    assert result.verification.mse == 0.0
    assert result.committed
    assert runtime.epoch == 1


def test_sparse_single_voxel_change_executes_only_active_cell() -> None:
    runtime = Kinetic3DRuntime()
    previous = [0.0] * 8
    current = previous.copy()
    current[6] = 10.0

    result = runtime.execute(
        current,
        (2, 2, 2),
        previous=previous,
        active_threshold=0.1,
        coarse_factor=2,
        refine_threshold=0.0,
        tolerance=0.0,
    )

    assert result.active_indices == (6,)
    assert result.telemetry.active_cells == 1
    assert result.telemetry.active_fraction == pytest.approx(1 / 8)
    assert len(result.coarse_latent) == 1
    assert len(result.fine_corrections) == 0
    assert result.reconstructed == tuple(current)
    assert result.verification.passed


def test_uniform_block_change_compresses_to_one_coarse_latent() -> None:
    runtime = Kinetic3DRuntime()
    current = [2.0] * 8

    result = runtime.execute(
        current,
        (2, 2, 2),
        active_threshold=0.0,
        coarse_factor=2,
        refine_threshold=0.0,
        tolerance=0.0,
    )

    assert len(result.coarse_latent) == 1
    assert result.coarse_latent[0].residual == 2.0
    assert len(result.fine_corrections) == 0
    assert result.telemetry.latent_values == 1
    assert result.telemetry.value_compression_ratio == 8.0
    assert result.reconstructed == tuple(current)


def test_subthreshold_change_is_deferred_until_error_budget_allows() -> None:
    runtime = Kinetic3DRuntime()
    previous = [1.0] * 8
    current = previous.copy()
    current[0] = 1.05

    result = runtime.execute(
        current,
        (2, 2, 2),
        previous=previous,
        active_threshold=0.1,
        coarse_factor=2,
        refine_threshold=0.0,
        tolerance=0.1,
    )

    assert not result.active_indices
    assert result.reconstructed == tuple(previous)
    assert result.verification.max_abs_error == pytest.approx(0.05)
    assert result.verification.passed
    assert runtime.committed_world == tuple(previous)


def test_failed_verification_rolls_back_authoritative_world() -> None:
    runtime = Kinetic3DRuntime()
    baseline = [1.0] * 8
    first = runtime.execute(baseline, (2, 2, 2), tolerance=0.0)
    assert first.committed
    assert runtime.epoch == 1

    current = baseline.copy()
    current[0] = 1.05
    failed = runtime.execute(
        current,
        (2, 2, 2),
        active_threshold=0.1,
        tolerance=0.0,
    )

    assert not failed.verification.passed
    assert not failed.committed
    assert failed.epoch_before == 1
    assert failed.epoch_after == 1
    assert runtime.epoch == 1
    assert runtime.committed_world == tuple(baseline)


def test_committed_world_becomes_next_prediction() -> None:
    runtime = Kinetic3DRuntime()
    first = [3.0] * 8
    runtime.execute(first, (2, 2, 2), tolerance=0.0)

    second = first.copy()
    second[4] = 9.0
    result = runtime.execute(second, (2, 2, 2), tolerance=0.0)

    assert result.prediction == tuple(first)
    assert result.active_indices == (4,)
    assert result.reconstructed == tuple(second)
    assert runtime.epoch == 2


def test_spatial_ir_forms_inward_outward_kinetic_wave() -> None:
    ir = compile_kinetic_ir()
    assert [node.op for node in ir] == [
        KineticOp.OBSERVE,
        KineticOp.PREDICT,
        KineticOp.RESIDUAL,
        KineticOp.ACTIVE_SET,
        KineticOp.ENCODE_COARSE,
        KineticOp.LATENT_WRITE,
        KineticOp.REFINE,
        KineticOp.DECODE,
        KineticOp.VERIFY,
        KineticOp.COMMIT,
        KineticOp.TELEMETRY,
        KineticOp.EMIT,
        KineticOp.HALT,
    ]
    assert [node.z for node in ir] == [0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 1, 0, 0]
    assert [node.x for node in ir] == list(range(len(ir)))
    assert [node.y for node in ir] == list(range(len(ir)))


def test_refine_threshold_trades_accuracy_for_latent_size() -> None:
    runtime = Kinetic3DRuntime()
    values = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]

    result = runtime.execute(
        values,
        (2, 2, 2),
        coarse_factor=2,
        refine_threshold=10.0,
        tolerance=4.0,
    )

    assert len(result.coarse_latent) == 1
    assert len(result.fine_corrections) == 0
    assert result.reconstructed == pytest.approx((3.5,) * 8)
    assert result.verification.mse == pytest.approx(5.25)
    assert result.verification.max_abs_error == pytest.approx(3.5)
    assert result.verification.passed


def test_validation_fails_closed() -> None:
    runtime = Kinetic3DRuntime(max_voxels=8)

    with pytest.raises(ValueError, match="values|current length"):
        runtime.execute([1.0], (2, 2, 2))

    with pytest.raises(ValueError, match="exceeds runtime limit"):
        runtime.execute([0.0] * 9, (3, 3, 1))

    with pytest.raises(ValueError, match="finite"):
        runtime.execute([math.inf] * 8, (2, 2, 2))

    with pytest.raises(ValueError, match="active_threshold"):
        runtime.execute([0.0] * 8, (2, 2, 2), active_threshold=-1.0)
