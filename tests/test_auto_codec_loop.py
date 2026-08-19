from __future__ import annotations

import pytest

from jarvisx.auto_codec_loop import (
    AutoCodecLoop,
    AutoCodecLoopConfig,
    UniformQuantizedFieldCodec,
    digest_field,
    field_to_cells,
)
from jarvisx.dr_moagi_field_runtime import DrMoagiFieldConfig, DrMoagiFieldRuntime


def _runtime(*, step=0.1, dt=0.1):
    return DrMoagiFieldRuntime(
        UniformQuantizedFieldCodec(step),
        DrMoagiFieldConfig(
            side=8,
            alpha=1.0,
            lambda_residual=0.0,
            eta=0.0,
            dt=dt,
            expand_halo=False,
        ),
    )


def test_uniform_quantized_codec_round_trip_is_sparse_and_deterministic():
    codec = UniformQuantizedFieldCodec(0.1)
    latent = codec.encode({(1, 2, 3): 0.26, (0, 0, 0): 0.01})

    assert latent.codes == {(1, 2, 3): 3}
    assert codec.decode(latent, [(1, 2, 3), (0, 0, 0)]) == {
        (1, 2, 3): pytest.approx(0.3),
        (0, 0, 0): pytest.approx(0.0),
    }


def test_auto_codec_loop_reduces_reconstruction_error_to_target():
    runtime = _runtime(step=0.1, dt=0.1)
    loop = AutoCodecLoop(
        runtime,
        AutoCodecLoopConfig(
            max_cycles=16,
            reconstruction_mse_target=1e-3,
            stop_on_fixed_point=False,
        ),
    )
    loop.load({(2, 2, 2): 0.26})

    summary = loop.run()

    assert summary.converged
    assert summary.stop_reason == "reconstruction_target"
    assert 1 <= summary.cycles_executed <= 16
    assert summary.final_reconstruction_mse is not None
    assert summary.final_reconstruction_mse <= 1e-3
    assert summary.final_state[(2, 2, 2)] > 0.26
    assert summary.journal_verified
    assert summary.journal_entries == summary.cycles_executed + 2
    assert summary.journal_head_hash is not None


def test_rejection_budget_stops_without_committing_candidate():
    runtime = _runtime()
    initial = {(2, 2, 2): 0.26}
    loop = AutoCodecLoop(
        runtime,
        AutoCodecLoopConfig(max_cycles=8, max_consecutive_rejections=2),
    )
    loop.load(initial)

    summary = loop.run(validator=lambda candidate, metrics: False)

    assert summary.stop_reason == "rejection_limit"
    assert not summary.converged
    assert summary.cycles_executed == 2
    assert summary.committed_cycles == 0
    assert summary.rejected_cycles == 2
    assert summary.final_state == initial
    assert summary.journal_verified


def test_cycle_budget_is_a_hard_upper_bound():
    runtime = _runtime(step=0.1, dt=0.01)
    loop = AutoCodecLoop(
        runtime,
        AutoCodecLoopConfig(
            max_cycles=2,
            reconstruction_mse_target=0.0,
            stop_on_fixed_point=False,
        ),
    )
    loop.load({(2, 2, 2): 0.26})

    summary = loop.run()

    assert summary.stop_reason == "cycle_limit"
    assert summary.cycles_executed == 2
    assert summary.committed_cycles == 2
    assert not summary.converged


def test_loop_requires_explicit_load():
    loop = AutoCodecLoop(_runtime())

    with pytest.raises(RuntimeError, match="load a field"):
        loop.run()


def test_field_receipts_have_stable_order_and_digest():
    a = {(2, 1, 0): -0.5, (0, 0, 0): 0.25}
    b = {(0, 0, 0): 0.25, (2, 1, 0): -0.5}

    assert field_to_cells(a) == field_to_cells(b)
    assert digest_field(a) == digest_field(b)
