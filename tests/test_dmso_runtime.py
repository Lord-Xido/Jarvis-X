import math

import pytest

from jarvisx.dmso_runtime import DMSOConfig, DMSOParameters, DMSORuntime


def test_neighbourhood_is_exact_26_shell():
    offsets = DMSORuntime.neighbour_offsets()
    assert len(offsets) == 26
    assert len(set(offsets)) == 26
    assert (0, 0, 0) not in offsets
    assert all(max(abs(x), abs(y), abs(z)) == 1 for x, y, z in offsets)


def test_front_decoder_uses_smallest_depth_and_preserves_channels():
    runtime = DMSORuntime(DMSOConfig(side=4, channels=2))
    runtime.seed({(1, 1, 2): (0.2, 0.3), (1, 1, 0): (0.8, 0.9)})
    assert runtime.decode()[(1, 1)] == (0.8, 0.9)


def test_step_is_deterministic_for_equal_state_and_input():
    config = DMSOConfig(side=4, channels=1)
    a = DMSORuntime(config)
    b = DMSORuntime(config)
    seed = {(1, 1, 1): 0.25, (2, 1, 1): -0.1}
    control = {(1, 1, 1): 0.5}
    a.seed(seed)
    b.seed(seed)
    assert a.step(control) == b.step(control)
    assert a.state == b.state


def test_zero_state_is_fixed_point():
    runtime = DMSORuntime(DMSOConfig(side=4, tolerance=1.0e-12))
    runtime.seed({(1, 1, 1): 0.0})
    metrics = runtime.settle()
    assert metrics.stable
    assert metrics.fixed_point_residual == pytest.approx(0.0)


def test_learning_moves_input_gain_toward_target():
    config = DMSOConfig(side=4, alpha=1.0, learning_rate=0.1)
    runtime = DMSORuntime(
        config,
        DMSOParameters(
            self_gain=0.0,
            neighbour_gain=0.0,
            projection_gain=0.0,
            input_gain=0.0,
            bias=0.0,
        ),
    )
    runtime.seed({(1, 1, 1): 0.0})
    before = runtime.parameters.input_gain
    first = runtime.step(
        {(1, 1, 1): 1.0},
        {(1, 1, 1): 0.5},
        learn=True,
    )
    assert first.task_loss == pytest.approx(0.25)
    assert runtime.parameters.input_gain > before
    assert runtime.verify()


def test_trace_promotes_exact_macro_and_reports_description_compression():
    runtime = DMSORuntime(DMSOConfig(side=4, promotion_repeats=2))
    runtime.seed({(1, 1, 1): 0.1, (2, 1, 1): 0.2})
    metrics = runtime.step()
    assert metrics.operator_count == 1
    assert runtime.operators[0].verified
    assert runtime.operators[0].depth == 1
    assert metrics.description_compression_ratio > 1.0


def test_higher_order_promotion_requires_human_approval():
    runtime = DMSORuntime(DMSOConfig(side=4, promotion_repeats=1, auto_approve_depth=1))
    runtime.seed({(1, 1, 1): 0.1})
    runtime.step()
    child = runtime.operators[0].operator_id
    with pytest.raises(PermissionError):
        runtime.promote_operator((child, child))
    promoted = runtime.promote_operator((child, child), human_approved=True)
    assert promoted.depth == 2
    assert promoted.human_approved
    assert runtime.verify()


def test_invalid_external_input_does_not_mutate_state():
    runtime = DMSORuntime(DMSOConfig(side=4))
    runtime.seed({(1, 1, 1): 0.25})
    before = runtime.state_digest()
    with pytest.raises(ValueError):
        runtime.step({(1, 1, 1): math.nan})
    assert runtime.state_digest() == before


def test_digest_replays_for_equal_execution():
    config = DMSOConfig(side=4, promotion_repeats=2)
    runtimes = [DMSORuntime(config), DMSORuntime(config)]
    for runtime in runtimes:
        runtime.seed({(1, 1, 1): 0.2, (2, 2, 2): -0.4})
        runtime.step({(1, 1, 1): 0.1})
        runtime.step()
    assert runtimes[0].state_digest() == runtimes[1].state_digest()
