import pytest

from jarvisx.dmso_bytecode import (
    CellExecutionContext,
    FusedOperatorCompiler,
    PrimitiveCellInterpreter,
    benchmark_operator,
    verify_fused_equivalence,
)
from jarvisx.dmso_runtime import DMSOConfig, DMSOParameters, DMSORuntime


def _promoted_operator():
    runtime = DMSORuntime(DMSOConfig(side=4, promotion_repeats=1))
    runtime.seed({(1, 1, 1): 0.1})
    runtime.step()
    return runtime.operators[0]


def _context():
    return CellExecutionContext(
        current=(0.25, -0.5),
        neighbour_mean=(0.1, 0.2),
        projected=(0.25, -0.5),
        stimulus=(0.7, -0.1),
        alpha=0.25,
    )


def test_fused_operator_is_semantically_exact_for_canonical_macro():
    operator = _promoted_operator()
    params = DMSOParameters()
    assert verify_fused_equivalence(operator, _context(), params) == 0.0


def test_fused_operator_reduces_dispatch_count_from_seven_to_one():
    operator = _promoted_operator()
    params = DMSOParameters()
    primitive = PrimitiveCellInterpreter().execute(_context(), params, operator.expansion)
    fused = FusedOperatorCompiler.compile(operator).execute(_context(), params)
    assert primitive.value == fused.value
    assert primitive.dispatches == 7
    assert fused.dispatches == 1


def test_compiler_rejects_unverified_operator():
    operator = _promoted_operator()
    invalid = type(operator)(
        operator_id=operator.operator_id,
        expansion=operator.expansion,
        depth=operator.depth,
        observed_repeats=operator.observed_repeats,
        utility=operator.utility,
        verified=False,
        human_approved=operator.human_approved,
    )
    with pytest.raises(ValueError):
        FusedOperatorCompiler.compile(invalid)


def test_benchmark_reports_equivalence_and_dispatch_reduction():
    result = benchmark_operator(
        _promoted_operator(),
        _context(),
        DMSOParameters(),
        repetitions=200,
        samples=3,
    )
    assert result.output_max_abs_error == 0.0
    assert result.dispatch_reduction_ratio == pytest.approx(7.0)
    assert result.primitive_ns_per_call > 0.0
    assert result.fused_ns_per_call > 0.0
    assert result.measured_speedup > 0.0
