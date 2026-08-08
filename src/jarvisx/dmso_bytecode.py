"""Executable primitive-versus-fused bytecode backend for the DMSO reference runtime.

This module gives promoted level-1 operators an execution meaning. The primitive path dispatches
seven named operations through an interpreter; the fused path executes the same mathematics in
one direct call. Timing is telemetry only and is never used as a correctness condition.
"""

from __future__ import annotations

import math
import statistics
import time
from dataclasses import dataclass
from typing import Callable, Sequence, Tuple

from .dmso_runtime import DMSOParameters, OperatorDefinition, PRIMITIVE_TRACE, Vector


@dataclass(frozen=True)
class CellExecutionContext:
    current: Vector
    neighbour_mean: Vector
    projected: Vector
    stimulus: Vector
    alpha: float

    def __post_init__(self) -> None:
        lengths = {
            len(self.current),
            len(self.neighbour_mean),
            len(self.projected),
            len(self.stimulus),
        }
        if len(lengths) != 1 or not self.current:
            raise ValueError("all context vectors must have one equal, non-zero channel count")
        if not 0.0 < self.alpha <= 1.0 or not math.isfinite(self.alpha):
            raise ValueError("alpha must be finite and in (0, 1]")
        for vector in (self.current, self.neighbour_mean, self.projected, self.stimulus):
            if not all(math.isfinite(value) for value in vector):
                raise ValueError("execution context contains non-finite values")


@dataclass(frozen=True)
class CellExecutionResult:
    value: Vector
    dispatches: int


@dataclass(frozen=True)
class FusedOperator:
    operator_id: str
    expansion: Tuple[str, ...]
    execute: Callable[[CellExecutionContext, DMSOParameters], CellExecutionResult]


@dataclass(frozen=True)
class OperatorBenchmark:
    repetitions: int
    samples: int
    primitive_dispatches: int
    fused_dispatches: int
    dispatch_reduction_ratio: float
    primitive_ns_per_call: float
    fused_ns_per_call: float
    measured_speedup: float
    output_max_abs_error: float


class PrimitiveCellInterpreter:
    """Dispatch the canonical cell update one primitive opcode at a time."""

    def execute(
        self,
        context: CellExecutionContext,
        parameters: DMSOParameters,
        program: Sequence[str] = PRIMITIVE_TRACE,
    ) -> CellExecutionResult:
        current: Vector = context.current
        neighbours: Vector = tuple(0.0 for _ in current)
        projected: Vector = tuple(0.0 for _ in current)
        stimulus: Vector = tuple(0.0 for _ in current)
        affine: Vector = tuple(0.0 for _ in current)
        mapped: Vector = tuple(0.0 for _ in current)
        output: Vector = current
        dispatches = 0

        for opcode in program:
            dispatches += 1
            if opcode == "LOAD_SELF":
                current = context.current
            elif opcode == "AGGREGATE_26":
                neighbours = context.neighbour_mean
            elif opcode == "DECODE_FRONT":
                projected = context.projected
            elif opcode == "LOAD_INPUT":
                stimulus = context.stimulus
            elif opcode == "AFFINE":
                affine = tuple(
                    parameters.self_gain * current[channel]
                    + parameters.neighbour_gain * neighbours[channel]
                    + parameters.projection_gain * projected[channel]
                    + parameters.input_gain * stimulus[channel]
                    + parameters.bias
                    for channel in range(len(current))
                )
            elif opcode == "TANH":
                mapped = tuple(math.tanh(value) for value in affine)
            elif opcode == "RELAX":
                output = tuple(
                    current[channel]
                    + context.alpha * (mapped[channel] - current[channel])
                    for channel in range(len(current))
                )
            else:
                raise ValueError(f"unknown primitive opcode: {opcode}")
        return CellExecutionResult(value=output, dispatches=dispatches)


class FusedOperatorCompiler:
    """Compile a verified canonical promoted macro into one direct Python kernel call."""

    @staticmethod
    def compile(definition: OperatorDefinition) -> FusedOperator:
        if not definition.verified:
            raise ValueError("operator must be verified before compilation")
        if definition.expansion != PRIMITIVE_TRACE:
            raise ValueError("only the canonical level-1 DMSO trace is currently fusible")

        def execute(
            context: CellExecutionContext,
            parameters: DMSOParameters,
        ) -> CellExecutionResult:
            value = tuple(
                context.current[channel]
                + context.alpha
                * (
                    math.tanh(
                        parameters.self_gain * context.current[channel]
                        + parameters.neighbour_gain * context.neighbour_mean[channel]
                        + parameters.projection_gain * context.projected[channel]
                        + parameters.input_gain * context.stimulus[channel]
                        + parameters.bias
                    )
                    - context.current[channel]
                )
                for channel in range(len(context.current))
            )
            return CellExecutionResult(value=value, dispatches=1)

        return FusedOperator(
            operator_id=definition.operator_id,
            expansion=definition.expansion,
            execute=execute,
        )


def verify_fused_equivalence(
    definition: OperatorDefinition,
    context: CellExecutionContext,
    parameters: DMSOParameters,
    *,
    tolerance: float = 0.0,
) -> float:
    """Return max absolute error after comparing primitive and fused execution."""

    if tolerance < 0.0 or not math.isfinite(tolerance):
        raise ValueError("tolerance must be a finite non-negative value")
    primitive = PrimitiveCellInterpreter().execute(context, parameters, definition.expansion)
    fused = FusedOperatorCompiler.compile(definition).execute(context, parameters)
    error = float(max(abs(left - right) for left, right in zip(primitive.value, fused.value)))
    if error > tolerance:
        raise RuntimeError(f"fused operator failed semantic verification: error={error}")
    return error


def benchmark_operator(
    definition: OperatorDefinition,
    context: CellExecutionContext,
    parameters: DMSOParameters,
    *,
    repetitions: int = 10_000,
    samples: int = 7,
) -> OperatorBenchmark:
    """Benchmark primitive dispatch against fused dispatch after exact semantic verification."""

    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions <= 0:
        raise ValueError("repetitions must be a positive integer")
    if isinstance(samples, bool) or not isinstance(samples, int) or samples <= 0:
        raise ValueError("samples must be a positive integer")

    error = verify_fused_equivalence(definition, context, parameters, tolerance=0.0)
    interpreter = PrimitiveCellInterpreter()
    fused = FusedOperatorCompiler.compile(definition)

    for _ in range(min(100, repetitions)):
        interpreter.execute(context, parameters, definition.expansion)
        fused.execute(context, parameters)

    primitive_times = []
    fused_times = []
    checksum = 0.0
    for _ in range(samples):
        start = time.perf_counter_ns()
        for _ in range(repetitions):
            result = interpreter.execute(context, parameters, definition.expansion)
            checksum += result.value[0]
        primitive_times.append((time.perf_counter_ns() - start) / repetitions)

        start = time.perf_counter_ns()
        for _ in range(repetitions):
            result = fused.execute(context, parameters)
            checksum += result.value[0]
        fused_times.append((time.perf_counter_ns() - start) / repetitions)

    if not math.isfinite(checksum):
        raise RuntimeError("benchmark checksum became non-finite")

    primitive_ns = float(statistics.median(primitive_times))
    fused_ns = float(statistics.median(fused_times))
    primitive_dispatches = len(definition.expansion)
    fused_dispatches = 1
    return OperatorBenchmark(
        repetitions=repetitions,
        samples=samples,
        primitive_dispatches=primitive_dispatches,
        fused_dispatches=fused_dispatches,
        dispatch_reduction_ratio=primitive_dispatches / fused_dispatches,
        primitive_ns_per_call=primitive_ns,
        fused_ns_per_call=fused_ns,
        measured_speedup=primitive_ns / fused_ns if fused_ns > 0.0 else math.inf,
        output_max_abs_error=error,
    )
