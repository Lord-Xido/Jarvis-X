"""Signal-space dynamics for the Dr. Moagi Unified Autoencoding system."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal, cast

from .uea_model import (
    FixedPointReport,
    LinearGaussianAutoencoder,
    LossBreakdown,
    MoagiCoefficients,
    OperationSet,
    Signal3D,
    SignalMetric,
    Vector3,
    _add,
    _norm,
    _scale,
    signal_residual,
    signal_squared_error,
    wrap_phase,
)

ForcingMode = Literal["absolute", "delta"]


@dataclass(frozen=True)
class SignalBounds:
    minimum_frequency: float | None = None
    maximum_frequency: float | None = None
    minimum_amplitude: float | None = None
    maximum_amplitude: float | None = None

    def project(self, signal: Signal3D) -> Signal3D:
        frequency = signal.frequency
        amplitude = signal.amplitude
        if self.minimum_frequency is not None:
            frequency = max(frequency, self.minimum_frequency)
        if self.maximum_frequency is not None:
            frequency = min(frequency, self.maximum_frequency)
        if self.minimum_amplitude is not None:
            amplitude = max(amplitude, self.minimum_amplitude)
        if self.maximum_amplitude is not None:
            amplitude = min(amplitude, self.maximum_amplitude)
        return Signal3D(frequency, amplitude, wrap_phase(signal.phase))


@dataclass(frozen=True)
class EvolutionTrace:
    states: tuple[Signal3D, ...]
    losses: tuple[float, ...]
    derivative_norms: tuple[float, ...]
    converged: bool
    steps: int


class DrMoagiUEA:
    """Executable UEA objective, fixed-point test and signal evolution."""

    def __init__(
        self,
        model: LinearGaussianAutoencoder | None = None,
        operations: OperationSet | None = None,
        coefficients: MoagiCoefficients | None = None,
        metric: SignalMetric | None = None,
    ) -> None:
        self.model = model or LinearGaussianAutoencoder()
        self.operations = operations or OperationSet.default()
        self.coefficients = coefficients or MoagiCoefficients()
        self.metric = metric or SignalMetric()

    @staticmethod
    def _batch(signals: Iterable[Signal3D]) -> tuple[Signal3D, ...]:
        batch = tuple(signals)
        if not batch:
            raise ValueError("at least one signal is required")
        return batch

    def loss(self, signals: Iterable[Signal3D]) -> LossBreakdown:
        batch = self._batch(signals)
        count = len(batch)
        base = (
            sum(
                signal_squared_error(self.model.reconstruct(signal), signal, self.metric)
                for signal in batch
            )
            / count
        )

        operation_losses: list[tuple[str, float]] = []
        for operation in self.operations.items():
            total = 0.0
            for signal in batch:
                transformed = operation.apply(signal)
                total += signal_squared_error(
                    self.model.reconstruct(transformed),
                    transformed,
                    self.metric,
                )
            operation_losses.append((operation.name, total / count))

        kl = sum(self.model.posterior(signal).kl_standard_normal for signal in batch)
        kl /= count
        total_loss = (
            base + sum(value for _, value in operation_losses) + self.coefficients.beta * kl
        )
        return LossBreakdown(base, tuple(operation_losses), kl, total_loss)

    def fixed_point_report(
        self,
        signal: Signal3D,
        tolerance: float = 1.0e-8,
    ) -> FixedPointReport:
        if not math.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("tolerance must be finite and positive")
        base_rms = _norm(signal_residual(self.model.reconstruct(signal), signal))
        base_rms /= math.sqrt(3.0)
        operation_rms: list[tuple[str, float]] = []
        for operation in self.operations.items():
            transformed = operation.apply(signal)
            residual = signal_residual(self.model.reconstruct(transformed), transformed)
            operation_rms.append((operation.name, _norm(residual) / math.sqrt(3.0)))
        maximum = max([base_rms, *(value for _, value in operation_rms)])
        return FixedPointReport(
            base_rms,
            tuple(operation_rms),
            maximum,
            maximum <= tolerance,
        )

    def input_gradient(
        self,
        signal: Signal3D,
        finite_difference_step: float = 1.0e-5,
    ) -> Vector3:
        if not math.isfinite(finite_difference_step) or finite_difference_step <= 0.0:
            raise ValueError("finite-difference step must be finite and positive")
        base = signal.as_vector()
        gradient: list[float] = []
        for axis in range(3):
            plus = list(base)
            minus = list(base)
            plus[axis] += finite_difference_step
            minus[axis] -= finite_difference_step
            plus_signal = Signal3D.from_vector(cast(Vector3, tuple(plus)))
            minus_signal = Signal3D.from_vector(cast(Vector3, tuple(minus)))
            derivative = (self.loss((plus_signal,)).total - self.loss((minus_signal,)).total) / (
                2.0 * finite_difference_step
            )
            gradient.append(derivative)
        return cast(Vector3, tuple(gradient))

    def operation_forcing(
        self,
        signal: Signal3D,
        mode: ForcingMode = "delta",
    ) -> Vector3:
        if mode not in ("absolute", "delta"):
            raise ValueError("forcing mode must be 'absolute' or 'delta'")
        total: Vector3 = (0.0, 0.0, 0.0)
        for weight, operation in zip(
            self.coefficients.operation_weights(),
            self.operations.items(),
        ):
            transformed = operation.apply(signal)
            vector = transformed.as_vector()
            if mode == "delta":
                vector = signal_residual(transformed, signal)
            total = _add(total, _scale(weight, vector))
        return total

    def derivative(
        self,
        signal: Signal3D,
        finite_difference_step: float = 1.0e-5,
        forcing_mode: ForcingMode = "delta",
    ) -> Vector3:
        gradient = self.input_gradient(signal, finite_difference_step)
        descent = _scale(-self.coefficients.gamma, gradient)
        return _add(descent, self.operation_forcing(signal, forcing_mode))

    def evolve_step(
        self,
        signal: Signal3D,
        time_step: float,
        *,
        finite_difference_step: float = 1.0e-5,
        forcing_mode: ForcingMode = "delta",
        bounds: SignalBounds | None = None,
    ) -> Signal3D:
        if not math.isfinite(time_step) or time_step <= 0.0:
            raise ValueError("time step must be finite and positive")
        derivative = self.derivative(signal, finite_difference_step, forcing_mode)
        candidate = Signal3D.from_vector(_add(signal.as_vector(), _scale(time_step, derivative)))
        return bounds.project(candidate) if bounds is not None else candidate

    def run_to_equilibrium(
        self,
        signal: Signal3D,
        *,
        time_step: float = 1.0e-2,
        tolerance: float = 1.0e-7,
        max_steps: int = 100,
        finite_difference_step: float = 1.0e-5,
        forcing_mode: ForcingMode = "delta",
        bounds: SignalBounds | None = None,
    ) -> EvolutionTrace:
        if not math.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("tolerance must be finite and positive")
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        current = signal
        states = [current]
        losses = [self.loss((current,)).total]
        derivative_norms: list[float] = []
        for step in range(1, max_steps + 1):
            derivative = self.derivative(
                current,
                finite_difference_step,
                forcing_mode,
            )
            derivative_norm = _norm(derivative)
            derivative_norms.append(derivative_norm)
            if derivative_norm <= tolerance:
                return EvolutionTrace(
                    tuple(states),
                    tuple(losses),
                    tuple(derivative_norms),
                    True,
                    step - 1,
                )
            current = self.evolve_step(
                current,
                time_step,
                finite_difference_step=finite_difference_step,
                forcing_mode=forcing_mode,
                bounds=bounds,
            )
            states.append(current)
            losses.append(self.loss((current,)).total)
        return EvolutionTrace(
            tuple(states),
            tuple(losses),
            tuple(derivative_norms),
            False,
            max_steps,
        )
