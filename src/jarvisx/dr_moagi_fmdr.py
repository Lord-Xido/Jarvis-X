"""Fourier-Markov-diffusion-resonance analysis and bounded inward feedback.

This dependency-free reference module operates on a short history of sparse 3D
scalar fields. It intentionally separates four operations:

1. Fourier analysis: measure configured spatial wave-vector modes.
2. Markov analysis: model transitions between dominant spatial modes.
3. Diffusion: compute per-mode attenuation under a diagonal diffusion tensor.
4. Resonance: detect temporally coherent modal oscillation and score it against
   damping plus diffusive loss.

The optional inward loop feeds only a bounded modal correction into the current
field. Candidate-first validation preserves the Jarvis-X research authority
boundary: analysis may propose a state, but it cannot publish an unbounded or
validator-rejected candidate.

This is a numerical reference, not a production FFT, PDE solver, neural model,
or claim that model hidden states physically inhabit Euclidean 3D space.
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from .dr_moagi_field_runtime import Coordinate, SparseField

WaveVector = tuple[float, float, float]
DiffusionDiagonal = tuple[float, float, float]
FieldValidator = Callable[[Mapping[Coordinate, float], "FMDRReport"], bool]


def _finite(value: float, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive(value: float, *, name: str) -> float:
    result = _finite(value, name=name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _non_negative(value: float, *, name: str) -> float:
    result = _finite(value, name=name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _wavevector(values: Sequence[float]) -> WaveVector:
    result = tuple(_finite(value, name="wavevector component") for value in values)
    if len(result) != 3:
        raise ValueError("wavevector must contain exactly three components")
    if result == (0.0, 0.0, 0.0):
        raise ValueError("zero wavevector is not a resonance-analysis mode")
    return (result[0], result[1], result[2])


def _field(field: Mapping[Coordinate, float], *, max_active_cells: int) -> SparseField:
    if len(field) > max_active_cells:
        raise RuntimeError("field exceeds configured active-cell budget")
    result: SparseField = {}
    for coordinate, raw_value in field.items():
        if len(coordinate) != 3:
            raise ValueError("field coordinates must be 3D")
        if any(isinstance(item, bool) or not isinstance(item, int) for item in coordinate):
            raise TypeError("field coordinates must contain integers")
        value = _finite(raw_value, name="field value")
        if value != 0.0:
            result[(int(coordinate[0]), int(coordinate[1]), int(coordinate[2]))] = value
    return result


def axis_wavevectors(side: int, harmonics: int = 1) -> tuple[WaveVector, ...]:
    """Return positive axis-aligned Fourier modes for a cubic logical side."""

    if isinstance(side, bool) or not isinstance(side, int) or side <= 1:
        raise ValueError("side must be an integer greater than one")
    if isinstance(harmonics, bool) or not isinstance(harmonics, int) or harmonics <= 0:
        raise ValueError("harmonics must be a positive integer")
    vectors: list[WaveVector] = []
    for harmonic in range(1, harmonics + 1):
        k = 2.0 * math.pi * harmonic / side
        vectors.extend(((k, 0.0, 0.0), (0.0, k, 0.0), (0.0, 0.0, k)))
    return tuple(vectors)


def spatial_fourier(field: Mapping[Coordinate, float], wavevector: WaveVector) -> complex:
    """Return the active-support-normalized sparse spatial Fourier coefficient."""

    if not field:
        return 0.0j
    kx, ky, kz = _wavevector(wavevector)
    total = 0.0j
    for (x, y, z), value in field.items():
        phase = -(kx * x + ky * y + kz * z)
        total += float(value) * cmath.exp(1j * phase)
    return total / len(field)


def diffusion_rate(wavevector: WaveVector, diagonal: DiffusionDiagonal) -> float:
    """Return k^T D k for diagonal anisotropic diffusion."""

    kx, ky, kz = _wavevector(wavevector)
    if len(diagonal) != 3:
        raise ValueError("diffusion diagonal must contain exactly three values")
    dx, dy, dz = (_non_negative(value, name="diffusion coefficient") for value in diagonal)
    return dx * kx * kx + dy * ky * ky + dz * kz * kz


def diffusion_attenuation(
    wavevector: WaveVector, diagonal: DiffusionDiagonal, dt: float
) -> float:
    """Return exp(-k^T D k dt), the exact linear diffusion multiplier per mode."""

    return math.exp(-diffusion_rate(wavevector, diagonal) * _positive(dt, name="dt"))


def _temporal_spectrum(
    samples: Sequence[complex], dt: float
) -> tuple[tuple[float, complex, float], ...]:
    """Naive positive-frequency DFT for a short conformance window."""

    count = len(samples)
    if count < 2:
        return ()
    dt = _positive(dt, name="dt")
    bins: list[tuple[float, complex, float]] = []
    for index in range(1, count // 2 + 1):
        coefficient = sum(
            sample * cmath.exp(-2j * math.pi * index * n / count)
            for n, sample in enumerate(samples)
        ) / count
        omega = 2.0 * math.pi * index / (count * dt)
        power = abs(coefficient) ** 2
        bins.append((omega, coefficient, power))
    return tuple(bins)


def _transition_matrix(states: Sequence[int], state_count: int) -> tuple[tuple[float, ...], ...]:
    counts = [[0 for _ in range(state_count)] for _ in range(state_count)]
    for left, right in zip(states, states[1:]):
        counts[left][right] += 1

    rows: list[tuple[float, ...]] = []
    for index, row in enumerate(counts):
        total = sum(row)
        if total == 0:
            rows.append(tuple(1.0 if j == index else 0.0 for j in range(state_count)))
        else:
            rows.append(tuple(value / total for value in row))
    return tuple(rows)


@dataclass(frozen=True)
class FMDRConfig:
    """Numerical, resource and inward-feedback bounds."""

    wavevectors: tuple[WaveVector, ...] = ()
    dt: float = 1.0
    diffusion: DiffusionDiagonal = (0.05, 0.05, 0.05)
    damping: float = 0.05
    feedback_gain: float = 0.10
    max_feedback_delta: float = 0.10
    value_min: float = -1.0
    value_max: float = 1.0
    min_history: int = 4
    max_history: int = 32
    max_active_cells: int = 100_000

    def __post_init__(self) -> None:
        if not self.wavevectors:
            raise ValueError("wavevectors must contain at least one non-zero 3D mode")
        normalized = tuple(_wavevector(vector) for vector in self.wavevectors)
        object.__setattr__(self, "wavevectors", normalized)

        _positive(self.dt, name="dt")
        if len(self.diffusion) != 3:
            raise ValueError("diffusion must contain exactly three diagonal coefficients")
        diffusion = tuple(
            _non_negative(value, name="diffusion coefficient") for value in self.diffusion
        )
        object.__setattr__(self, "diffusion", (diffusion[0], diffusion[1], diffusion[2]))
        _positive(self.damping, name="damping")
        _non_negative(self.feedback_gain, name="feedback_gain")
        _non_negative(self.max_feedback_delta, name="max_feedback_delta")
        _finite(self.value_min, name="value_min")
        _finite(self.value_max, name="value_max")
        if self.value_min >= self.value_max:
            raise ValueError("value_min must be smaller than value_max")

        for name in ("min_history", "max_history", "max_active_cells"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.min_history > self.max_history:
            raise ValueError("min_history cannot exceed max_history")


@dataclass(frozen=True)
class ModeReport:
    index: int
    wavevector: WaveVector
    current_coefficient_real: float
    current_coefficient_imag: float
    current_amplitude: float
    diffusion_rate: float
    diffusion_attenuation: float
    dominant_omega: float
    temporal_peak_power: float
    spectral_coherence: float
    self_transition_probability: float
    resonance_score: float


@dataclass(frozen=True)
class FMDRReport:
    sample_count: int
    dominant_mode_index: int
    selected_mode_index: int
    markov_transition_matrix: tuple[tuple[float, ...], ...]
    markov_persistence: float
    modes: tuple[ModeReport, ...]

    @property
    def selected_mode(self) -> ModeReport:
        return self.modes[self.selected_mode_index]


@dataclass(frozen=True)
class FMDRState:
    cycle: int
    field: SparseField
    report: FMDRReport


def analyze_history(
    history: Sequence[Mapping[Coordinate, float]], config: FMDRConfig
) -> FMDRReport:
    """Analyze sparse-field history with Fourier, Markov, diffusion and resonance."""

    if not history:
        raise ValueError("history must contain at least one field")
    fields = tuple(
        _field(field, max_active_cells=config.max_active_cells) for field in history
    )
    mode_series: list[list[complex]] = [[] for _ in config.wavevectors]
    dominant_states: list[int] = []

    for field in fields:
        coefficients = [
            spatial_fourier(field, wavevector) for wavevector in config.wavevectors
        ]
        for index, coefficient in enumerate(coefficients):
            mode_series[index].append(coefficient)
        dominant_states.append(
            max(range(len(coefficients)), key=lambda index: abs(coefficients[index]))
        )

    matrix = _transition_matrix(dominant_states, len(config.wavevectors))
    transition_counts = [0 for _ in config.wavevectors]
    self_counts = [0 for _ in config.wavevectors]
    for left, right in zip(dominant_states, dominant_states[1:]):
        transition_counts[left] += 1
        if left == right:
            self_counts[left] += 1
    total_transitions = sum(transition_counts)
    markov_persistence = (
        sum(self_counts) / total_transitions if total_transitions else 1.0
    )

    reports: list[ModeReport] = []
    for index, (wavevector, series) in enumerate(zip(config.wavevectors, mode_series)):
        spectrum = _temporal_spectrum(series, config.dt)
        total_power = sum(power for _, _, power in spectrum)
        if spectrum and total_power > 0.0:
            omega, _, peak_power = max(spectrum, key=lambda item: item[2])
            coherence = peak_power / total_power
        else:
            omega = 0.0
            peak_power = 0.0
            coherence = 0.0

        rate = diffusion_rate(wavevector, config.diffusion)
        attenuation = math.exp(-rate * config.dt)
        self_probability = (
            self_counts[index] / transition_counts[index]
            if transition_counts[index]
            else 0.5
        )
        current = series[-1]
        stability = 0.5 + 0.5 * self_probability
        score = (
            coherence
            * math.sqrt(peak_power)
            * stability
            / (config.damping + rate)
        )
        reports.append(
            ModeReport(
                index=index,
                wavevector=wavevector,
                current_coefficient_real=current.real,
                current_coefficient_imag=current.imag,
                current_amplitude=abs(current),
                diffusion_rate=rate,
                diffusion_attenuation=attenuation,
                dominant_omega=omega,
                temporal_peak_power=peak_power,
                spectral_coherence=coherence,
                self_transition_probability=self_probability,
                resonance_score=score,
            )
        )

    current_coefficients = [series[-1] for series in mode_series]
    dominant_mode = max(
        range(len(current_coefficients)),
        key=lambda index: abs(current_coefficients[index]),
    )
    selected_mode = max(
        range(len(reports)), key=lambda index: reports[index].resonance_score
    )
    return FMDRReport(
        sample_count=len(fields),
        dominant_mode_index=dominant_mode,
        selected_mode_index=selected_mode,
        markov_transition_matrix=matrix,
        markov_persistence=markov_persistence,
        modes=tuple(reports),
    )


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _project_field(field: SparseField, config: FMDRConfig) -> SparseField:
    projected: SparseField = {}
    for coordinate, value in field.items():
        updated = _clamp(value, config.value_min, config.value_max)
        if updated != 0.0:
            projected[coordinate] = updated
    return projected


def feedback_candidate(
    field: Mapping[Coordinate, float], report: FMDRReport, config: FMDRConfig
) -> SparseField:
    """Apply bounded reinforce/suppress feedback for the selected resonant mode."""

    source = _project_field(
        _field(field, max_active_cells=config.max_active_cells), config
    )
    if not source:
        return {}

    mode = report.selected_mode
    maximum_score = max((item.resonance_score for item in report.modes), default=0.0)
    normalized_resonance = (
        mode.resonance_score / (1.0 + maximum_score) if maximum_score > 0.0 else 0.0
    )
    signed_stability = 2.0 * mode.self_transition_probability - 1.0
    gain = config.feedback_gain * normalized_resonance * signed_stability
    coefficient = complex(mode.current_coefficient_real, mode.current_coefficient_imag)
    kx, ky, kz = mode.wavevector

    candidate: SparseField = {}
    for coordinate, value in source.items():
        x, y, z = coordinate
        phase = kx * x + ky * y + kz * z
        modal_component = (coefficient * cmath.exp(1j * phase)).real
        delta = _clamp(
            gain * modal_component,
            -config.max_feedback_delta,
            config.max_feedback_delta,
        )
        updated = _clamp(value + delta, config.value_min, config.value_max)
        if updated != 0.0:
            candidate[coordinate] = updated
    return candidate


class FourierMarkovDiffusionResonanceEngine:
    """Bounded analyze -> feedback -> validate -> commit inward loop."""

    LAW_ID = "DM-vOmegaXi+-FOURIER-MARKOV-DIFFUSION-RESONANCE"
    RECURRENCE = (
        "Psi[t] -> F_k -> P_t -> exp(-k^T D k dt) -> R(k,omega) "
        "-> bounded modal feedback -> Pi_Lambda -> Psi[t+1]"
    )

    def __init__(self, initial_field: Mapping[Coordinate, float], config: FMDRConfig) -> None:
        initial = _project_field(
            _field(initial_field, max_active_cells=config.max_active_cells), config
        )
        self.config = config
        self._history: tuple[SparseField, ...] = (initial,)
        report = analyze_history(self._history, config)
        self._state = FMDRState(0, initial, report)

    @property
    def history(self) -> tuple[SparseField, ...]:
        return self._history

    @property
    def state(self) -> FMDRState:
        return self._state

    def step(
        self,
        observation: Mapping[Coordinate, float],
        *,
        validator: FieldValidator | None = None,
    ) -> FMDRState:
        observed = _project_field(
            _field(observation, max_active_cells=self.config.max_active_cells),
            self.config,
        )
        pending_history = (*self._history, observed)[-self.config.max_history :]
        report = analyze_history(pending_history, self.config)
        candidate = (
            feedback_candidate(observed, report, self.config)
            if len(pending_history) >= self.config.min_history
            else observed
        )
        if len(candidate) > self.config.max_active_cells:
            raise RuntimeError("feedback candidate exceeds configured active-cell budget")
        if validator is not None and not bool(validator(candidate, report)):
            raise RuntimeError("FMDR candidate rejected by validator")

        pending = FMDRState(self._state.cycle + 1, candidate, report)
        self._history = tuple(dict(field) for field in pending_history)
        self._state = pending
        return pending
