"""Deterministic end-to-end AEDSIE-Sigma virtual reference engine.

The implementation operationalises the mathematical pipeline without claiming
lossless universal reconstruction or measured RF/FPGA performance.  It uses a
small deterministic signal source, finite-difference 3D operators, an adjoint
linear residual autoencoder, bounded mechanics optimisation, and a SHA3-256
provenance chain.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

Vec3 = Tuple[float, float, float]
Field = Tuple[Vec3, ...]
Spectrum = Tuple[Tuple[complex, ...], ...]


PROGRAM = (
    "ACQUIRE_RF",
    "DDC_CHANNELIZE",
    "TENSORIZE_3D",
    "DR_MOAGI_OPERATOR",
    "ENCODE_RESIDUAL",
    "DECODE_RESIDUAL",
    "COMPARE",
    "UPDATE_OMEGA",
    "INWARD_SHADOW",
    "PROJECT_MANIFOLD",
    "ROUTE_EXPERTS",
    "ESTIMATE_AOA",
    "SEAL_SHA3",
    "COMMIT",
)


@dataclass(frozen=True)
class AEDSIEConfig:
    antennas: int = 4
    samples_per_frame: int = 32
    fft_bins: int = 8
    time_depth: int = 4
    latent_dim: int = 16
    expert_count: int = 9
    class_count: int = 4
    carrier_cycles_per_sample: float = 0.125
    routing_temperature: float = 0.80
    source_blend: float = 0.35
    metric_step: float = 0.05
    omega_decay: float = 0.90
    omega_gain: float = 0.10
    analysis_budget: float = 0.50

    def validate(self) -> None:
        dimensions = (
            self.antennas,
            self.samples_per_frame,
            self.fft_bins,
            self.time_depth,
            self.latent_dim,
            self.expert_count,
            self.class_count,
        )
        if any(value <= 0 for value in dimensions):
            raise ValueError("all AEDSIE dimensions must be positive")
        if self.fft_bins > self.samples_per_frame:
            raise ValueError("fft_bins cannot exceed samples_per_frame")
        if not 0.0 < self.routing_temperature:
            raise ValueError("routing temperature must be positive")
        if not 0.0 <= self.source_blend <= 1.0:
            raise ValueError("source_blend must be in [0, 1]")
        if not 0.0 <= self.analysis_budget <= 0.5:
            raise ValueError("analysis_budget must be in [0, 0.5]")


@dataclass(frozen=True)
class Mechanics:
    alpha: float = 0.050
    beta: float = 0.030
    gamma: float = 0.020
    delta: float = 0.040
    coherence: float = 0.250
    dt: float = 0.100
    version: int = 0

    def stability_load(self) -> float:
        # Conservative explicit-step proxy for six-neighbour diffusion plus
        # the reconstruction reaction.  It is a policy gate, not a theorem for
        # every nonlinear workload.
        return self.dt * (6.0 * abs(self.alpha) + abs(self.coherence))

    def admissible(self) -> bool:
        values = (self.alpha, self.beta, self.gamma, self.delta, self.coherence, self.dt)
        return all(math.isfinite(v) and v >= 0.0 for v in values) and self.stability_load() <= 0.90


@dataclass(frozen=True)
class InwardDecision:
    attempted: bool
    accepted: bool
    rule: str
    baseline_cost: float
    shadow_cost: float
    analysis_share: float
    mechanics_version: int


@dataclass(frozen=True)
class CycleReport:
    cycle: int
    reconstruction_mse: float
    field_energy: float
    latent_energy: float
    predicted_class: int
    confidence: float
    aoa_degrees: float
    routing_weights: Tuple[float, ...]
    metric_min: float
    metric_max: float
    mechanics: Mechanics
    inward: InwardDecision
    state_hash: str
    ledger_head: str
    program_trace: Tuple[str, ...]


@dataclass(frozen=True)
class EngineReport:
    cycles: int
    field_shape: Tuple[int, int, int, int]
    latent_dim: int
    final_cycle: CycleReport
    ledger_entries: int
    ledger_head: str
    mechanics_history: Tuple[str, ...]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _swish(value: float) -> float:
    if value >= 0.0:
        return value / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return value * exp_value / (1.0 + exp_value)


def _softmax(values: Sequence[float], temperature: float = 1.0) -> Tuple[float, ...]:
    if not values:
        raise ValueError("softmax requires at least one value")
    scaled = [value / temperature for value in values]
    peak = max(scaled)
    exps = [math.exp(value - peak) for value in scaled]
    total = sum(exps)
    return tuple(value / total for value in exps)


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError("cosine vectors must have equal length")
    numerator = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return numerator / (norm_a * norm_b + 1e-12)


def _field_mse(a: Field, b: Field) -> float:
    if len(a) != len(b):
        raise ValueError("field sizes differ")
    return sum((x - y) ** 2 for va, vb in zip(a, b) for x, y in zip(va, vb)) / (3 * len(a))


def _field_energy(field: Field) -> float:
    return sum(value * value for vector in field for value in vector) / (3 * len(field))


def _latent_energy(latent: Sequence[float]) -> float:
    return sum(value * value for value in latent) / max(1, len(latent))


def _blend_fields(previous: Optional[Field], observed: Field, source_blend: float) -> Field:
    if previous is None:
        return observed
    keep = 1.0 - source_blend
    return tuple(
        tuple(keep * old + source_blend * new for old, new in zip(v_old, v_new))  # type: ignore[misc]
        for v_old, v_new in zip(previous, observed)
    )  # type: ignore[return-value]


def _quantize(value: float, scale: int = 1_000_000) -> int:
    if not math.isfinite(value):
        raise ValueError("non-finite value cannot be sealed")
    return int(round(value * scale))


def _canonical_field(field: Field) -> List[int]:
    return [_quantize(value) for vector in field for value in vector]


class SyntheticRFSource:
    """Deterministic multi-emitter complex baseband source."""

    def __init__(self, config: AEDSIEConfig):
        self.config = config

    def frame(self, frame_index: int) -> Tuple[Tuple[complex, ...], ...]:
        output: List[Tuple[complex, ...]] = []
        n_samples = self.config.samples_per_frame
        for antenna in range(self.config.antennas):
            samples: List[complex] = []
            for n in range(n_samples):
                absolute = frame_index * n_samples + n
                p1 = 2.0 * math.pi * (0.1875 * absolute + 0.115 * antenna)
                p2 = 2.0 * math.pi * (0.3125 * absolute - 0.071 * antenna + 0.03 * frame_index)
                deterministic_noise = 0.015 * math.sin(0.73 * absolute + 0.41 * antenna)
                value = complex(math.cos(p1), math.sin(p1))
                value += 0.55 * complex(math.cos(p2), math.sin(p2))
                value += complex(deterministic_noise, -0.5 * deterministic_noise)
                samples.append(value)
            output.append(tuple(samples))
        return tuple(output)


class Channelizer:
    def __init__(self, config: AEDSIEConfig):
        self.config = config

    def process(self, samples: Tuple[Tuple[complex, ...], ...]) -> Spectrum:
        result: List[Tuple[complex, ...]] = []
        size = self.config.samples_per_frame
        for antenna_samples in samples:
            if len(antenna_samples) != size:
                raise ValueError("unexpected RF frame length")
            downconverted = []
            for n, sample in enumerate(antenna_samples):
                carrier_phase = -2.0 * math.pi * self.config.carrier_cycles_per_sample * n
                carrier = complex(math.cos(carrier_phase), math.sin(carrier_phase))
                window = 0.5 - 0.5 * math.cos(2.0 * math.pi * n / max(1, size - 1))
                downconverted.append(sample * carrier * window)
            bins = []
            for k in range(self.config.fft_bins):
                total = 0j
                for n, sample in enumerate(downconverted):
                    phase = -2.0 * math.pi * k * n / size
                    total += sample * complex(math.cos(phase), math.sin(phase))
                bins.append(total / size)
            result.append(tuple(bins))
        return tuple(result)


class Tensorizer:
    def __init__(self, config: AEDSIEConfig):
        self.config = config
        self.history: List[Spectrum] = []

    def push(self, spectrum: Spectrum) -> Field:
        self.history.append(spectrum)
        self.history = self.history[-self.config.time_depth :]
        padded = [self.history[0]] * (self.config.time_depth - len(self.history)) + self.history
        field: List[Vec3] = []
        for antenna in range(self.config.antennas):
            for frequency in range(self.config.fft_bins):
                for time_index in range(self.config.time_depth):
                    value = padded[time_index][antenna][frequency]
                    field.append((value.real, value.imag, abs(value)))
        return tuple(field)


class DrMoagi3DOperator:
    def __init__(self, config: AEDSIEConfig):
        self.config = config
        self.nx = config.antennas
        self.ny = config.fft_bins
        self.nz = config.time_depth

    def _index(self, x: int, y: int, z: int) -> int:
        return ((x % self.nx) * self.ny + (y % self.ny)) * self.nz + (z % self.nz)

    def _vector(self, field: Field, x: int, y: int, z: int) -> Vec3:
        return field[self._index(x, y, z)]

    def _divergence(self, field: Field, x: int, y: int, z: int) -> float:
        xp, xm = self._vector(field, x + 1, y, z), self._vector(field, x - 1, y, z)
        yp, ym = self._vector(field, x, y + 1, z), self._vector(field, x, y - 1, z)
        zp, zm = self._vector(field, x, y, z + 1), self._vector(field, x, y, z - 1)
        return 0.5 * ((xp[0] - xm[0]) + (yp[1] - ym[1]) + (zp[2] - zm[2]))

    def apply(self, field: Field, mechanics: Mechanics) -> Field:
        if len(field) != self.nx * self.ny * self.nz:
            raise ValueError("field shape does not match operator geometry")
        output: List[Vec3] = []
        for x in range(self.nx):
            for y in range(self.ny):
                for z in range(self.nz):
                    center = self._vector(field, x, y, z)
                    xp, xm = self._vector(field, x + 1, y, z), self._vector(field, x - 1, y, z)
                    yp, ym = self._vector(field, x, y + 1, z), self._vector(field, x, y - 1, z)
                    zp, zm = self._vector(field, x, y, z + 1), self._vector(field, x, y, z - 1)
                    neighbours = (xp, xm, yp, ym, zp, zm)
                    lap = tuple(sum(v[c] for v in neighbours) - 6.0 * center[c] for c in range(3))
                    curl = (
                        0.5 * ((yp[2] - ym[2]) - (zp[1] - zm[1])),
                        0.5 * ((zp[0] - zm[0]) - (xp[2] - xm[2])),
                        0.5 * ((xp[1] - xm[1]) - (yp[0] - ym[0])),
                    )
                    grad_div = (
                        0.5
                        * (
                            self._divergence(field, x + 1, y, z)
                            - self._divergence(field, x - 1, y, z)
                        ),
                        0.5
                        * (
                            self._divergence(field, x, y + 1, z)
                            - self._divergence(field, x, y - 1, z)
                        ),
                        0.5
                        * (
                            self._divergence(field, x, y, z + 1)
                            - self._divergence(field, x, y, z - 1)
                        ),
                    )
                    local_mix = tuple(sum(v[c] for v in neighbours) / 6.0 - center[c] for c in range(3))
                    combined = tuple(
                        mechanics.alpha * lap[c]
                        + mechanics.beta * curl[c]
                        + mechanics.gamma * grad_div[c]
                        + mechanics.delta * local_mix[c]
                        for c in range(3)
                    )
                    output.append(tuple(_swish(value) for value in combined))  # type: ignore[arg-type]
        return tuple(output)


class ResidualAutoencoder:
    """Finite orthonormal projection with an explicit per-channel baseband skip."""

    def __init__(self, field_cells: int, latent_dim: int):
        self.field_cells = field_cells
        self.width = field_cells * 3
        self.latent_dim = min(latent_dim, max(1, self.width - 3))
        scale = math.sqrt(2.0 / self.width)
        self.basis: Tuple[Tuple[float, ...], ...] = tuple(
            tuple(scale * math.cos(math.pi * (j + 0.5) * (k + 1) / self.width) for j in range(self.width))
            for k in range(self.latent_dim)
        )

    def encode(self, field: Field) -> Tuple[Tuple[float, ...], Vec3]:
        if len(field) != self.field_cells:
            raise ValueError("unexpected autoencoder field size")
        base = tuple(sum(vector[c] for vector in field) / len(field) for c in range(3))
        residual = [vector[c] - base[c] for vector in field for c in range(3)]
        latent = tuple(sum(weight * value for weight, value in zip(row, residual)) for row in self.basis)
        return latent, base  # type: ignore[return-value]

    def decode(self, latent: Sequence[float], base: Vec3) -> Field:
        if len(latent) != self.latent_dim:
            raise ValueError("unexpected latent size")
        reconstructed = [0.0] * self.width
        for coefficient, row in zip(latent, self.basis):
            for index, weight in enumerate(row):
                reconstructed[index] += coefficient * weight
        field = []
        for cell in range(self.field_cells):
            offset = cell * 3
            field.append(
                (
                    base[0] + reconstructed[offset],
                    base[1] + reconstructed[offset + 1],
                    base[2] + reconstructed[offset + 2],
                )
            )
        return tuple(field)


class ExpertRouter:
    def __init__(self, config: AEDSIEConfig, latent_dim: int):
        self.config = config
        self.latent_dim = latent_dim
        norm = math.sqrt(max(1, latent_dim))
        self.embeddings = tuple(
            tuple(math.cos((expert + 1) * (index + 1) * 0.37) / norm for index in range(latent_dim))
            for expert in range(config.expert_count)
        )
        self.weights = tuple(
            tuple(
                tuple(
                    math.sin((expert + 1) * (class_id + 2) * (index + 1) * 0.11) / norm
                    for index in range(latent_dim)
                )
                for class_id in range(config.class_count)
            )
            for expert in range(config.expert_count)
        )

    def infer(self, latent: Sequence[float]) -> Tuple[int, float, Tuple[float, ...]]:
        similarities = [_cosine(latent, embedding) for embedding in self.embeddings]
        routing = _softmax(similarities, self.config.routing_temperature)
        logits = [0.0] * self.config.class_count
        for expert, route_weight in enumerate(routing):
            for class_id in range(self.config.class_count):
                expert_logit = sum(
                    weight * value for weight, value in zip(self.weights[expert][class_id], latent)
                )
                logits[class_id] += route_weight * expert_logit
        probabilities = _softmax(logits)
        predicted = max(range(len(probabilities)), key=probabilities.__getitem__)
        return predicted, probabilities[predicted], routing


class OmegaLedger:
    def __init__(self):
        self.head = "0" * 64
        self.records: List[str] = []

    def append(self, payload: Dict[str, object]) -> str:
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        self.head = hashlib.sha3_256(bytes.fromhex(self.head) + canonical).hexdigest()
        self.records.append(self.head)
        return self.head


class AEDSIEVirtualEngine:
    def __init__(
        self,
        config: Optional[AEDSIEConfig] = None,
        mechanics: Optional[Mechanics] = None,
    ):
        self.config = config or AEDSIEConfig()
        self.config.validate()
        self.mechanics = mechanics or Mechanics()
        if not self.mechanics.admissible():
            raise ValueError("initial mechanics are not admissible")
        self.source = SyntheticRFSource(self.config)
        self.channelizer = Channelizer(self.config)
        self.tensorizer = Tensorizer(self.config)
        cells = self.config.antennas * self.config.fft_bins * self.config.time_depth
        self.operator = DrMoagi3DOperator(self.config)
        self.autoencoder = ResidualAutoencoder(cells, self.config.latent_dim)
        self.router = ExpertRouter(self.config, self.autoencoder.latent_dim)
        self.ledger = OmegaLedger()
        self.field: Optional[Field] = None
        self.omega: Field = tuple((0.0, 0.0, 0.0) for _ in range(cells))
        self.metric: Tuple[float, ...] = tuple(1.0 for _ in range(self.autoencoder.latent_dim))
        self.cycle_index = 0
        self.mechanics_history: List[str] = [self._mechanics_hash(self.mechanics, "BOOT")]

    @staticmethod
    def _mechanics_hash(mechanics: Mechanics, parent: str) -> str:
        payload = json.dumps(
            {"mechanics": asdict(mechanics), "parent": parent}, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha3_256(payload).hexdigest()

    def _transition(
        self,
        field: Field,
        reconstructed: Field,
        omega: Field,
        mechanics: Mechanics,
    ) -> Tuple[Field, Field]:
        if not mechanics.admissible():
            raise ValueError("inadmissible mechanics")
        differential = self.operator.apply(field, mechanics)
        next_omega = []
        next_field = []
        for current, target, memory, motion in zip(field, reconstructed, omega, differential):
            error = tuple(target[c] - current[c] for c in range(3))
            updated_memory = tuple(
                self.config.omega_decay * memory[c] + self.config.omega_gain * error[c]
                for c in range(3)
            )
            proposal = tuple(
                current[c]
                + mechanics.dt
                * (motion[c] + mechanics.coherence * error[c] + updated_memory[c])
                for c in range(3)
            )
            next_omega.append(updated_memory)
            next_field.append(proposal)
        return tuple(next_field), tuple(next_omega)

    @staticmethod
    def _cost(candidate: Field, observed: Field, reconstructed: Field) -> float:
        return (
            _field_mse(candidate, observed)
            + 0.20 * _field_mse(candidate, reconstructed)
            + 0.001 * _field_energy(candidate)
        )

    def _candidate(self, reconstruction_mse: float) -> Tuple[str, Mechanics]:
        current = self.mechanics
        if reconstruction_mse > 1e-4:
            return (
                "INCREASE_COHERENCE",
                Mechanics(
                    current.alpha,
                    current.beta,
                    current.gamma,
                    current.delta,
                    min(0.80, current.coherence * 1.10),
                    current.dt,
                    current.version + 1,
                ),
            )
        return (
            "REDUCE_DIFFUSION",
            Mechanics(
                max(0.005, current.alpha * 0.90),
                current.beta,
                current.gamma,
                current.delta,
                current.coherence,
                current.dt,
                current.version + 1,
            ),
        )

    def _inward_turn(
        self,
        field: Field,
        observed: Field,
        reconstructed: Field,
        baseline_field: Field,
        baseline_omega: Field,
        reconstruction_mse: float,
        enabled: bool,
    ) -> Tuple[Field, Field, InwardDecision]:
        baseline_cost = self._cost(baseline_field, observed, reconstructed)
        if not enabled:
            decision = InwardDecision(
                False,
                False,
                "DISABLED",
                baseline_cost,
                baseline_cost,
                0.0,
                self.mechanics.version,
            )
            return baseline_field, baseline_omega, decision
        rule, candidate = self._candidate(reconstruction_mse)
        if not candidate.admissible():
            decision = InwardDecision(
                True,
                False,
                rule,
                baseline_cost,
                float("inf"),
                0.5,
                self.mechanics.version,
            )
            return baseline_field, baseline_omega, decision
        shadow_field, shadow_omega = self._transition(field, reconstructed, self.omega, candidate)
        shadow_cost = self._cost(shadow_field, observed, reconstructed)
        analysis_share = 0.5
        accepted = (
            analysis_share <= self.config.analysis_budget
            and shadow_cost + 1e-12 < baseline_cost
            and all(math.isfinite(value) for vector in shadow_field for value in vector)
        )
        if accepted:
            parent = self.mechanics_history[-1]
            self.mechanics = candidate
            self.mechanics_history.append(self._mechanics_hash(candidate, parent + rule))
            selected_field, selected_omega = shadow_field, shadow_omega
        else:
            selected_field, selected_omega = baseline_field, baseline_omega
        decision = InwardDecision(
            True,
            accepted,
            rule,
            baseline_cost,
            shadow_cost,
            analysis_share,
            self.mechanics.version,
        )
        return selected_field, selected_omega, decision

    def _update_metric(self, latent: Sequence[float]) -> None:
        step = self.config.metric_step
        self.metric = tuple(
            max(1e-6, (1.0 - step) * current + step * (1.0 + abs(value)))
            for current, value in zip(self.metric, latent)
        )

    @staticmethod
    def _estimate_aoa(spectrum: Spectrum) -> float:
        if len(spectrum) < 2 or not spectrum[0]:
            return 0.0
        strongest = max(
            range(len(spectrum[0])),
            key=lambda index: sum(abs(antenna[index]) for antenna in spectrum),
        )
        differences = []
        for left, right in zip(spectrum, spectrum[1:]):
            product = right[strongest] * left[strongest].conjugate()
            differences.append(math.atan2(product.imag, product.real))
        mean_phase = sum(differences) / max(1, len(differences))
        return math.degrees(math.asin(_clamp(mean_phase / math.pi, -1.0, 1.0)))

    def step(self, inward: bool = True) -> CycleReport:
        trace: List[str] = []
        samples = self.source.frame(self.cycle_index)
        trace.append("ACQUIRE_RF")
        spectrum = self.channelizer.process(samples)
        trace.append("DDC_CHANNELIZE")
        observed = self.tensorizer.push(spectrum)
        trace.append("TENSORIZE_3D")
        field = _blend_fields(self.field, observed, self.config.source_blend)
        trace.append("DR_MOAGI_OPERATOR")
        latent, base = self.autoencoder.encode(field)
        trace.append("ENCODE_RESIDUAL")
        reconstructed = self.autoencoder.decode(latent, base)
        trace.append("DECODE_RESIDUAL")
        reconstruction_mse = _field_mse(field, reconstructed)
        trace.append("COMPARE")
        baseline_field, baseline_omega = self._transition(
            field, reconstructed, self.omega, self.mechanics
        )
        trace.append("UPDATE_OMEGA")
        selected_field, selected_omega, decision = self._inward_turn(
            field,
            observed,
            reconstructed,
            baseline_field,
            baseline_omega,
            reconstruction_mse,
            inward,
        )
        trace.append("INWARD_SHADOW")
        self.field = selected_field
        self.omega = selected_omega
        self._update_metric(latent)
        trace.append("PROJECT_MANIFOLD")
        predicted, confidence, routing = self.router.infer(latent)
        trace.append("ROUTE_EXPERTS")
        aoa = self._estimate_aoa(spectrum)
        trace.append("ESTIMATE_AOA")
        state_payload = {
            "cycle": self.cycle_index + 1,
            "field": _canonical_field(self.field),
            "latent": [_quantize(value) for value in latent],
            "metric": [_quantize(value) for value in self.metric],
            "mechanics": asdict(self.mechanics),
            "prediction": predicted,
            "aoa": _quantize(aoa),
        }
        state_bytes = json.dumps(state_payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        state_hash = hashlib.sha3_256(state_bytes).hexdigest()
        ledger_head = self.ledger.append(
            {
                "cycle": self.cycle_index + 1,
                "state_hash": state_hash,
                "mechanics_hash": self.mechanics_history[-1],
                "program": PROGRAM,
            }
        )
        trace.append("SEAL_SHA3")
        self.cycle_index += 1
        trace.append("COMMIT")
        return CycleReport(
            self.cycle_index,
            reconstruction_mse,
            _field_energy(self.field),
            _latent_energy(latent),
            predicted,
            confidence,
            aoa,
            routing,
            min(self.metric),
            max(self.metric),
            self.mechanics,
            decision,
            state_hash,
            ledger_head,
            tuple(trace),
        )

    def run(self, cycles: int = 4, inward: bool = True) -> EngineReport:
        if cycles < 1:
            raise ValueError("cycles must be positive")
        final = None
        for _ in range(cycles):
            final = self.step(inward=inward)
        assert final is not None
        return EngineReport(
            cycles,
            (self.config.antennas, self.config.fft_bins, self.config.time_depth, 3),
            self.autoencoder.latent_dim,
            final,
            len(self.ledger.records),
            self.ledger.head,
            tuple(self.mechanics_history),
        )


def run_aedsie(cycles: int = 4, inward: bool = True) -> Dict[str, object]:
    """Run the complete deterministic reference pipeline and return JSON-safe data."""

    return AEDSIEVirtualEngine().run(cycles=cycles, inward=inward).to_dict()
