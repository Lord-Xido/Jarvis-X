"""Operational reference implementation of the Dr. Moagi Equation System E8.

The module maps the eight equations into executable stages:

M1 form map -> M2 encoder -> M3 latent code -> M4 decoder -> M5 loss ->
M6 value functional -> M7 heredity -> M8 adaptive governor.

Two quantities in M8 are intentionally treated as explicit interface signals:
``coherence`` (C) and scalar ``epsilon``.  When epsilon is omitted from
``forward``, the implementation uses the measured mean magnitude of the
realized decoder perturbation.  Coherence is never guessed.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Mapping, Sequence

TAU = 2.0 * math.pi


@dataclass(frozen=True)
class GeneSpec:
    name: str
    minimum: float
    maximum: float
    integer: bool = False


GENE_SPECS: tuple[GeneSpec, ...] = (
    GeneSpec("m1", 1.0, 12.0, True),
    GeneSpec("m2", 1.0, 12.0, True),
    GeneSpec("n11", 0.1, 4.0),
    GeneSpec("n12", 0.1, 4.0),
    GeneSpec("n13", 0.1, 4.0),
    GeneSpec("n21", 0.1, 4.0),
    GeneSpec("n22", 0.1, 4.0),
    GeneSpec("n23", 0.1, 4.0),
    GeneSpec("twist", 0.0, 2.5),
    GeneSpec("warp", 0.0, 1.4),
    GeneSpec("pulse", 0.0, 1.0),
    GeneSpec("hue", 0.0, 360.0),
)


@dataclass(frozen=True)
class Point3:
    x: float
    y: float
    z: float

    @property
    def radius(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)


@dataclass(frozen=True)
class EncodedState:
    assignments: tuple[int, ...]
    radial_sums: tuple[float, ...]
    counts: tuple[int, ...]


@dataclass(frozen=True)
class LatentCode:
    mean_radii: tuple[float, ...]
    entropy: float
    active_nodes: int


@dataclass(frozen=True)
class GovernorState:
    normalized_loss: float
    normalized_epsilon: float
    lambda_value: float
    clock_velocity: float


@dataclass(frozen=True)
class ForwardPass:
    points: tuple[Point3, ...]
    encoded: EncodedState
    latent: LatentCode
    reconstruction: tuple[Point3, ...]
    perturbations: tuple[Point3, ...]
    sigma_epsilon: float
    loss: float
    fitness: float
    coherence: float
    epsilon: float
    governor: GovernorState


def clamp(value: float, minimum: float, maximum: float) -> float:
    return minimum if value < minimum else maximum if value > maximum else value


def _finite(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def normalize_genome(genome: Mapping[str, float]) -> dict[str, float]:
    """Clamp a genome to the canonical 12-dimensional E8 gene domain."""

    normalized: dict[str, float] = {}
    for spec in GENE_SPECS:
        if spec.name not in genome:
            raise KeyError(f"missing genome parameter: {spec.name}")
        value = clamp(_finite(spec.name, genome[spec.name]), spec.minimum, spec.maximum)
        normalized[spec.name] = float(round(value)) if spec.integer else value
    return normalized


def random_genome(rng: random.Random | None = None) -> dict[str, float]:
    rng = rng or random.Random()
    result: dict[str, float] = {}
    for spec in GENE_SPECS:
        value = rng.uniform(spec.minimum, spec.maximum)
        result[spec.name] = float(round(value)) if spec.integer else value
    return result


def superformula(alpha: float, m: float, n1: float, n2: float, n3: float) -> float:
    """M1 radial superformula."""

    n1 = max(float(n1), 1.0e-12)
    a = abs(math.cos(float(m) * alpha / 4.0)) ** float(n2)
    b = abs(math.sin(float(m) * alpha / 4.0)) ** float(n3)
    base = a + b
    if base <= 0.0:
        return 0.0
    radius = base ** (-1.0 / n1)
    return min(radius, 3.0) if math.isfinite(radius) else 0.0


def form_map(
    genome: Mapping[str, float],
    *,
    resolution: int = 18,
    scale: float = 1.35,
) -> tuple[Point3, ...]:
    """M1: sample x(theta, phi) over a deterministic angular grid."""

    if isinstance(resolution, bool) or not isinstance(resolution, int) or resolution < 2:
        raise ValueError("resolution must be an integer >= 2")
    if scale <= 0.0 or not math.isfinite(scale):
        raise ValueError("scale must be finite and positive")

    g = normalize_genome(genome)
    columns = resolution * 2
    points: list[Point3] = []
    for i in range(resolution + 1):
        theta = -math.pi / 2.0 + math.pi * i / resolution
        r2 = superformula(theta, g["m2"], g["n21"], g["n22"], g["n23"])
        ct, st = math.cos(theta), math.sin(theta)
        for j in range(columns):
            phi = -math.pi + TAU * j / columns
            r1 = superformula(phi, g["m1"], g["n11"], g["n12"], g["n13"])
            x = r1 * math.cos(phi) * r2 * ct
            y = r1 * math.sin(phi) * r2 * ct
            z = r2 * st

            twist = g["twist"] * theta
            if twist:
                c, s = math.cos(twist), math.sin(twist)
                x, y = x * c - y * s, x * s + y * c

            warp = 1.0 + g["warp"] * 0.3 * math.sin(3.0 * theta + 2.0 * phi)
            x, y, z = x * warp, y * warp, z * warp
            radius = math.sqrt(x * x + y * y + z * z)
            if radius > 2.8:
                factor = 2.8 / radius
                x, y, z = x * factor, y * factor, z * factor
            points.append(Point3(x * scale, y * scale, z * scale))
    return tuple(points)


def fibonacci_nodes(k: int = 48) -> tuple[Point3, ...]:
    if isinstance(k, bool) or not isinstance(k, int) or k < 2:
        raise ValueError("k must be an integer >= 2")
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    nodes: list[Point3] = []
    for i in range(k):
        y = 1.0 - 2.0 * i / (k - 1)
        radial = math.sqrt(max(0.0, 1.0 - y * y))
        theta = golden_angle * i
        nodes.append(Point3(math.cos(theta) * radial, y, math.sin(theta) * radial))
    return tuple(nodes)


def encode(points: Sequence[Point3], nodes: Sequence[Point3]) -> EncodedState:
    """M2: directional vector quantization with radial accumulation."""

    if not nodes:
        raise ValueError("nodes cannot be empty")
    radial_sums = [0.0] * len(nodes)
    counts = [0] * len(nodes)
    assignments: list[int] = []

    for point in points:
        radius = point.radius
        if radius <= 1.0e-12:
            direction = Point3(0.0, 0.0, 0.0)
        else:
            direction = Point3(point.x / radius, point.y / radius, point.z / radius)
        best_index = 0
        best_dot = -math.inf
        for index, node in enumerate(nodes):
            dot = direction.x * node.x + direction.y * node.y + direction.z * node.z
            if dot > best_dot:
                best_dot = dot
                best_index = index
        assignments.append(best_index)
        radial_sums[best_index] += radius
        counts[best_index] += 1

    return EncodedState(tuple(assignments), tuple(radial_sums), tuple(counts))


def latent_code(encoded: EncodedState) -> LatentCode:
    """M3: mean radial code and normalized entropy H(c).

    The distribution p_k is c_k / sum_j c_j, where c_k is the M2 radial sum.
    """

    mean_radii = tuple(
        encoded.radial_sums[k] / encoded.counts[k] if encoded.counts[k] else 0.0
        for k in range(len(encoded.radial_sums))
    )
    total = sum(encoded.radial_sums)
    entropy = 0.0
    if total > 0.0 and len(encoded.radial_sums) > 1:
        for value in encoded.radial_sums:
            if value > 0.0:
                probability = value / total
                entropy -= probability * math.log(probability)
        entropy /= math.log(len(encoded.radial_sums))
    return LatentCode(
        mean_radii=mean_radii,
        entropy=clamp(entropy, 0.0, 1.0),
        active_nodes=sum(1 for count in encoded.counts if count),
    )


def decode(
    encoded: EncodedState,
    latent: LatentCode,
    nodes: Sequence[Point3],
    *,
    loss_signal: float,
    rng: random.Random | None = None,
) -> tuple[tuple[Point3, ...], tuple[Point3, ...], float]:
    """M4: reconstruct each point and add U(-sigma_epsilon, sigma_epsilon) noise."""

    rng = rng or random.Random()
    sigma_epsilon = clamp(0.5 * max(0.0, _finite("loss_signal", loss_signal)), 0.02, 0.22)
    reconstruction: list[Point3] = []
    perturbations: list[Point3] = []
    for assignment in encoded.assignments:
        node = nodes[assignment]
        radius = latent.mean_radii[assignment]
        perturbation = Point3(
            rng.uniform(-sigma_epsilon, sigma_epsilon),
            rng.uniform(-sigma_epsilon, sigma_epsilon),
            rng.uniform(-sigma_epsilon, sigma_epsilon),
        )
        perturbations.append(perturbation)
        reconstruction.append(
            Point3(
                node.x * radius + perturbation.x,
                node.y * radius + perturbation.y,
                node.z * radius + perturbation.z,
            )
        )
    return tuple(reconstruction), tuple(perturbations), sigma_epsilon


def reconstruction_loss(points: Sequence[Point3], reconstruction: Sequence[Point3]) -> float:
    """M5: mean Euclidean reconstruction error."""

    if len(points) != len(reconstruction):
        raise ValueError("points and reconstruction must have equal length")
    if not points:
        return 0.0
    total = 0.0
    for point, estimate in zip(points, reconstruction):
        dx, dy, dz = point.x - estimate.x, point.y - estimate.y, point.z - estimate.z
        total += math.sqrt(dx * dx + dy * dy + dz * dz)
    return total / len(points)


def value_functional(genome: Mapping[str, float], points: Sequence[Point3]) -> float:
    """M6: phenotype value functional F(g)."""

    if not points:
        return 0.0
    g = normalize_genome(genome)
    radii = [point.radius for point in points]
    mean = sum(radii) / len(radii)
    variance = sum((radius - mean) ** 2 for radius in radii) / len(radii)
    sd = math.sqrt(max(variance, 0.0))
    cv = sd / mean if mean > 1.0e-12 else 9.0
    r_max = max(radii)
    return float(
        1.5 * math.exp(-((mean - 1.1) ** 2) / 0.22)
        + 1.3 * math.exp(-((cv - 0.3) ** 2) / 0.07)
        + 0.6 * min(r_max / 2.6, 1.0)
        + 0.35 * math.exp(-((g["twist"] - 1.1) ** 2) / 1.4)
    )


def tournament(
    population: Sequence[Mapping[str, float]],
    fitnesses: Sequence[float],
    *,
    k: int = 3,
    rng: random.Random | None = None,
) -> Mapping[str, float]:
    """M7 tournament selector."""

    if len(population) != len(fitnesses) or not population:
        raise ValueError("population and fitnesses must be non-empty and equal length")
    if k <= 0:
        raise ValueError("k must be positive")
    rng = rng or random.Random()
    indices = rng.sample(range(len(population)), k=min(k, len(population)))
    winner = max(indices, key=lambda index: fitnesses[index])
    return population[winner]


def heredity(
    parent_a: Mapping[str, float],
    parent_b: Mapping[str, float],
    *,
    mutation_sigma: float = 0.12,
    mutation_probability: float = 0.15,
    rng: random.Random | None = None,
) -> dict[str, float]:
    """M7: mask crossover followed by bounded Gaussian mutation."""

    if mutation_sigma < 0.0 or not math.isfinite(mutation_sigma):
        raise ValueError("mutation_sigma must be finite and non-negative")
    if not 0.0 <= mutation_probability <= 1.0:
        raise ValueError("mutation_probability must be in [0, 1]")
    rng = rng or random.Random()
    a, b = normalize_genome(parent_a), normalize_genome(parent_b)
    child: dict[str, float] = {}
    for spec in GENE_SPECS:
        value = a[spec.name] if rng.random() < 0.5 else b[spec.name]
        if rng.random() < mutation_probability:
            delta_scale = (spec.maximum - spec.minimum) * mutation_sigma
            value += rng.gauss(0.0, delta_scale)
        value = clamp(value, spec.minimum, spec.maximum)
        child[spec.name] = float(round(value)) if spec.integer else value
    return child


def master_equation(*, coherence: float, loss: float, epsilon: float) -> GovernorState:
    """M8: adaptive lambda governor and clock law."""

    coherence = clamp(_finite("coherence", coherence), 0.0, 1.0)
    loss = max(0.0, _finite("loss", loss))
    epsilon = max(0.0, _finite("epsilon", epsilon))
    normalized_loss = min(loss / 0.35, 1.0)
    normalized_epsilon = min(epsilon / 0.30, 1.0)
    lambda_value = clamp(
        0.16
        + 0.5 * (1.0 - coherence)
        + 0.2 * normalized_loss
        + 0.25 * normalized_epsilon,
        0.04,
        1.0,
    )
    return GovernorState(
        normalized_loss=normalized_loss,
        normalized_epsilon=normalized_epsilon,
        lambda_value=lambda_value,
        clock_velocity=1.25 - 0.8 * lambda_value,
    )


def mean_perturbation_magnitude(perturbations: Sequence[Point3]) -> float:
    if not perturbations:
        return 0.0
    return sum(point.radius for point in perturbations) / len(perturbations)


class DrMoagiE8System:
    """Executable M1 -> M8 reference pipeline."""

    def __init__(self, *, k: int = 48, resolution: int = 18, scale: float = 1.35) -> None:
        self.nodes = fibonacci_nodes(k)
        self.resolution = resolution
        self.scale = scale

    def forward(
        self,
        genome: Mapping[str, float],
        *,
        coherence: float,
        loss_signal: float | None = None,
        epsilon: float | None = None,
        rng: random.Random | None = None,
    ) -> ForwardPass:
        """Execute M1-M6 and M8 for one genome.

        ``loss_signal`` drives M4's sigma_epsilon.  If omitted, the noiseless
        vector-quantization distortion is used as the same-cycle operational
        signal.  This makes the M4/M5 dependency explicit and deterministic in
        ordering without redefining the equations.
        """

        rng = rng or random.Random()
        points = form_map(genome, resolution=self.resolution, scale=self.scale)
        encoded = encode(points, self.nodes)
        latent = latent_code(encoded)

        centers = tuple(
            Point3(
                self.nodes[index].x * latent.mean_radii[index],
                self.nodes[index].y * latent.mean_radii[index],
                self.nodes[index].z * latent.mean_radii[index],
            )
            for index in encoded.assignments
        )
        signal = reconstruction_loss(points, centers) if loss_signal is None else loss_signal
        reconstruction, perturbations, sigma_epsilon = decode(
            encoded,
            latent,
            self.nodes,
            loss_signal=signal,
            rng=rng,
        )
        loss = reconstruction_loss(points, reconstruction)
        measured_epsilon = (
            mean_perturbation_magnitude(perturbations) if epsilon is None else max(0.0, epsilon)
        )
        fitness = value_functional(genome, points)
        governor = master_equation(coherence=coherence, loss=loss, epsilon=measured_epsilon)
        return ForwardPass(
            points=points,
            encoded=encoded,
            latent=latent,
            reconstruction=reconstruction,
            perturbations=perturbations,
            sigma_epsilon=sigma_epsilon,
            loss=loss,
            fitness=fitness,
            coherence=clamp(coherence, 0.0, 1.0),
            epsilon=measured_epsilon,
            governor=governor,
        )

    def evolve_generation(
        self,
        population: Sequence[Mapping[str, float]],
        *,
        rng: random.Random | None = None,
        mutation_sigma: float = 0.12,
        mutation_probability: float = 0.15,
        elites: int = 2,
    ) -> tuple[dict[str, float], ...]:
        """Execute M6-M7 over one population generation."""

        if len(population) < 2:
            raise ValueError("population must contain at least two genomes")
        rng = rng or random.Random()
        normalized = [normalize_genome(genome) for genome in population]
        fitnesses = [
            value_functional(genome, form_map(genome, resolution=12, scale=self.scale))
            for genome in normalized
        ]
        ranked = sorted(range(len(normalized)), key=lambda index: fitnesses[index], reverse=True)
        elite_count = min(max(0, elites), len(normalized))
        next_population = [dict(normalized[index]) for index in ranked[:elite_count]]
        while len(next_population) < len(normalized):
            parent_a = tournament(normalized, fitnesses, k=3, rng=rng)
            parent_b = tournament(normalized, fitnesses, k=3, rng=rng)
            next_population.append(
                heredity(
                    parent_a,
                    parent_b,
                    mutation_sigma=mutation_sigma,
                    mutation_probability=mutation_probability,
                    rng=rng,
                )
            )
        return tuple(next_population)
