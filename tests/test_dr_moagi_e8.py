from __future__ import annotations

import math
import random

import pytest

from jarvisx.dr_moagi_e8 import (
    DrMoagiE8System,
    encode,
    fibonacci_nodes,
    form_map,
    heredity,
    latent_code,
    master_equation,
    random_genome,
)


def test_form_map_has_deterministic_grid_size():
    genome = random_genome(random.Random(1))
    points = form_map(genome, resolution=8)
    assert len(points) == (8 + 1) * (8 * 2)
    assert all(math.isfinite(point.radius) for point in points)


def test_encoder_and_latent_code_preserve_population_cardinality():
    genome = random_genome(random.Random(2))
    points = form_map(genome, resolution=6)
    encoded = encode(points, fibonacci_nodes(16))
    latent = latent_code(encoded)

    assert len(encoded.assignments) == len(points)
    assert sum(encoded.counts) == len(points)
    assert len(encoded.radial_sums) == 16
    assert len(latent.mean_radii) == 16
    assert 0.0 <= latent.entropy <= 1.0
    assert 1 <= latent.active_nodes <= 16


def test_master_equation_enforces_lambda_and_clock_bounds():
    stable = master_equation(coherence=1.0, loss=0.0, epsilon=0.0)
    stressed = master_equation(coherence=0.0, loss=999.0, epsilon=999.0)

    assert stable.lambda_value == pytest.approx(0.16)
    assert stable.clock_velocity == pytest.approx(1.122)
    assert stressed.lambda_value == pytest.approx(1.0)
    assert stressed.clock_velocity == pytest.approx(0.45)
    assert stressed.clock_velocity < stable.clock_velocity


def test_heredity_keeps_every_gene_in_canonical_domain():
    rng = random.Random(4)
    a = random_genome(rng)
    b = random_genome(rng)
    child = heredity(a, b, mutation_probability=1.0, mutation_sigma=10.0, rng=rng)

    assert 1 <= child["m1"] <= 12 and child["m1"].is_integer()
    assert 1 <= child["m2"] <= 12 and child["m2"].is_integer()
    assert 0.0 <= child["twist"] <= 2.5
    assert 0.0 <= child["warp"] <= 1.4
    assert 0.0 <= child["pulse"] <= 1.0
    assert 0.0 <= child["hue"] <= 360.0


def test_forward_executes_full_m1_m8_path_with_explicit_coherence():
    system = DrMoagiE8System(k=24, resolution=8)
    genome = random_genome(random.Random(5))
    result = system.forward(genome, coherence=0.82, rng=random.Random(6))

    assert len(result.points) == len(result.reconstruction)
    assert result.loss >= 0.0
    assert 0.02 <= result.sigma_epsilon <= 0.22
    assert result.fitness > 0.0
    assert 0.0 <= result.latent.entropy <= 1.0
    assert 0.04 <= result.governor.lambda_value <= 1.0
    assert 0.45 <= result.governor.clock_velocity <= 1.218


def test_coherence_is_not_silently_inferred():
    system = DrMoagiE8System(k=8, resolution=4)
    genome = random_genome(random.Random(7))
    with pytest.raises(TypeError):
        system.forward(genome)  # type: ignore[call-arg]
