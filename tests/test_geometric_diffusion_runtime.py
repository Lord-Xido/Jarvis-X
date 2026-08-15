import math

import pytest

from jarvisx.geometric_diffusion_runtime import (
    FitnessMetrics,
    GeometricDiffusionConfig,
    GeometricDiffusionRuntime,
    GeometricGraph,
    GeometricNode,
    RuntimeMutation,
    edge_energy,
    evaluate_mutation,
    forward_diffuse,
    graph_rms,
    reverse_denoise_step,
)


def fixture_graph() -> GeometricGraph:
    return GeometricGraph(
        nodes=(
            GeometricNode((0.0, 0.0, 0.0), (1.0, 0.0)),
            GeometricNode((1.0, 0.0, 0.0), (0.5, 0.5)),
            GeometricNode((1.0, 1.0, 0.0), (0.0, 1.0)),
            GeometricNode((0.0, 1.0, 0.0), (0.5, 0.5)),
        ),
        edges=((0, 1), (1, 2), (2, 3), (3, 0)),
    )


def test_graph_normalizes_undirected_edges_and_rejects_invalid_topology():
    graph = GeometricGraph(
        nodes=(
            GeometricNode((0, 0, 0), (1,)),
            GeometricNode((1, 0, 0), (2,)),
        ),
        edges=((1, 0), (0, 1)),
    )
    assert graph.edges == ((0, 1),)
    with pytest.raises(ValueError, match="self edges"):
        GeometricGraph(nodes=graph.nodes, edges=((0, 0),))
    with pytest.raises(ValueError, match="outside"):
        GeometricGraph(nodes=graph.nodes, edges=((0, 2),))


def test_forward_diffusion_is_seeded_and_reverse_step_reduces_anchor_rms():
    anchor = fixture_graph()
    noisy_a = forward_diffuse(anchor, beta=0.25, seed=7)
    noisy_b = forward_diffuse(anchor, beta=0.25, seed=7)
    noisy_c = forward_diffuse(anchor, beta=0.25, seed=8)
    assert noisy_a == noisy_b
    assert noisy_a != noisy_c

    before = graph_rms(anchor, noisy_a)[2]
    denoised = reverse_denoise_step(
        noisy_a,
        anchor,
        geometry_gain=0.65,
        graph_gain=0.0,
    )
    after = graph_rms(anchor, denoised)[2]
    assert after < before


def test_edge_energy_is_explicit_smoothness_metric():
    graph = fixture_graph()
    assert edge_energy(graph) > 0.0
    flat = GeometricGraph(
        nodes=tuple(GeometricNode(n.position, (1.0, 1.0)) for n in graph.nodes),
        edges=graph.edges,
    )
    assert edge_energy(flat) == 0.0


def test_runtime_commits_only_passing_candidate_and_rolls_back_on_validator_failure():
    anchor = fixture_graph()
    config = GeometricDiffusionConfig(
        beta=0.05,
        denoise_steps=5,
        geometry_gain=0.8,
        graph_gain=0.0,
        max_cycle_rms=0.20,
        verification_threshold=0.90,
    )
    runtime = GeometricDiffusionRuntime(anchor, config)
    committed = runtime.step(anchor, seed=11)
    assert committed.cycle == 1
    previous = runtime.state

    with pytest.raises(RuntimeError, match="validator"):
        runtime.step(anchor, seed=12, validator=lambda _: False)
    assert runtime.state is previous


def test_runtime_rejects_candidate_when_numerical_gate_is_too_strict():
    anchor = fixture_graph()
    config = GeometricDiffusionConfig(
        beta=0.30,
        denoise_steps=1,
        geometry_gain=0.05,
        graph_gain=0.0,
        max_cycle_rms=0.01,
        verification_threshold=0.0,
    )
    runtime = GeometricDiffusionRuntime(anchor, config)
    with pytest.raises(RuntimeError, match="cycle RMS"):
        runtime.step(anchor, seed=3)
    assert runtime.state.cycle == 0


def test_branch_generation_is_bounded_and_deterministic():
    anchor = fixture_graph()
    runtime = GeometricDiffusionRuntime(anchor, GeometricDiffusionConfig(branch_width=3))
    left = runtime.branch_candidates(anchor, seed=100)
    right = runtime.branch_candidates(anchor, seed=100)
    assert len(left) == 3
    assert left == right
    assert len(set(left)) == 3


def test_memory_shape_and_values_remain_finite_after_commit():
    anchor = fixture_graph()
    runtime = GeometricDiffusionRuntime(
        anchor,
        GeometricDiffusionConfig(
            beta=0.04,
            denoise_steps=6,
            geometry_gain=0.8,
            graph_gain=0.0,
            verification_threshold=0.90,
        ),
    )
    state = runtime.step(anchor, seed=21)
    assert len(state.memory) == len(anchor.nodes)
    assert all(len(item) == anchor.feature_width for item in state.memory)
    assert all(math.isfinite(x) for item in state.memory for x in item)


def test_mutation_promotion_requires_both_fitness_gain_and_verification():
    mutation = RuntimeMutation(
        "diffusion-tune-001",
        GeometricDiffusionConfig(verification_threshold=0.90),
    )
    current = FitnessMetrics(.70, .75, .70, .72, .05)
    improved = FitnessMetrics(.82, .86, .76, .84, .03)
    degraded = FitnessMetrics(.60, .65, .60, .62, .08)

    low_verify = evaluate_mutation(
        mutation,
        current_metrics=current,
        candidate_metrics=improved,
        verification_score=.89,
    )
    assert not low_verify.promoted
    assert "verification" in low_verify.reason

    no_gain = evaluate_mutation(
        mutation,
        current_metrics=current,
        candidate_metrics=degraded,
        verification_score=.99,
    )
    assert not no_gain.promoted
    assert "did not improve" in no_gain.reason

    accepted = evaluate_mutation(
        mutation,
        current_metrics=current,
        candidate_metrics=improved,
        verification_score=.99,
    )
    assert accepted.promoted
    assert accepted.candidate_fitness > accepted.current_fitness
