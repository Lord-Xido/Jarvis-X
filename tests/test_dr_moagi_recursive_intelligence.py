import math

from jarvisx.dr_moagi_inward_swarm import InwardSwarmConfig
from jarvisx.dr_moagi_recursive_intelligence import (
    Recursive3DMultimodalIntelligenceEngine,
    RecursiveIntelligenceConfig,
    frequency_modulation,
    modeled_information_throughput,
)


def test_frequency_law_is_bounded_and_monotonic():
    values = [frequency_modulation(r, 0.12, 1.0, 144.0, 0.025) for r in (0.0, 0.12, 1.0)]
    assert values == sorted(values)
    assert all(1.0 <= value <= 144.0 for value in values)
    assert math.isclose(values[1], 72.5)


def test_information_throughput_identity():
    physical, represented = modeled_information_throughput(1.0e6, 512.0, 1_000.0, 1.0)
    assert physical == 512.0e6
    assert represented == 512.0e9


def test_engine_cycle_verifies_mechanics():
    config = RecursiveIntelligenceConfig(
        modeled_codec_capacity_hz=1.0e6,
        abstraction_gain=1_000.0,
        execution_efficiency=0.9,
    )
    swarm_config = InwardSwarmConfig(proxy_agents=128, fold_levels=2, seed=1234)
    engine = Recursive3DMultimodalIntelligenceEngine(config=config, swarm_config=swarm_config)
    metrics = engine.step(virtual_ops=0, auto_optimize=False)

    assert metrics.geometry_verified
    assert metrics.work_model_verified
    assert metrics.frequency_verified
    assert metrics.throughput_verified
    assert metrics.operational_mechanics_verified
    assert 1.0 <= metrics.allocated_local_frequency_hz <= 144.0
    assert 1.0 <= metrics.modeled_work_reduction <= 1.0e6
    assert metrics.modeled_represented_bytes_per_second >= metrics.modeled_physical_bytes_per_second


def test_run_cycle_count():
    engine = Recursive3DMultimodalIntelligenceEngine(
        swarm_config=InwardSwarmConfig(proxy_agents=128, seed=7)
    )
    metrics = engine.run(3, deterministic_virtual_stride=10, auto_optimize=False)
    assert [m.cycle for m in metrics] == [1, 2, 3]
    assert all(m.operational_mechanics_verified for m in metrics)
