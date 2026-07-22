import math

import pytest

from jarvisx.swarm800 import (
    AGE_MAX,
    INSTANCE_COUNT,
    PIPELINE_LATENCY_US,
    ROM_BYTES,
    SVI,
    SVICodec,
    SharedROM,
    Swarm800,
    SwarmInstance,
    next_pc,
    pc_to_xyz,
    q_div,
    q_from_float,
    q_mul,
    q_to_float,
    xyz_to_pc,
)


def test_q88_round_trip_and_arithmetic():
    a = q_from_float(1.5)
    b = q_from_float(-2.0)
    assert q_to_float(a) == 1.5
    assert q_to_float(q_mul(a, b)) == -3.0
    assert q_to_float(q_div(a, q_from_float(0.5))) == 3.0
    with pytest.raises(ZeroDivisionError):
        q_div(a, 0)


def test_svi_is_exactly_128_bits_and_round_trips():
    instruction = SVI(
        opcode=255,
        flags=3,
        x=-2048,
        y=2047,
        z=-1,
        operand=0xDEADBEEF,
        edge_fingerprint=0x12345678,
        age_timer=AGE_MAX,
    )
    payload = instruction.to_bytes()
    assert len(payload) == 16
    assert SVI.from_bytes(payload) == instruction


def test_exact_128_dimensional_latent_codec():
    instruction = SVI(opcode=17, x=3, y=2, z=1, operand=42)
    latent = SVICodec.encode(instruction)
    assert len(latent) == 128
    assert SVICodec.decode(latent) == instruction


def test_rom_geometry_is_8_by_8_by_4():
    for pc in range(0, ROM_BYTES, 16):
        assert xyz_to_pc(*pc_to_xyz(pc)) == pc
    assert pc_to_xyz(ROM_BYTES - 16) == (7, 7, 3)


def test_pc_stride_is_16_bytes():
    assert next_pc(0) == 16
    assert next_pc(16, branch=True, instruction_offset=2) == 64


def test_instance_safe_mutation_never_changes_protected_fields():
    rom = SharedROM.deterministic(1)
    instance = SwarmInstance(0, rom, seed=2)
    original = rom.fetch(0, {})
    instance.step(mutate=True)
    changed = instance.patches[0]
    assert changed.opcode == original.opcode
    assert changed.flags == original.flags
    assert (changed.x, changed.y, changed.z) == (original.x, original.y, original.z)
    assert changed.age_timer == min(AGE_MAX, original.age_timer + 1)


def test_canonical_hierarchy_and_zone_fusion():
    swarm = Swarm800(seed=3)
    assert len(swarm.instances) == INSTANCE_COUNT
    report = swarm.run(cycles=10, mutate=False)
    assert report.zone_count == 80
    assert report.region_count == 8
    assert report.zone_fusions == 80
    assert report.region_fusions == 0
    assert report.current.count == 800
    assert math.isclose(report.current.p95_latency_us, PIPELINE_LATENCY_US)


def test_best_checkpoint_loss_is_monotonic():
    swarm = Swarm800(seed=4)
    first = swarm.step(mutate=False).global_best_loss
    second = swarm.step(mutate=False).global_best_loss
    assert second <= first
