import math

import pytest

from jarvisx.core import CodexVM
from jarvisx.engine30d import Engine30DConfig, ThirtyDAutoEncodingEngine
from jarvisx.parser import Parser
from jarvisx.assembler import Assembler


def test_cycle_is_deterministic_and_30_dimensional():
    left = ThirtyDAutoEncodingEngine()
    right = ThirtyDAutoEncodingEngine()

    left_result = left.cycle("Jarvis X, echo through.")
    right_result = right.cycle("Jarvis X, echo through.")

    assert left_result.output == right_result.output
    assert left_result.active_coordinates == right_result.active_coordinates
    assert left_result.committed is True
    assert all(len(coordinate) == 30 for coordinate in left_result.active_coordinates)
    assert math.isfinite(left_result.reconstruction_error)
    assert math.isfinite(left_result.prediction_error)


def test_sparse_manifold_is_bounded():
    engine = ThirtyDAutoEncodingEngine(
        Engine30DConfig(latent_width=2, max_active_cells=3)
    )

    for value in range(20):
        engine.cycle([float(value), float(value + 1)])

    assert len(engine.active_coordinates) <= 3


def test_repeated_observation_updates_omega_memory():
    engine = ThirtyDAutoEncodingEngine()
    first = engine.cycle([0.25, -0.5, 0.75])
    coordinate = first.active_coordinates[0]
    omega_before = tuple(engine.manifold.cells[coordinate].omega)

    engine.cycle([0.25, -0.5, 0.75])
    omega_after = tuple(engine.manifold.cells[coordinate].omega)

    assert omega_after != omega_before
    assert engine.manifold.cells[coordinate].visits == 2


def test_non_finite_observation_is_rejected_without_commit():
    engine = ThirtyDAutoEncodingEngine()

    with pytest.raises(ValueError):
        engine.cycle([float("nan")])

    assert engine.cycles == 0
    assert engine.active_coordinates == ()


def test_vm_permeates_each_instruction_into_30d_engine():
    code = "SET Ψ 10\nSET Φ 20\nADD A Ψ Φ\nHALT"
    ast = Parser().parse(code)
    bytecode = Assembler().assemble(ast)
    vm = CodexVM()

    vm.load(bytecode)
    vm.run()

    assert vm.regs["A"] == 30
    assert vm.ai30d.cycles == vm.cycles
    assert vm.ai30d.last_result is not None
    assert vm.ai30d.last_result.committed is True
