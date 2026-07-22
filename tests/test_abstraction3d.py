import math

import pytest

from jarvisx.abstraction3d import (
    FEATURE_WIDTH,
    AbstractionANNCore3D,
    Instruction3D,
    Opcode3D,
    continuous_coordinate,
    project_features,
    trilinear_route,
)
from jarvisx.assembler import Assembler
from jarvisx.core import CodexVM
from jarvisx.parser import Parser

ABSTRACTION_SOURCE = """LOAD3D
ABSTRACT3D
ROUTE3D
ATTEND3D
PREDICT3D
COMPARE3D
LEARN3D
PROJECT3D
DECODE3D
HALT3D"""


def test_projection_and_route_are_bounded_and_deterministic():
    first = project_features([0.8, -0.3, 0.5, 1.0])
    second = project_features([0.8, -0.3, 0.5, 1.0])
    assert first == pytest.approx(second)
    assert len(first) == FEATURE_WIDTH
    magnitude = math.sqrt(sum(value * value for value in first))
    assert magnitude == pytest.approx(1.0)

    point = continuous_coordinate(first, side=64)
    route = trilinear_route(point, side=64)
    assert 1 <= len(route) <= 8
    assert sum(weight for _, weight in route) == pytest.approx(1.0)
    assert all(
        all(0 <= component < 64 for component in coordinate)
        for coordinate, _ in route
    )


def test_full_pipeline_is_sparse_and_operational():
    core = AbstractionANNCore3D()
    snapshot = core.run([0.8, -0.3, 0.5, 1.0], target=0.8)
    assert snapshot.dimensions == 3
    assert snapshot.theoretical_nodes == 64 ** 3
    assert 1 <= snapshot.active_nodes <= 8
    assert len(snapshot.route) == len(snapshot.attention)
    assert sum(snapshot.attention) == pytest.approx(1.0)
    assert len(snapshot.output) == 4
    assert snapshot.cycles == 10
    assert snapshot.halted is True
    assert math.isfinite(snapshot.loss)


def test_associative_memory_reduces_repeated_observation_loss():
    core = AbstractionANNCore3D()
    first = core.run([0.8, -0.3, 0.5, 1.0], target=0.8)
    latest = first
    for _ in range(12):
        latest = core.run([0.8, -0.3, 0.5, 1.0], target=0.8)
    assert latest.loss < first.loss
    assert latest.memory_norm > first.memory_norm
    assert latest.active_nodes == first.active_nodes


def test_invalid_instruction_order_is_rejected_without_mutation():
    core = AbstractionANNCore3D()
    before = core.state_hash()
    with pytest.raises(RuntimeError):
        core.execute(Instruction3D(Opcode3D.PREDICT))
    assert core.state_hash() == before
    assert core.cycles == 0
    assert core.lattice.active_nodes == 0


def test_active_node_quota_rolls_back_partial_route_materialization():
    core = AbstractionANNCore3D(max_active_nodes=4)
    with pytest.raises(MemoryError):
        core.run([1.0, 2.0, 3.0], target=0.25)
    assert core.lattice.active_nodes == 0


def test_equal_initial_state_produces_equal_hash_and_output():
    first = AbstractionANNCore3D()
    second = AbstractionANNCore3D()
    a = first.run([1.0, 2.0, 3.0], target=0.25)
    b = second.run([1.0, 2.0, 3.0], target=0.25)
    assert a.output == pytest.approx(b.output)
    assert a.prediction == pytest.approx(b.prediction)
    assert first.state_hash() == second.state_hash()


def test_unified_vm_executes_3d_abstraction_isa():
    bytecode = Assembler().assemble(Parser().parse(ABSTRACTION_SOURCE))
    vm = CodexVM()
    vm.load(bytecode, ann_input=[0.8, -0.3, 0.5, 1.0], ann_target=0.8)
    snapshot = vm.run()

    abstraction = snapshot["abstraction3d"]
    assert abstraction["halted"] is True
    assert abstraction["cycles"] == 10
    assert 1 <= abstraction["active_nodes"] <= 8
    assert len(abstraction["output"]) == 4
    assert snapshot["ledger_valid"] is True
    assert snapshot["registers"]["C"] == abstraction["active_nodes"]


def test_assembler_rejects_operands_on_3d_pipeline_opcodes():
    with pytest.raises(ValueError):
        Assembler().assemble(Parser().parse("ABSTRACT3D A"))
