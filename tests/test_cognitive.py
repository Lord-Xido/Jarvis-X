import math

import pytest

from jarvisx.cognitive import (
    CognitiveConfig,
    CognitiveKernel,
    CognitiveVMBridge,
    quantize_q3,
)
from jarvisx.core import CodexVM
from jarvisx.registers import Registers


def test_q3_quantization_is_bounded_and_signed():
    assert quantize_q3(-99) == -4
    assert quantize_q3(-1.5) == -2
    assert quantize_q3(1.5) == 2
    assert quantize_q3(99) == 3


def test_q3_quantization_rejects_non_finite_values():
    with pytest.raises(ValueError, match="finite"):
        quantize_q3(math.inf)
    with pytest.raises(ValueError, match="finite"):
        quantize_q3(math.nan)


def test_hierarchy_condenses_to_single_root():
    kernel = CognitiveKernel(CognitiveConfig(branch_factor=2))
    result = kernel.step([3, 2, 1, 0, -1, -2, -3, -4])
    assert result.committed
    assert result.hierarchy[0] == (3, 2, 1, 0, -1, -2, -3, -4)
    assert len(result.hierarchy[-1]) == 1
    assert result.metrics["condensation_ratio"] == 8.0


def test_hierarchy_enforces_maximum_depth_before_allocation_continues():
    kernel = CognitiveKernel(CognitiveConfig(branch_factor=2, max_levels=2))
    with pytest.raises(ValueError, match="max_levels"):
        kernel.step([1, 1, 1, 1])
    assert kernel.snapshot()["state_hash"] == "GENESIS"


def test_prediction_error_updates_cumulative_memory():
    kernel = CognitiveKernel()
    first = kernel.step([3, 3, 3, 3])
    second = kernel.step([0, 0, 0, 0])
    assert first.omega_after == (2, 2, 2, 2)
    assert second.prediction == (3, 3, 3, 3)
    assert second.residual == (-3, -3, -3, -3)
    assert second.omega_after == (0, 0, 0, 0)


def test_same_stream_produces_same_hash_chain():
    stream = [[1, 2, 3, -4], [1, 1, 2, -3], [0, 1, 1, -2]]
    left = CognitiveKernel().run(stream)
    right = CognitiveKernel().run(stream)
    assert [item.state_hash for item in left] == [item.state_hash for item in right]


def test_lambda_rejection_rolls_back_committed_state():
    kernel = CognitiveKernel(CognitiveConfig(max_residual_l1=0))
    before = kernel.snapshot()
    result = kernel.step([1, 0, 0, 0])
    assert not result.committed
    assert result.reason == "residual budget exceeded"
    assert kernel.snapshot() == before
    assert result.state_hash == "GENESIS"


def test_vm_bridge_maps_cycle_to_existing_registers():
    regs = Registers()
    bridge = CognitiveVMBridge(regs)
    result = bridge.cycle([3, 1, -1, -3])
    assert result.committed
    assert regs["Λ"] == 1
    assert regs["Ψ"] == result.hierarchy[-1][0]
    assert regs["𝒮"] == int(result.metrics["residual_l1"])


def test_vm_bridge_rejection_does_not_leak_candidate_registers():
    regs = Registers()
    kernel = CognitiveKernel(CognitiveConfig(max_residual_l1=4))
    bridge = CognitiveVMBridge(regs, kernel)

    committed = bridge.cycle([1, 1, 1, 1])
    assert committed.committed
    before = regs.snapshot()

    rejected = bridge.cycle([-4, -4, -4, -4])
    after = regs.snapshot()

    assert not rejected.committed
    assert rejected.reason == "residual budget exceeded"
    assert rejected.state_hash == committed.state_hash
    assert after["Λ"] == 0
    for name, value in before.items():
        if name != "Λ":
            assert after[name] == value


def test_vm_rolls_back_kernel_and_registers_on_projection_failure():
    class FailingRegisters(Registers):
        def __init__(self):
            super().__init__()
            self.fail_once = False

        def __setitem__(self, key, value):
            if self.fail_once and key == "Ω":
                self.fail_once = False
                raise RuntimeError("simulated register failure")
            super().__setitem__(key, value)

    vm = CodexVM()
    failing = FailingRegisters()
    vm.regs = failing
    vm.cognitive_bridge.registers = failing
    before_state = vm.cognitive.snapshot()
    before_registers = failing.snapshot()
    failing.fail_once = True

    with pytest.raises(RuntimeError, match="simulated register failure"):
        vm.cognitive_cycle([3, 1, -1, -3])

    assert vm.cognitive.snapshot() == before_state
    assert vm.cognitive.journal == []
    assert failing.snapshot() == before_registers


def test_configuration_rejects_unbounded_update_ratios():
    with pytest.raises(ValueError, match="retention ratio"):
        CognitiveKernel(CognitiveConfig(retention_numerator=9, retention_denominator=8))
    with pytest.raises(ValueError, match="non-negative"):
        CognitiveKernel(CognitiveConfig(learning_numerator=-1))
