from jarvisx.cognitive import (
    CognitiveConfig,
    CognitiveKernel,
    CognitiveVMBridge,
    quantize_q3,
)
from jarvisx.registers import Registers


def test_q3_quantization_is_bounded_and_signed():
    assert quantize_q3(-99) == -4
    assert quantize_q3(-1.5) == -2
    assert quantize_q3(1.5) == 2
    assert quantize_q3(99) == 3


def test_hierarchy_condenses_to_single_root():
    kernel = CognitiveKernel(CognitiveConfig(branch_factor=2))
    result = kernel.step([3, 2, 1, 0, -1, -2, -3, -4])
    assert result.committed
    assert result.hierarchy[0] == (3, 2, 1, 0, -1, -2, -3, -4)
    assert len(result.hierarchy[-1]) == 1
    assert result.metrics["condensation_ratio"] == 8.0


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
