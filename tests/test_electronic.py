from types import SimpleNamespace

from jarvisx.electronic import ElectronicConfig, ElectronicSubstrate


def instruction(opcode, dst=0, src1=0, src2=0, imm=0):
    return SimpleNamespace(opcode=opcode, dst=dst, src1=src1, src2=src2, imm=imm)


def test_add_generates_deterministic_electronic_trace():
    substrate = ElectronicSubstrate()
    before = {"A": 0, "B": 10, "C": 20, "IP": 2}
    after = {"A": 30, "B": 10, "C": 20, "IP": 2}

    trace = substrate.tick(instruction(0x03, 8, 9, 10), before, after)

    assert trace.opcode == 0x03
    assert trace.register_bit_transitions == 4
    assert trace.gate_toggles["XOR"] == 2 * substrate.config.word_bits
    assert trace.total_gate_toggles > trace.register_bit_transitions
    assert trace.source == "deterministic-model"
    assert substrate.snapshot()["telemetry_is_measured"] is False


def test_electronic_lambda_gate_detects_timing_violation():
    substrate = ElectronicSubstrate(
        ElectronicConfig(clock_hz=10_000_000_000.0, enforce_limits=True)
    )
    trace = substrate.tick(instruction(0x03), {"A": 0}, {"A": 1})

    assert trace.timing_ok is False
    assert trace.lambda_accept is False


def test_trace_depth_is_bounded():
    substrate = ElectronicSubstrate(ElectronicConfig(trace_depth=2))
    for value in range(3):
        substrate.tick(instruction(0x01, imm=value), {"A": value}, {"A": value + 1})

    assert len(substrate.trace) == 2
    assert substrate.trace[-1].cycle == 3
