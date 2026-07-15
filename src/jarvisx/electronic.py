"""Deterministic electronic-substrate model for Jarvis-X.

This module does not claim access to physical chip telemetry. It converts VM
state transitions into an auditable estimate of bus activity, gate switching,
timing, energy, power, and thermal evolution.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Optional


def _word(value: int, bits: int) -> int:
    return int(value) & ((1 << bits) - 1)


def _hamming_distance(left: int, right: int, bits: int) -> int:
    return bin(_word(left, bits) ^ _word(right, bits)).count("1")


@dataclass(frozen=True)
class ElectronicConfig:
    """Physical assumptions used by the deterministic substrate model."""

    word_bits: int = 64
    supply_voltage_v: float = 0.90
    clock_hz: float = 1_000_000_000.0
    switched_capacitance_f: float = 2.0e-15
    gate_delay_ps: float = 12.0
    static_power_w: float = 0.50
    ambient_temp_c: float = 25.0
    thermal_resistance_c_per_w: float = 0.35
    thermal_time_constant_s: float = 0.05
    max_junction_temp_c: float = 95.0
    timing_guard_fraction: float = 0.10
    trace_depth: int = 4096
    enforce_limits: bool = False

    def __post_init__(self) -> None:
        if self.word_bits <= 0:
            raise ValueError("word_bits must be positive")
        if self.supply_voltage_v <= 0.0:
            raise ValueError("supply_voltage_v must be positive")
        if self.clock_hz <= 0.0:
            raise ValueError("clock_hz must be positive")
        if self.switched_capacitance_f < 0.0:
            raise ValueError("switched_capacitance_f cannot be negative")
        if self.thermal_time_constant_s <= 0.0:
            raise ValueError("thermal_time_constant_s must be positive")
        if not 0.0 <= self.timing_guard_fraction < 1.0:
            raise ValueError("timing_guard_fraction must be in [0, 1)")
        if self.trace_depth <= 0:
            raise ValueError("trace_depth must be positive")


@dataclass(frozen=True)
class ElectronicTelemetry:
    cycle: int
    opcode: int
    register_bit_transitions: int
    instruction_bus_bit_transitions: int
    gate_toggles: Dict[str, int]
    total_gate_toggles: int
    dynamic_energy_j: float
    cumulative_energy_j: float
    dynamic_power_w: float
    total_power_w: float
    junction_temp_c: float
    critical_path_ns: float
    clock_period_ns: float
    timing_margin_ns: float
    timing_ok: bool
    thermal_ok: bool
    lambda_accept: bool
    register_checksum: str
    source: str = "deterministic-model"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ElectronicSubstrate:
    """Maps one VM instruction commit to an electronic state-transition trace."""

    _OPCODE_NAMES = {0x01: "SET", 0x03: "ADD", 0x04: "SUB", 0x0A: "HALT"}

    def __init__(self, config: Optional[ElectronicConfig] = None) -> None:
        self.config = config or ElectronicConfig()
        self.trace: List[ElectronicTelemetry] = []
        self.reset()

    def reset(self) -> None:
        self.cycle = 0
        self.junction_temp_c = self.config.ambient_temp_c
        self.cumulative_energy_j = 0.0
        self._previous_instruction_word = 0
        self.trace.clear()

    @property
    def last(self) -> Optional[ElectronicTelemetry]:
        return self.trace[-1] if self.trace else None

    def _instruction_word(self, instr: Any) -> int:
        return (
            (int(instr.opcode) << 56)
            | (int(getattr(instr, "dst", 0)) << 40)
            | (int(getattr(instr, "src1", 0)) << 32)
            | (int(getattr(instr, "src2", 0)) << 24)
            | (int(getattr(instr, "imm", 0)) << 8)
        )

    def _register_transitions(
        self, before: Mapping[str, int], after: Mapping[str, int]
    ) -> int:
        keys = set(before) | set(after)
        return sum(
            _hamming_distance(before.get(key, 0), after.get(key, 0), self.config.word_bits)
            for key in keys
        )

    def _gate_activity(self, opcode: int, transitions: int) -> Dict[str, int]:
        width = self.config.word_bits
        activity = {
            "AND": 0,
            "OR": 0,
            "XOR": 0,
            "NAND": 8,
            "NOT": 0,
            "MUX": width,
            "FF": transitions,
        }

        if opcode == 0x01:
            activity["NAND"] += 16
            activity["MUX"] += width
        elif opcode == 0x03:
            activity["XOR"] += 2 * width
            activity["AND"] += 2 * width
            activity["OR"] += width
        elif opcode == 0x04:
            activity["NOT"] += width
            activity["XOR"] += 2 * width
            activity["AND"] += 2 * width
            activity["OR"] += width
        elif opcode == 0x0A:
            activity["NAND"] += 4
            activity["MUX"] = 4
        else:
            activity["NAND"] += 24
            activity["MUX"] += width

        return activity

    def _critical_path_ns(self, opcode: int) -> float:
        gate_ns = self.config.gate_delay_ps / 1000.0
        if opcode in (0x03, 0x04):
            # Conservative ripple-carry estimate. A later backend may substitute
            # carry-lookahead or measured hardware timing without changing the API.
            return (self.config.word_bits + 3) * gate_ns
        if opcode == 0x01:
            return 3 * gate_ns
        if opcode == 0x0A:
            return gate_ns
        return 5 * gate_ns

    def _checksum(self, registers: Mapping[str, int]) -> str:
        # Stable 64-bit FNV-1a-like checksum over sorted register names and values.
        value = 0xCBF29CE484222325
        for name in sorted(registers):
            payload = (name + "=" + str(int(registers[name]))).encode("utf-8")
            for byte in payload:
                value ^= byte
                value = (value * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
        return "{:016x}".format(value)

    def tick(
        self,
        instr: Any,
        before: Mapping[str, int],
        after: Mapping[str, int],
        instruction_word: Optional[int] = None,
    ) -> ElectronicTelemetry:
        """Commit one deterministic electronic trace for a VM instruction."""

        opcode = int(instr.opcode)
        word = self._instruction_word(instr) if instruction_word is None else int(instruction_word)
        register_transitions = self._register_transitions(before, after)
        instruction_transitions = _hamming_distance(
            self._previous_instruction_word, word, self.config.word_bits
        )
        gate_toggles = self._gate_activity(opcode, register_transitions)
        total_gate_toggles = sum(gate_toggles.values()) + instruction_transitions

        clock_period_s = 1.0 / self.config.clock_hz
        clock_period_ns = clock_period_s * 1.0e9
        dynamic_energy_j = (
            total_gate_toggles
            * self.config.switched_capacitance_f
            * self.config.supply_voltage_v
            * self.config.supply_voltage_v
        )
        dynamic_power_w = dynamic_energy_j / clock_period_s
        total_power_w = self.config.static_power_w + dynamic_power_w
        self.cumulative_energy_j += dynamic_energy_j + self.config.static_power_w * clock_period_s

        thermal_target = (
            self.config.ambient_temp_c
            + total_power_w * self.config.thermal_resistance_c_per_w
        )
        thermal_alpha = min(1.0, clock_period_s / self.config.thermal_time_constant_s)
        self.junction_temp_c += thermal_alpha * (thermal_target - self.junction_temp_c)

        critical_path_ns = self._critical_path_ns(opcode)
        usable_period_ns = clock_period_ns * (1.0 - self.config.timing_guard_fraction)
        timing_margin_ns = usable_period_ns - critical_path_ns
        timing_ok = timing_margin_ns >= 0.0
        thermal_ok = self.junction_temp_c <= self.config.max_junction_temp_c
        lambda_accept = timing_ok and thermal_ok

        self.cycle += 1
        telemetry = ElectronicTelemetry(
            cycle=self.cycle,
            opcode=opcode,
            register_bit_transitions=register_transitions,
            instruction_bus_bit_transitions=instruction_transitions,
            gate_toggles=gate_toggles,
            total_gate_toggles=total_gate_toggles,
            dynamic_energy_j=dynamic_energy_j,
            cumulative_energy_j=self.cumulative_energy_j,
            dynamic_power_w=dynamic_power_w,
            total_power_w=total_power_w,
            junction_temp_c=self.junction_temp_c,
            critical_path_ns=critical_path_ns,
            clock_period_ns=clock_period_ns,
            timing_margin_ns=timing_margin_ns,
            timing_ok=timing_ok,
            thermal_ok=thermal_ok,
            lambda_accept=lambda_accept,
            register_checksum=self._checksum(after),
        )
        self.trace.append(telemetry)
        if len(self.trace) > self.config.trace_depth:
            del self.trace[: len(self.trace) - self.config.trace_depth]
        self._previous_instruction_word = word
        return telemetry

    def snapshot(self) -> Dict[str, Any]:
        """Return a JSON-serializable state summary."""

        return {
            "model": "Jarvis-X deterministic electronic substrate",
            "telemetry_is_measured": False,
            "cycle": self.cycle,
            "junction_temp_c": self.junction_temp_c,
            "cumulative_energy_j": self.cumulative_energy_j,
            "last": self.last.to_dict() if self.last else None,
        }

    def opcode_name(self, opcode: int) -> str:
        return self._OPCODE_NAMES.get(int(opcode), "UNKNOWN")
