from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Callable

import numpy as np

from .ann import TinyAutoencoder
from .geometry import Address3D, GeometricProgramLayout
from .isa import Instruction, Opcode
from .optimizer import AutoOptimizer, RuntimeMetrics, RuntimePolicy
from .rom import Sparse3DROM


@dataclass(slots=True)
class VMConfig:
    max_cycles: int = 10_000
    lambda_root_mask: int = 0xFFFF
    train_on_update: bool = True
    optimizer_enabled: bool = True
    deterministic_seed: int = 7


@dataclass(slots=True)
class VMResult:
    halted: bool
    cycles: int
    output: list[list[float]] | None
    latent: list[list[float]] | None
    residual: list[list[float]] | None
    metrics: dict[str, float | int]
    policy: dict[str, float | int]
    journal: list[dict[str, object]]


@dataclass(slots=True)
class OmegaMemory:
    working_bias: np.ndarray
    residual_ema: np.ndarray
    episodes: list[dict[str, object]] = field(default_factory=list)
    semantic_updates: int = 0


class VANNVirtualMachine:
    """Reference VANN-ROM Ω³ bytecode virtual machine."""

    def __init__(
        self,
        model: TinyAutoencoder,
        *,
        rom: Sparse3DROM | None = None,
        config: VMConfig | None = None,
        output_sink: Callable[[str], None] | None = None,
    ) -> None:
        self.model = model
        self.config = config or VMConfig()
        self.rom = rom or Sparse3DROM(GeometricProgramLayout())
        self.output_sink = output_sink or (lambda text: None)
        self.optimizer = AutoOptimizer(RuntimePolicy(learning_rate=model.learning_rate))
        self.metrics = RuntimeMetrics()
        self.reset()

    def reset(self) -> None:
        self.pc_index = 0
        self.pc3 = Address3D(0, 0, 0)
        self.halted = False
        self.cycles = 0
        self.registers: dict[str, object] = {
            "X": None,
            "Z": None,
            "P": None,
            "R": None,
            "E": None,
            "Y": None,
            "G": None,
        }
        self.transaction: dict[str, object] = {}
        self.cache: dict[int, Instruction] = {}
        self.journal: list[dict[str, object]] = []
        self.omega = OmegaMemory(
            working_bias=np.zeros(self.model.input_dim, dtype=np.float32),
            residual_ema=np.zeros(self.model.input_dim, dtype=np.float32),
        )
        self.metrics = RuntimeMetrics()

    def load_program(self, instructions: list[Instruction]) -> None:
        if not instructions:
            raise ValueError("program cannot be empty")
        self.rom.load_program(instructions)
        self.pc_index = 0
        self.pc3 = self.rom.instruction_address(0)
        self.halted = False
        self.cache.clear()

    def set_input(self, data: np.ndarray | list[float] | list[list[float]]) -> None:
        x = np.asarray(data, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
        if x.ndim != 2 or x.shape[1] != self.model.input_dim:
            raise ValueError(f"input must have shape (batch, {self.model.input_dim})")
        self.registers["X"] = np.clip(x, 0.0, 1.0)
        self.registers["R"] = np.clip(x.copy(), 0.0, 1.0)

    def run(self) -> VMResult:
        if self.registers["X"] is None:
            raise RuntimeError("set_input must be called before run")
        started = time.perf_counter()
        while not self.halted and self.cycles < self.config.max_cycles:
            self.step()
        self.metrics.elapsed_seconds += time.perf_counter() - started
        if not self.halted:
            raise RuntimeError("cycle limit reached before HALT")
        return self._result()

    def step(self) -> None:
        instruction = self._fetch(self.pc_index)
        self.cycles += 1
        self.metrics.cycles += 1
        self.metrics.instructions += 1
        next_pc = self.pc_index + 1

        op = instruction.opcode
        if op == Opcode.NOP:
            pass
        elif op == Opcode.LOAD_INPUT:
            self._require("X")
        elif op == Opcode.NORMALIZE:
            x = self._require_array("X")
            lo = x.min(axis=1, keepdims=True)
            hi = x.max(axis=1, keepdims=True)
            self.registers["X"] = (x - lo) / np.maximum(hi - lo, 1e-6)
            self.registers["R"] = self.registers["X"].copy()
        elif op == Opcode.VOXELIZE:
            x = self._require_array("X")
            active = np.argwhere(x > self.optimizer.policy.sparsity_threshold)
            self.registers["G"] = active.tolist()
        elif op == Opcode.PREFETCH3D:
            depth = self.optimizer.policy.prefetch_depth
            for index in range(self.pc_index + 1, min(self.pc_index + 1 + depth, len(self.rom))):
                if index not in self.cache:
                    self.cache[index] = self.rom.fetch_instruction(self.rom.instruction_address(index))
                    self.metrics.rom_fetches += 1
        elif op == Opcode.ENCODE3D:
            self.registers["Z"] = self.model.encode(self._require_array("X"))
        elif op == Opcode.PREDICT:
            z = self._require_array("Z")
            prediction = self.model.decode(z)
            prediction = np.clip(prediction + self.omega.working_bias, 0.0, 1.0)
            self.registers["P"] = prediction
        elif op == Opcode.COMPARE:
            residual = self._require_array("R") - self._require_array("P")
            self.registers["E"] = residual
            self.metrics.reconstruction_error = float(np.mean(np.abs(residual)))
        elif op == Opcode.UPDATE_OMEGA:
            residual = self._require_array("E")
            mean_residual = residual.mean(axis=0)
            self.omega.residual_ema = 0.95 * self.omega.residual_ema + 0.05 * mean_residual
            candidate_bias = np.clip(
                self.omega.working_bias + self.optimizer.policy.learning_rate * self.omega.residual_ema,
                -0.25,
                0.25,
            )
            self.transaction["omega_bias"] = candidate_bias
            if self.config.train_on_update:
                train = self.model.train_step(self._require_array("X"))
                self.metrics.loss = train.loss
        elif op == Opcode.DECODE3D:
            decoded = self.model.decode(self._require_array("Z"))
            self.registers["Y"] = np.clip(decoded + self.omega.working_bias, 0.0, 1.0)
        elif op == Opcode.PROJECT_LAMBDA:
            if instruction.lambda_mask & self.config.lambda_root_mask != instruction.lambda_mask:
                self.transaction.clear()
                self.metrics.rollbacks += 1
                raise PermissionError("instruction violates Λ-root mask")
            if "omega_bias" in self.transaction:
                self.transaction["omega_bias"] = np.clip(
                    np.asarray(self.transaction["omega_bias"], dtype=np.float32), -0.25, 0.25
                )
            if self.registers["Y"] is not None:
                self.registers["Y"] = np.clip(self._require_array("Y"), 0.0, 1.0)
        elif op == Opcode.STAGE:
            if self.registers["Y"] is not None:
                self.transaction["Y"] = self._require_array("Y").copy()
        elif op == Opcode.VERIFY:
            self._verify_transaction()
        elif op == Opcode.COMMIT:
            self._commit_transaction()
        elif op == Opcode.RENDER:
            y = self._require_array("Y")
            self.output_sink(np.array2string(y, precision=4, suppress_small=True))
        elif op == Opcode.SAMPLE_METRICS:
            z = self._require_array("Z")
            self.metrics.latent_sparsity = float(np.mean(np.abs(z) <= 1e-8))
        elif op == Opcode.OPTIMIZE_POLICY:
            if self.config.optimizer_enabled:
                candidate = self.optimizer.propose(self.metrics)
                self.optimizer.commit(candidate)
                self.model.learning_rate = candidate.learning_rate
        elif op == Opcode.JOURNAL:
            self._append_journal(instruction)
        elif op == Opcode.ADVANCE:
            pass
        elif op == Opcode.JMP3D:
            target = instruction.immediate
            if target < 0 or target >= len(self.rom):
                raise IndexError(f"jump target {target} outside program")
            next_pc = target
        elif op == Opcode.HALT:
            self.halted = True
        else:
            raise NotImplementedError(op)

        if not self.halted:
            if next_pc >= len(self.rom):
                raise RuntimeError("program counter left ROM without HALT")
            self.pc_index = next_pc
            self.pc3 = self.rom.instruction_address(next_pc)

    def _fetch(self, index: int) -> Instruction:
        if index in self.cache:
            self.metrics.cache_hits += 1
            return self.cache.pop(index)
        self.metrics.rom_fetches += 1
        return self.rom.fetch_instruction(self.rom.instruction_address(index))

    def _require(self, name: str) -> object:
        value = self.registers.get(name)
        if value is None:
            raise RuntimeError(f"register {name} is empty")
        return value

    def _require_array(self, name: str) -> np.ndarray:
        value = self._require(name)
        if not isinstance(value, np.ndarray):
            raise TypeError(f"register {name} does not contain a tensor")
        return value

    def _verify_transaction(self) -> None:
        for name, value in self.transaction.items():
            if isinstance(value, np.ndarray) and not np.all(np.isfinite(value)):
                self.metrics.rollbacks += 1
                self.transaction.clear()
                raise FloatingPointError(f"non-finite value in staged field {name}")

    def _commit_transaction(self) -> None:
        if "omega_bias" in self.transaction:
            self.omega.working_bias = np.asarray(self.transaction["omega_bias"], dtype=np.float32)
            self.omega.semantic_updates += 1
        if "Y" in self.transaction:
            self.registers["Y"] = np.asarray(self.transaction["Y"], dtype=np.float32)
        self.transaction.clear()
        self.metrics.commits += 1

    def _append_journal(self, instruction: Instruction) -> None:
        entry = {
            "cycle": self.cycles,
            "pc": self.pc_index,
            "pc3": asdict(self.pc3),
            "opcode": instruction.opcode.name,
            "reconstruction_error": self.metrics.reconstruction_error,
            "loss": self.metrics.loss,
            "policy": asdict(self.optimizer.policy),
        }
        self.journal.append(entry)
        self.omega.episodes.append(entry)

    def _result(self) -> VMResult:
        def optional_list(value: object) -> list[list[float]] | None:
            return value.tolist() if isinstance(value, np.ndarray) else None

        return VMResult(
            halted=self.halted,
            cycles=self.cycles,
            output=optional_list(self.registers["Y"]),
            latent=optional_list(self.registers["Z"]),
            residual=optional_list(self.registers["E"]),
            metrics=self.metrics.as_dict(),
            policy=asdict(self.optimizer.policy),
            journal=list(self.journal),
        )

    def state_json(self) -> str:
        return json.dumps(asdict(self._result()), indent=2)
