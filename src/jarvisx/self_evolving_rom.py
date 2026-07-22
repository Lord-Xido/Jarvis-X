"""Bounded, semantics-preserving self-evolving ROM reference runtime.

The optimizer is deliberately narrower than unrestricted self-modification. It
profiles adjacent instruction pairs, proposes a declared macro-op rewrite, runs
baseline and candidate programs from the same deterministic snapshot, and
publishes a new immutable ROM version only when the final machine states are
identical and the instruction cost is lower.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from enum import IntEnum
from typing import Dict, List, Optional, Sequence, Tuple

WORD_MASK = (1 << 64) - 1
REGISTER_COUNT = 8
VECTOR_WIDTH = 8


class Opcode(IntEnum):
    NOP = 0x00
    LOAD3D = 0x01
    STORE3D = 0x02
    ENC = 0x03
    DEC = 0x04
    MAC3D = 0x05
    BUNDLE = 0x06
    META = 0x07
    LDC = 0x08
    DSM = 0x09
    HALT = 0x0A


@dataclass(frozen=True)
class Instruction:
    opcode: int
    rs: int = 0
    rt: int = 0
    ru: int = 0
    imm: int = 0
    addr: int = 0

    def validate(self) -> None:
        fields = (self.opcode, self.rs, self.rt, self.ru)
        if any(not 0 <= value <= 0xFF for value in fields):
            raise ValueError("opcode and register fields must fit in 8 bits")
        if any(value >= REGISTER_COUNT for value in (self.rs, self.rt, self.ru)):
            raise ValueError("register index outside the eight-register file")
        if not 0 <= self.imm <= 0xFFFF or not 0 <= self.addr <= 0xFFFF:
            raise ValueError("immediate and address fields must fit in 16 bits")
        if self.opcode not in {int(op) for op in Opcode}:
            raise ValueError("unknown opcode")

    def pack(self) -> int:
        self.validate()
        return (
            (self.opcode << 56)
            | (self.rs << 48)
            | (self.rt << 40)
            | (self.ru << 32)
            | (self.imm << 16)
            | self.addr
        ) & WORD_MASK

    @classmethod
    def unpack(cls, word: int) -> "Instruction":
        if word < 0 or word > WORD_MASK:
            raise ValueError("instruction must be an unsigned 64-bit word")
        instruction = cls(
            opcode=(word >> 56) & 0xFF,
            rs=(word >> 48) & 0xFF,
            rt=(word >> 40) & 0xFF,
            ru=(word >> 32) & 0xFF,
            imm=(word >> 16) & 0xFFFF,
            addr=word & 0xFFFF,
        )
        instruction.validate()
        return instruction

    @property
    def mnemonic(self) -> str:
        return Opcode(self.opcode).name


Vector = Tuple[float, ...]


def _zero_vector() -> Vector:
    return (0.0,) * VECTOR_WIDTH


def _deterministic_vector(address: int) -> Vector:
    """Generate stable pseudo-data without wall-clock or global randomness."""
    values = []
    state = (address ^ 0xA5A5) & 0xFFFFFFFF
    for lane in range(VECTOR_WIDTH):
        state = (1664525 * (state + lane) + 1013904223) & 0xFFFFFFFF
        values.append(((state >> 8) & 0xFFFF) / 32768.0 - 1.0)
    return tuple(values)


@dataclass(frozen=True)
class MachineSnapshot:
    registers: Tuple[Vector, ...]
    memory: Tuple[Tuple[int, Vector], ...]
    meta_rate: int

    def canonical_bytes(self) -> bytes:
        payload = {
            "registers": self.registers,
            "memory": self.memory,
            "meta_rate": self.meta_rate,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


@dataclass(frozen=True)
class ExecutionResult:
    snapshot: MachineSnapshot
    trace: Tuple[str, ...]
    cycles: int


class SpatialCore:
    """Small deterministic execution core used for equivalence checking."""

    def __init__(self, snapshot: Optional[MachineSnapshot] = None):
        if snapshot is None:
            self.registers = [_zero_vector() for _ in range(REGISTER_COUNT)]
            self.memory: Dict[int, Vector] = {}
            self.meta_rate = 0
        else:
            self.registers = [tuple(vector) for vector in snapshot.registers]
            self.memory = {address: tuple(vector) for address, vector in snapshot.memory}
            self.meta_rate = snapshot.meta_rate

    def snapshot(self) -> MachineSnapshot:
        return MachineSnapshot(
            registers=tuple(tuple(vector) for vector in self.registers),
            memory=tuple(sorted((address, tuple(vector)) for address, vector in self.memory.items())),
            meta_rate=self.meta_rate,
        )

    def _load(self, register: int, address: int) -> None:
        self.registers[register] = self.memory.get(address, _deterministic_vector(address))

    def _encode(self, source: int, target: int) -> None:
        mean = sum(self.registers[source]) / VECTOR_WIDTH
        self.registers[target] = (mean,) * VECTOR_WIDTH

    def _decode(self, source: int, target: int) -> None:
        # Reference decoder is deterministic and intentionally simple. A learned
        # backend may replace it while preserving the same transaction contract.
        self.registers[target] = tuple(self.registers[source])

    def execute(self, instruction: Instruction) -> bool:
        op = Opcode(instruction.opcode)
        if op == Opcode.NOP:
            return True
        if op == Opcode.HALT:
            return False
        if op == Opcode.LOAD3D:
            self._load(instruction.rs, instruction.addr)
        elif op == Opcode.STORE3D:
            self.memory[instruction.addr] = tuple(self.registers[instruction.rt])
        elif op == Opcode.ENC:
            self._encode(instruction.rs, instruction.rt)
        elif op == Opcode.DEC:
            self._decode(instruction.rs, instruction.rt)
        elif op == Opcode.MAC3D:
            self.registers[instruction.ru] = tuple(
                a * b for a, b in zip(self.registers[instruction.rs], self.registers[instruction.rt])
            )
        elif op == Opcode.BUNDLE:
            self.registers[instruction.ru] = tuple(
                a + b for a, b in zip(self.registers[instruction.rs], self.registers[instruction.rt])
            )
        elif op == Opcode.META:
            self.meta_rate = instruction.imm
        elif op == Opcode.LDC:
            self._load(instruction.rs, instruction.addr)
            self._encode(instruction.rs, instruction.rt)
        elif op == Opcode.DSM:
            self._decode(instruction.rs, instruction.rt)
            self.memory[instruction.addr] = tuple(self.registers[instruction.rt])
        else:  # pragma: no cover - Instruction.validate prevents this path.
            raise ValueError("unsupported opcode")
        return True

    def run(self, rom: Sequence[Instruction]) -> ExecutionResult:
        trace: List[str] = []
        cycles = 0
        for instruction in rom:
            trace.append(instruction.mnemonic)
            cycles += 1
            if not self.execute(instruction):
                break
        return ExecutionResult(self.snapshot(), tuple(trace), cycles)


@dataclass(frozen=True)
class PatchProposal:
    rule: str
    index: int
    replacement: Instruction
    removed: Instruction


@dataclass(frozen=True)
class RomVersion:
    version: int
    parent_hash: str
    manifest_hash: str
    instructions: Tuple[int, ...]
    patch_rule: str


@dataclass(frozen=True)
class EpochReport:
    epoch: int
    accepted: bool
    rule: str
    before_length: int
    after_length: int
    baseline_cycles: int
    shadow_cycles: int
    semantic_equal: bool
    analysis_share: float
    manifest_hash: str


class SelfEvolvingROM:
    """Versioned ROM optimizer with shadow verification and bounded rules."""

    def __init__(self, rom: Sequence[Instruction], analysis_cycle_fraction: float = 0.5):
        if not rom or Opcode(rom[-1].opcode) != Opcode.HALT:
            raise ValueError("ROM must terminate with HALT")
        if not 0.0 < analysis_cycle_fraction <= 0.5:
            raise ValueError("analysis cycle fraction must be in (0, 0.5]")
        self.analysis_cycle_fraction = analysis_cycle_fraction
        self.rom = tuple(rom)
        self.epoch = 0
        self.locked = False
        self.journal: List[RomVersion] = []
        self._publish_version(parent_hash="0" * 64, patch_rule="BOOT")

    @staticmethod
    def demo_rom() -> Tuple[Instruction, ...]:
        """Eight-instruction demo containing two valid adjacent fusion pairs."""
        return (
            Instruction(Opcode.LOAD3D, rs=0, addr=0x1000),
            Instruction(Opcode.ENC, rs=0, rt=2),
            Instruction(Opcode.STORE3D, rt=2, addr=0x3000),
            Instruction(Opcode.DEC, rs=2, rt=3),
            Instruction(Opcode.STORE3D, rt=3, addr=0x4000),
            Instruction(Opcode.LOAD3D, rs=1, addr=0x2000),
            Instruction(Opcode.META, rs=0, imm=0x000A),
            Instruction(Opcode.HALT),
        )

    @staticmethod
    def profile(trace: Sequence[str]) -> Dict[str, int]:
        return dict(sorted(Counter(trace).items()))

    @staticmethod
    def _manifest(parent_hash: str, words: Sequence[int], rule: str) -> str:
        payload = bytearray.fromhex(parent_hash)
        payload.extend(rule.encode("utf-8"))
        for word in words:
            payload.extend(int(word).to_bytes(8, "big"))
        return hashlib.sha256(bytes(payload)).hexdigest()

    def _publish_version(self, parent_hash: str, patch_rule: str) -> None:
        words = tuple(instruction.pack() for instruction in self.rom)
        manifest = self._manifest(parent_hash, words, patch_rule)
        self.journal.append(
            RomVersion(
                version=len(self.journal),
                parent_hash=parent_hash,
                manifest_hash=manifest,
                instructions=words,
                patch_rule=patch_rule,
            )
        )

    def _candidate(self) -> Optional[PatchProposal]:
        for index in range(len(self.rom) - 1):
            first, second = self.rom[index], self.rom[index + 1]
            first_op, second_op = Opcode(first.opcode), Opcode(second.opcode)
            if first_op == Opcode.LOAD3D and second_op == Opcode.ENC and first.rs == second.rs:
                return PatchProposal(
                    rule="FUSE_LOAD3D_ENC_TO_LDC",
                    index=index,
                    replacement=Instruction(
                        Opcode.LDC, rs=first.rs, rt=second.rt, addr=first.addr
                    ),
                    removed=second,
                )
            if first_op == Opcode.DEC and second_op == Opcode.STORE3D and first.rt == second.rt:
                return PatchProposal(
                    rule="FUSE_DEC_STORE3D_TO_DSM",
                    index=index,
                    replacement=Instruction(
                        Opcode.DSM, rs=first.rs, rt=first.rt, addr=second.addr
                    ),
                    removed=second,
                )
        return None

    def _apply(self, proposal: PatchProposal) -> Tuple[Instruction, ...]:
        candidate = list(self.rom)
        candidate[proposal.index] = proposal.replacement
        del candidate[proposal.index + 1]
        return tuple(candidate)

    @staticmethod
    def _run_from(snapshot: MachineSnapshot, rom: Sequence[Instruction]) -> ExecutionResult:
        return SpatialCore(snapshot).run(rom)

    def inward_turn(self) -> EpochReport:
        if self.locked:
            latest = self.journal[-1]
            return EpochReport(
                epoch=self.epoch,
                accepted=False,
                rule="FIXED_POINT",
                before_length=len(self.rom),
                after_length=len(self.rom),
                baseline_cycles=0,
                shadow_cycles=0,
                semantic_equal=True,
                analysis_share=0.0,
                manifest_hash=latest.manifest_hash,
            )

        self.epoch += 1
        initial = SpatialCore().snapshot()
        baseline = self._run_from(initial, self.rom)
        proposal = self._candidate()
        if proposal is None:
            self.locked = True
            latest = self.journal[-1]
            return EpochReport(
                epoch=self.epoch,
                accepted=False,
                rule="FIXED_POINT",
                before_length=len(self.rom),
                after_length=len(self.rom),
                baseline_cycles=baseline.cycles,
                shadow_cycles=0,
                semantic_equal=True,
                analysis_share=0.0,
                manifest_hash=latest.manifest_hash,
            )

        candidate_rom = self._apply(proposal)
        shadow = self._run_from(initial, candidate_rom)
        semantic_equal = baseline.snapshot == shadow.snapshot
        lower_cost = shadow.cycles < baseline.cycles and len(candidate_rom) < len(self.rom)
        analysis_share = shadow.cycles / float(baseline.cycles + shadow.cycles)
        within_budget = analysis_share <= self.analysis_cycle_fraction
        accepted = semantic_equal and lower_cost and within_budget

        if accepted:
            parent_hash = self.journal[-1].manifest_hash
            self.rom = candidate_rom
            self._publish_version(parent_hash=parent_hash, patch_rule=proposal.rule)

        return EpochReport(
            epoch=self.epoch,
            accepted=accepted,
            rule=proposal.rule,
            before_length=len(self.rom) + (1 if accepted else 0),
            after_length=len(self.rom),
            baseline_cycles=baseline.cycles,
            shadow_cycles=shadow.cycles,
            semantic_equal=semantic_equal,
            analysis_share=analysis_share,
            manifest_hash=self.journal[-1].manifest_hash,
        )

    def optimize(self, max_epochs: int = 8) -> Dict[str, object]:
        if max_epochs < 1:
            raise ValueError("max_epochs must be positive")
        reports: List[EpochReport] = []
        for _ in range(max_epochs):
            report = self.inward_turn()
            reports.append(report)
            if self.locked:
                break
        final_run = SpatialCore().run(self.rom)
        return {
            "epochs": [asdict(report) for report in reports],
            "locked": self.locked,
            "rom_length": len(self.rom),
            "rom": [
                {
                    "address": index,
                    "word": "0x%016X" % instruction.pack(),
                    "opcode": instruction.mnemonic,
                }
                for index, instruction in enumerate(self.rom)
            ],
            "trace_profile": self.profile(final_run.trace),
            "versions": [asdict(version) for version in self.journal],
            "final_state_hash": hashlib.sha256(final_run.snapshot.canonical_bytes()).hexdigest(),
        }


def run_self_evolving_rom(max_epochs: int = 8) -> Dict[str, object]:
    return SelfEvolvingROM(SelfEvolvingROM.demo_rom()).optimize(max_epochs=max_epochs)
