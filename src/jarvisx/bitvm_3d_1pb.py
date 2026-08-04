"""Sparse transactional 3D bit virtual machine with a canonical 1 PB address space.

The default geometry is a ``1000 x 1000 x 1000`` lattice of decimal 1 MB bricks:

    1_000_000_000 bricks x 1_000_000 bytes = 1_000_000_000_000_000 bytes

Only bricks containing non-zero committed data are materialized. Missing bricks are an immutable
zero background. Every mutating instruction is copy-on-write, capability checked, journaled, and
committed atomically or rejected without changing authoritative state.

This module is a bounded correctness reference. It does not allocate one petabyte, provide process
isolation for hostile code, or claim production throughput.
"""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import asdict, dataclass
from enum import Enum, IntEnum
from typing import Any, Dict, Iterable, Iterator, Mapping, Sequence, Tuple

DECIMAL_MEGABYTE = 1_000_000
CANONICAL_SIDE = 1000
CANONICAL_BRICKS = CANONICAL_SIDE**3
CANONICAL_VIRTUAL_BYTES = CANONICAL_BRICKS * DECIMAL_MEGABYTE
CANONICAL_VIRTUAL_BITS = CANONICAL_VIRTUAL_BYTES * 8

BrickKey = Tuple[int, int, int, int]  # (asid, x, y, z)
_READABLE_CLASSES = frozenset((0, 2, 3, 4))
_WRITABLE_CLASSES = frozenset((1, 3, 4))
_HEX_DIGEST_LENGTH = 64


class AddressClass(IntEnum):
    """Three-bit access-class field carried by every packed virtual address."""

    READ = 0
    WRITE = 1
    EXECUTE = 2
    JOURNAL = 3
    CONTROL = 4


class BitOpcode(str, Enum):
    """Bounded reference instruction set for bit-addressed operations."""

    BSET = "BSET"
    BCLR = "BCLR"
    BCOPY = "BCOPY"
    BNOT = "BNOT"
    BAND = "BAND"
    BOR = "BOR"
    BXOR = "BXOR"
    BPOPCNT = "BPOPCNT"
    BHASH = "BHASH"


@dataclass(frozen=True)
class BitVMConfig:
    """Geometry and execution bounds for one sparse VM instance."""

    side: int = CANONICAL_SIDE
    brick_bytes: int = DECIMAL_MEGABYTE
    max_resident_bricks: int = 64
    max_instruction_bits: int = 65_536
    vector_line_bytes: int = 64
    prune_zero_bricks: bool = True

    def __post_init__(self) -> None:
        for name in (
            "side",
            "brick_bytes",
            "max_resident_bricks",
            "max_instruction_bits",
            "vector_line_bytes",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.side > CANONICAL_SIDE:
            raise ValueError("side must fit the canonical 10-bit coordinate field")
        if self.brick_bytes > DECIMAL_MEGABYTE:
            raise ValueError("brick_bytes must not exceed the canonical decimal 1 MB brick")
        if self.vector_line_bytes > self.brick_bytes:
            raise ValueError("vector_line_bytes must not exceed brick_bytes")
        if self.brick_bytes % self.vector_line_bytes != 0:
            raise ValueError("brick_bytes must be divisible by vector_line_bytes")
        if not isinstance(self.prune_zero_bricks, bool):
            raise TypeError("prune_zero_bricks must be a boolean")

    @property
    def virtual_bricks(self) -> int:
        return self.side**3

    @property
    def virtual_bytes(self) -> int:
        return self.virtual_bricks * self.brick_bytes

    @property
    def virtual_bits(self) -> int:
        return self.virtual_bytes * 8

    @property
    def lines_per_brick(self) -> int:
        return self.brick_bytes // self.vector_line_bytes


@dataclass(frozen=True)
class BitAddress:
    """Canonical packed 64-bit address.

    Layout, most-significant field first::

        ASID[8] | CLASS[3] | Z[10] | Y[10] | X[10] | BYTE[20] | BIT[3]
    """

    asid: int
    access_class: int
    x: int
    y: int
    z: int
    byte_offset: int
    bit_offset: int

    def __post_init__(self) -> None:
        fields = {
            "asid": (self.asid, 0, 255),
            "access_class": (self.access_class, 0, 7),
            "x": (self.x, 0, 999),
            "y": (self.y, 0, 999),
            "z": (self.z, 0, 999),
            "byte_offset": (self.byte_offset, 0, DECIMAL_MEGABYTE - 1),
            "bit_offset": (self.bit_offset, 0, 7),
        }
        for name, (value, minimum, maximum) in fields.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < minimum or value > maximum:
                raise ValueError(f"{name} must be in [{minimum}, {maximum}]")

    @property
    def brick_key(self) -> BrickKey:
        return self.asid, self.x, self.y, self.z

    def pack(self) -> int:
        return (
            (self.asid << 56)
            | (self.access_class << 53)
            | (self.z << 43)
            | (self.y << 33)
            | (self.x << 23)
            | (self.byte_offset << 3)
            | self.bit_offset
        )

    @classmethod
    def unpack(cls, value: int) -> "BitAddress":
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("packed address must be an integer")
        if value < 0 or value >= 1 << 64:
            raise ValueError("packed address must be an unsigned 64-bit integer")
        return cls(
            asid=(value >> 56) & 0xFF,
            access_class=(value >> 53) & 0x7,
            z=(value >> 43) & 0x3FF,
            y=(value >> 33) & 0x3FF,
            x=(value >> 23) & 0x3FF,
            byte_offset=(value >> 3) & 0xFFFFF,
            bit_offset=value & 0x7,
        )


@dataclass(frozen=True)
class Capability:
    """Address and opcode authority granted to one VM caller."""

    allowed_asids: Tuple[int, ...] = (0,)
    allowed_classes: Tuple[int, ...] = tuple(int(item) for item in AddressClass)
    allowed_opcodes: Tuple[BitOpcode, ...] = tuple(BitOpcode)
    x_range: Tuple[int, int] = (0, 999)
    y_range: Tuple[int, int] = (0, 999)
    z_range: Tuple[int, int] = (0, 999)
    max_accessed_bricks: int = 64

    def __post_init__(self) -> None:
        if not self.allowed_asids:
            raise ValueError("allowed_asids must not be empty")
        if not self.allowed_classes:
            raise ValueError("allowed_classes must not be empty")
        if not self.allowed_opcodes:
            raise ValueError("allowed_opcodes must not be empty")
        for opcode in self.allowed_opcodes:
            if not isinstance(opcode, BitOpcode):
                raise TypeError("allowed_opcodes must contain BitOpcode values")
        for asid in self.allowed_asids:
            if isinstance(asid, bool) or not isinstance(asid, int):
                raise TypeError("allowed_asids must contain integers")
            if not 0 <= asid <= 255:
                raise ValueError("allowed ASID must be in [0, 255]")
        for access_class in self.allowed_classes:
            if isinstance(access_class, bool) or not isinstance(access_class, int):
                raise TypeError("allowed_classes must contain integers")
            if not 0 <= access_class <= 7:
                raise ValueError("allowed access class must be in [0, 7]")
        for name in ("x_range", "y_range", "z_range"):
            value = getattr(self, name)
            if len(value) != 2 or any(
                isinstance(item, bool) or not isinstance(item, int) for item in value
            ):
                raise TypeError(f"{name} must be a two-integer tuple")
            if value[0] < 0 or value[1] > 999 or value[0] > value[1]:
                raise ValueError(f"{name} must be an ordered subset of [0, 999]")
        if isinstance(self.max_accessed_bricks, bool) or not isinstance(
            self.max_accessed_bricks, int
        ):
            raise TypeError("max_accessed_bricks must be an integer")
        if self.max_accessed_bricks <= 0:
            raise ValueError("max_accessed_bricks must be positive")

    def authorize(self, opcode: BitOpcode, address: BitAddress) -> None:
        if opcode not in self.allowed_opcodes:
            raise PermissionError(f"opcode {opcode.value} is not allowed")
        if address.asid not in self.allowed_asids:
            raise PermissionError(f"ASID {address.asid} is not allowed")
        if address.access_class not in self.allowed_classes:
            raise PermissionError(f"access class {address.access_class} is not allowed")
        if not self.x_range[0] <= address.x <= self.x_range[1]:
            raise PermissionError("X coordinate is outside the capability")
        if not self.y_range[0] <= address.y <= self.y_range[1]:
            raise PermissionError("Y coordinate is outside the capability")
        if not self.z_range[0] <= address.z <= self.z_range[1]:
            raise PermissionError("Z coordinate is outside the capability")


@dataclass(frozen=True)
class BitInstruction:
    """One bounded bit-range operation."""

    opcode: BitOpcode
    destination: BitAddress | None = None
    source0: BitAddress | None = None
    source1: BitAddress | None = None
    length_bits: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.opcode, BitOpcode):
            raise TypeError("opcode must be a BitOpcode")
        if isinstance(self.length_bits, bool) or not isinstance(self.length_bits, int):
            raise TypeError("length_bits must be an integer")
        if self.length_bits <= 0:
            raise ValueError("length_bits must be positive")


@dataclass(frozen=True)
class JournalEntry:
    sequence: int
    opcode: str
    committed: bool
    destination: int | str | None
    source0: int | str | None
    source1: int | str | None
    length_bits: int
    result: int | str | None
    touched_bricks: Tuple[BrickKey, ...]
    before_state_digest: str
    after_state_digest: str
    previous_hash: str
    error: str | None
    hash: str


@dataclass(frozen=True)
class ExecutionReceipt:
    sequence: int
    committed: bool
    result: int | str | None
    touched_bricks: Tuple[BrickKey, ...]
    state_digest: str
    journal_digest: str


class TransactionRejected(RuntimeError):
    """Raised after a rejected instruction has been recorded in the journal."""

    def __init__(self, message: str, receipt: ExecutionReceipt) -> None:
        super().__init__(message)
        self.receipt = receipt


class Sparse3DBitVM:
    """Sparse, deterministic and transactional bit-addressed virtual machine."""

    CHECKPOINT_VERSION = 1

    def __init__(
        self,
        config: BitVMConfig | None = None,
        capability: Capability | None = None,
    ) -> None:
        self.config = config or BitVMConfig()
        self.capability = capability or Capability(
            x_range=(0, self.config.side - 1),
            y_range=(0, self.config.side - 1),
            z_range=(0, self.config.side - 1),
            max_accessed_bricks=self.config.max_resident_bricks,
        )
        self._bricks: Dict[BrickKey, bytes] = {}
        self._journal: list[JournalEntry] = []
        self._sequence = 0
        self._journal_digest = self._initial_journal_digest()

    @property
    def virtual_brick_count(self) -> int:
        return self.config.virtual_bricks

    @property
    def virtual_byte_count(self) -> int:
        return self.config.virtual_bytes

    @property
    def virtual_bit_count(self) -> int:
        return self.config.virtual_bits

    @property
    def resident_brick_count(self) -> int:
        return len(self._bricks)

    @property
    def resident_payload_bytes(self) -> int:
        return len(self._bricks) * self.config.brick_bytes

    @property
    def journal_digest(self) -> str:
        return self._journal_digest

    @property
    def journal(self) -> Tuple[JournalEntry, ...]:
        return tuple(self._journal)

    def estimate_dense_payload_bytes(self) -> int:
        return self.virtual_byte_count

    def read_bit(self, address: BitAddress) -> int:
        """Read one bit. An absent brick returns zero without allocation."""

        self._validate_address(address)
        if address.access_class not in _READABLE_CLASSES:
            raise PermissionError("read_bit requires a readable address class")
        self.capability.authorize(BitOpcode.BPOPCNT, address)
        return self._read_bit_from(self._bricks, address)

    def execute(self, instruction: BitInstruction) -> ExecutionReceipt:
        """Execute one instruction as an atomic copy-on-write transaction."""

        if not isinstance(instruction, BitInstruction):
            raise TypeError("instruction must be a BitInstruction")
        before_digest = self.state_digest()
        staged: Dict[BrickKey, bytearray] = {}
        accessed: set[BrickKey] = set()
        result: int | str | None = None

        try:
            if instruction.length_bits > self.config.max_instruction_bits:
                raise ValueError("instruction bit length exceeds the configured bound")
            self._validate_instruction_shape(instruction)

            destination = self._address_stream(instruction.destination, instruction.length_bits)
            source0 = self._address_stream(instruction.source0, instruction.length_bits)
            source1 = self._address_stream(instruction.source1, instruction.length_bits)

            if instruction.opcode in (BitOpcode.BSET, BitOpcode.BCLR):
                bit_value = 1 if instruction.opcode is BitOpcode.BSET else 0
                for dst in destination:
                    self._authorize_address(instruction.opcode, dst, accessed)
                    self._write_bit(staged, dst, bit_value)

            elif instruction.opcode in (BitOpcode.BCOPY, BitOpcode.BNOT):
                for dst, src in zip(destination, source0):
                    self._authorize_address(instruction.opcode, dst, accessed)
                    self._authorize_address(instruction.opcode, src, accessed)
                    source_bit = self._read_bit_from(self._bricks, src)
                    value = source_bit if instruction.opcode is BitOpcode.BCOPY else 1 - source_bit
                    self._write_bit(staged, dst, value)

            elif instruction.opcode in (BitOpcode.BAND, BitOpcode.BOR, BitOpcode.BXOR):
                for dst, left, right in zip(destination, source0, source1):
                    self._authorize_address(instruction.opcode, dst, accessed)
                    self._authorize_address(instruction.opcode, left, accessed)
                    self._authorize_address(instruction.opcode, right, accessed)
                    left_bit = self._read_bit_from(self._bricks, left)
                    right_bit = self._read_bit_from(self._bricks, right)
                    if instruction.opcode is BitOpcode.BAND:
                        value = left_bit & right_bit
                    elif instruction.opcode is BitOpcode.BOR:
                        value = left_bit | right_bit
                    else:
                        value = left_bit ^ right_bit
                    self._write_bit(staged, dst, value)

            elif instruction.opcode is BitOpcode.BPOPCNT:
                count = 0
                for src in source0:
                    self._authorize_address(instruction.opcode, src, accessed)
                    count += self._read_bit_from(self._bricks, src)
                result = count

            elif instruction.opcode is BitOpcode.BHASH:
                bits: list[int] = []
                for src in source0:
                    self._authorize_address(instruction.opcode, src, accessed)
                    bits.append(self._read_bit_from(self._bricks, src))
                result = hashlib.sha256(self._pack_bits_lsb_first(bits)).hexdigest()

            else:  # pragma: no cover
                raise ValueError(f"unsupported opcode {instruction.opcode.value}")

            next_bricks = dict(self._bricks)
            for key, payload in staged.items():
                immutable = bytes(payload)
                if self.config.prune_zero_bricks and not any(immutable):
                    next_bricks.pop(key, None)
                else:
                    next_bricks[key] = immutable
            if len(next_bricks) > self.config.max_resident_bricks:
                raise RuntimeError("resident-brick budget exceeded")

            after_digest = self._state_digest_for(next_bricks)
            touched = tuple(sorted(staged))
            entry = self._append_journal(
                instruction=instruction,
                committed=True,
                result=result,
                touched=touched,
                before_digest=before_digest,
                after_digest=after_digest,
                error=None,
            )
            self._bricks = next_bricks
            self._sequence = entry.sequence
            return ExecutionReceipt(
                sequence=entry.sequence,
                committed=True,
                result=result,
                touched_bricks=touched,
                state_digest=after_digest,
                journal_digest=entry.hash,
            )
        except Exception as exc:
            touched = tuple(sorted(staged))
            entry = self._append_journal(
                instruction=instruction,
                committed=False,
                result=None,
                touched=touched,
                before_digest=before_digest,
                after_digest=before_digest,
                error=f"{type(exc).__name__}: {exc}",
            )
            self._sequence = entry.sequence
            receipt = ExecutionReceipt(
                sequence=entry.sequence,
                committed=False,
                result=None,
                touched_bricks=touched,
                state_digest=before_digest,
                journal_digest=entry.hash,
            )
            raise TransactionRejected(str(exc), receipt) from exc

    def state_digest(self) -> str:
        return self._state_digest_for(self._bricks)

    def brick_digest(self, key: BrickKey) -> str:
        self._validate_brick_key(key)
        payload = self._bricks.get(key, bytes(self.config.brick_bytes))
        return hashlib.sha256(payload).hexdigest()

    def checkpoint(self) -> Dict[str, Any]:
        return {
            "version": self.CHECKPOINT_VERSION,
            "config": asdict(self.config),
            "sequence": self._sequence,
            "journal_digest": self._journal_digest,
            "state_digest": self.state_digest(),
            "bricks": [
                {"key": list(key), "payload_b64": base64.b64encode(payload).decode("ascii")}
                for key, payload in sorted(self._bricks.items())
            ],
            "journal": [asdict(entry) for entry in self._journal],
        }

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: Mapping[str, Any],
        capability: Capability | None = None,
    ) -> "Sparse3DBitVM":
        """Restore and verify a checkpoint, rejecting malformed or tampered state."""

        if not isinstance(checkpoint, Mapping):
            raise TypeError("checkpoint must be a mapping")
        if checkpoint.get("version") != cls.CHECKPOINT_VERSION:
            raise ValueError("unsupported checkpoint version")
        config_raw = checkpoint.get("config")
        if not isinstance(config_raw, Mapping):
            raise ValueError("checkpoint config must be a mapping")
        vm = cls(BitVMConfig(**dict(config_raw)), capability)

        bricks_raw = checkpoint.get("bricks")
        if not isinstance(bricks_raw, Sequence) or isinstance(bricks_raw, (str, bytes, bytearray)):
            raise ValueError("checkpoint bricks must be a sequence")
        bricks: Dict[BrickKey, bytes] = {}
        for item in bricks_raw:
            if not isinstance(item, Mapping):
                raise ValueError("checkpoint brick entry must be a mapping")
            raw_key = item.get("key")
            if not isinstance(raw_key, Sequence) or isinstance(raw_key, (str, bytes, bytearray)):
                raise ValueError("checkpoint brick key must be a sequence")
            key = tuple(raw_key)
            vm._validate_brick_key(key)
            if key in bricks:
                raise ValueError("checkpoint contains a duplicate brick key")
            payload_text = item.get("payload_b64")
            if not isinstance(payload_text, str):
                raise ValueError("checkpoint brick payload must be base64 text")
            try:
                payload = base64.b64decode(payload_text, validate=True)
            except Exception as exc:
                raise ValueError("checkpoint brick payload is not valid base64") from exc
            if len(payload) != vm.config.brick_bytes:
                raise ValueError("checkpoint brick payload has the wrong length")
            if vm.config.prune_zero_bricks and not any(payload):
                raise ValueError("checkpoint contains a non-canonical zero brick")
            bricks[key] = payload
        if len(bricks) > vm.config.max_resident_bricks:
            raise ValueError("checkpoint exceeds the resident-brick budget")
        vm._bricks = bricks

        journal_raw = checkpoint.get("journal")
        if not isinstance(journal_raw, Sequence) or isinstance(
            journal_raw, (str, bytes, bytearray)
        ):
            raise ValueError("checkpoint journal must be a sequence")
        previous_hash = vm._initial_journal_digest()
        expected_before = vm._state_digest_for({})
        entries: list[JournalEntry] = []
        for expected_sequence, raw in enumerate(journal_raw, start=1):
            if not isinstance(raw, Mapping):
                raise ValueError("checkpoint journal entry must be a mapping")
            normalized = dict(raw)
            touched_raw = normalized.get("touched_bricks")
            if not isinstance(touched_raw, Sequence) or isinstance(
                touched_raw, (str, bytes, bytearray)
            ):
                raise ValueError("checkpoint touched_bricks must be a sequence")
            touched: list[BrickKey] = []
            for raw_key in touched_raw:
                if not isinstance(raw_key, Sequence) or isinstance(
                    raw_key, (str, bytes, bytearray)
                ):
                    raise ValueError("checkpoint touched brick key must be a sequence")
                key = tuple(raw_key)
                vm._validate_brick_key(key)
                touched.append(key)
            if tuple(touched) != tuple(sorted(set(touched))):
                raise ValueError("checkpoint touched_bricks must be sorted and unique")
            normalized["touched_bricks"] = tuple(touched)
            try:
                entry = JournalEntry(**normalized)
            except TypeError as exc:
                raise ValueError("checkpoint journal entry schema mismatch") from exc
            vm._validate_journal_entry(entry, expected_sequence)
            if entry.previous_hash != previous_hash:
                raise ValueError("checkpoint journal chain mismatch")
            if entry.before_state_digest != expected_before:
                raise ValueError("checkpoint journal state continuity mismatch")
            if not entry.committed and entry.after_state_digest != entry.before_state_digest:
                raise ValueError("rejected journal entry must preserve state")
            if vm._journal_entry_hash(entry) != entry.hash:
                raise ValueError("checkpoint journal digest mismatch")
            previous_hash = entry.hash
            expected_before = entry.after_state_digest
            entries.append(entry)

        vm._journal = entries
        vm._journal_digest = previous_hash
        vm._sequence = len(entries)
        restored_state_digest = vm.state_digest()

        if checkpoint.get("sequence") != vm._sequence:
            raise ValueError("checkpoint sequence mismatch")
        if checkpoint.get("journal_digest") != vm._journal_digest:
            raise ValueError("checkpoint journal digest mismatch")
        if checkpoint.get("state_digest") != restored_state_digest:
            raise ValueError("checkpoint state digest mismatch")
        if expected_before != restored_state_digest:
            raise ValueError("checkpoint journal tip does not match restored state")
        return vm

    def _validate_instruction_shape(self, instruction: BitInstruction) -> None:
        mutating = {
            BitOpcode.BSET,
            BitOpcode.BCLR,
            BitOpcode.BCOPY,
            BitOpcode.BNOT,
            BitOpcode.BAND,
            BitOpcode.BOR,
            BitOpcode.BXOR,
        }
        operands = (instruction.destination, instruction.source0, instruction.source1)
        for operand in operands:
            if operand is not None and not isinstance(operand, BitAddress):
                raise TypeError("instruction addresses must be BitAddress values")

        if instruction.opcode in (BitOpcode.BSET, BitOpcode.BCLR):
            required = (instruction.destination,)
            forbidden = (instruction.source0, instruction.source1)
        elif instruction.opcode in (BitOpcode.BCOPY, BitOpcode.BNOT):
            required = (instruction.destination, instruction.source0)
            forbidden = (instruction.source1,)
        elif instruction.opcode in (BitOpcode.BAND, BitOpcode.BOR, BitOpcode.BXOR):
            required = (instruction.destination, instruction.source0, instruction.source1)
            forbidden = ()
        elif instruction.opcode in (BitOpcode.BPOPCNT, BitOpcode.BHASH):
            required = (instruction.source0,)
            forbidden = (instruction.destination, instruction.source1)
        else:  # pragma: no cover
            raise ValueError("unsupported opcode")
        if any(item is None for item in required):
            raise ValueError("instruction is missing a required operand")
        if any(item is not None for item in forbidden):
            raise ValueError("instruction contains an extraneous operand")

        if instruction.opcode in mutating:
            assert instruction.destination is not None
            if instruction.destination.access_class not in _WRITABLE_CLASSES:
                raise PermissionError("mutating destination must carry a write-capable class")
        for source in (instruction.source0, instruction.source1):
            if source is not None and source.access_class not in _READABLE_CLASSES:
                raise PermissionError("source must carry a readable class")

    def _validate_address(self, address: BitAddress) -> None:
        if not isinstance(address, BitAddress):
            raise TypeError("address must be a BitAddress")
        if max(address.x, address.y, address.z) >= self.config.side:
            raise ValueError("address coordinate is outside the configured lattice")
        if address.byte_offset >= self.config.brick_bytes:
            raise ValueError("byte offset is outside the configured brick")

    def _validate_brick_key(self, key: object) -> None:
        if not isinstance(key, tuple) or len(key) != 4:
            raise TypeError("brick key must be a four-integer tuple")
        asid, x, y, z = key
        for item in key:
            if isinstance(item, bool) or not isinstance(item, int):
                raise TypeError("brick key components must be integers")
        if not 0 <= asid <= 255:
            raise ValueError("brick ASID is outside [0, 255]")
        if min(x, y, z) < 0 or max(x, y, z) >= self.config.side:
            raise ValueError("brick coordinate is outside the configured lattice")

    def _authorize_address(
        self, opcode: BitOpcode, address: BitAddress, accessed: set[BrickKey]
    ) -> None:
        self._validate_address(address)
        self.capability.authorize(opcode, address)
        accessed.add(address.brick_key)
        if len(accessed) > self.capability.max_accessed_bricks:
            raise PermissionError("instruction exceeds the capability brick-access bound")

    def _address_stream(
        self, start: BitAddress | None, length_bits: int
    ) -> Iterator[BitAddress]:
        if start is None:
            return iter(())
        self._validate_address(start)
        absolute = self._absolute_bit(start)
        final = absolute + length_bits
        if final > self.virtual_bit_count:
            raise ValueError("bit range exceeds the configured virtual lattice")

        def iterator() -> Iterator[BitAddress]:
            for offset in range(length_bits):
                yield self._address_from_absolute(absolute + offset, start)

        return iterator()

    def _absolute_bit(self, address: BitAddress) -> int:
        brick_index = address.x + self.config.side * (
            address.y + self.config.side * address.z
        )
        return (
            (brick_index * self.config.brick_bytes + address.byte_offset) * 8
            + address.bit_offset
        )

    def _address_from_absolute(self, absolute: int, template: BitAddress) -> BitAddress:
        byte_index, bit_offset = divmod(absolute, 8)
        brick_index, byte_offset = divmod(byte_index, self.config.brick_bytes)
        x = brick_index % self.config.side
        y = (brick_index // self.config.side) % self.config.side
        z = brick_index // (self.config.side * self.config.side)
        return BitAddress(
            asid=template.asid,
            access_class=template.access_class,
            x=x,
            y=y,
            z=z,
            byte_offset=byte_offset,
            bit_offset=bit_offset,
        )

    def _read_bit_from(self, mapping: Mapping[BrickKey, bytes], address: BitAddress) -> int:
        payload = mapping.get(address.brick_key)
        if payload is None:
            return 0
        return (payload[address.byte_offset] >> address.bit_offset) & 1

    def _write_bit(
        self, staged: Dict[BrickKey, bytearray], address: BitAddress, value: int
    ) -> None:
        payload = staged.get(address.brick_key)
        if payload is None:
            payload = bytearray(
                self._bricks.get(address.brick_key, bytes(self.config.brick_bytes))
            )
            staged[address.brick_key] = payload
        mask = 1 << address.bit_offset
        if value:
            payload[address.byte_offset] |= mask
        else:
            payload[address.byte_offset] &= ~mask & 0xFF

    def _state_digest_for(self, mapping: Mapping[BrickKey, bytes]) -> str:
        digest = hashlib.sha256()
        digest.update(b"jarvisx-jx-3d-1pb-bitvm-state-v1\0")
        digest.update(self._canonical_json(asdict(self.config)))
        for key, payload in sorted(mapping.items()):
            digest.update(bytes((key[0],)))
            for component in key[1:]:
                digest.update(component.to_bytes(2, "big"))
            digest.update(payload)
        return digest.hexdigest()

    def _initial_journal_digest(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"jarvisx-jx-3d-1pb-bitvm-journal-v1\0")
        digest.update(self._canonical_json(asdict(self.config)))
        return digest.hexdigest()

    def _append_journal(
        self,
        *,
        instruction: BitInstruction,
        committed: bool,
        result: int | str | None,
        touched: Tuple[BrickKey, ...],
        before_digest: str,
        after_digest: str,
        error: str | None,
    ) -> JournalEntry:
        payload = {
            "sequence": self._sequence + 1,
            "opcode": instruction.opcode.value,
            "committed": committed,
            "destination": self._serialize_address(instruction.destination),
            "source0": self._serialize_address(instruction.source0),
            "source1": self._serialize_address(instruction.source1),
            "length_bits": instruction.length_bits,
            "result": result,
            "touched_bricks": touched,
            "before_state_digest": before_digest,
            "after_state_digest": after_digest,
            "previous_hash": self._journal_digest,
            "error": error,
        }
        entry_hash = hashlib.sha256(self._canonical_json(payload)).hexdigest()
        entry = JournalEntry(hash=entry_hash, **payload)
        self._journal.append(entry)
        self._journal_digest = entry_hash
        return entry

    @staticmethod
    def _serialize_address(value: object) -> int | str | None:
        if value is None:
            return None
        if isinstance(value, BitAddress):
            return value.pack()
        return f"<invalid:{type(value).__module__}.{type(value).__qualname__}>"

    def _validate_journal_entry(self, entry: JournalEntry, expected_sequence: int) -> None:
        if isinstance(entry.sequence, bool) or entry.sequence != expected_sequence:
            raise ValueError("checkpoint journal sequence mismatch")
        if entry.opcode not in {opcode.value for opcode in BitOpcode}:
            raise ValueError("checkpoint journal opcode is invalid")
        if not isinstance(entry.committed, bool):
            raise ValueError("checkpoint journal committed flag is invalid")
        if isinstance(entry.length_bits, bool) or not isinstance(entry.length_bits, int):
            raise ValueError("checkpoint journal length_bits is invalid")
        if entry.length_bits <= 0 or entry.length_bits > self.config.max_instruction_bits:
            raise ValueError("checkpoint journal length_bits is outside bounds")
        for value in (entry.destination, entry.source0, entry.source1):
            if value is not None and not isinstance(value, (int, str)):
                raise ValueError("checkpoint journal address encoding is invalid")
            if isinstance(value, int) and not 0 <= value < 1 << 64:
                raise ValueError("checkpoint journal packed address is outside bounds")
            if isinstance(value, str) and not value.startswith("<invalid:"):
                raise ValueError("checkpoint journal invalid-address marker is malformed")
        for digest in (
            entry.before_state_digest,
            entry.after_state_digest,
            entry.previous_hash,
            entry.hash,
        ):
            self._validate_digest(digest)
        if entry.committed and entry.error is not None:
            raise ValueError("committed journal entry must not contain an error")
        if not entry.committed and not isinstance(entry.error, str):
            raise ValueError("rejected journal entry must contain an error")
        if entry.committed and any(
            isinstance(value, str)
            for value in (entry.destination, entry.source0, entry.source1)
        ):
            raise ValueError("committed journal entry cannot contain invalid addresses")

    @staticmethod
    def _validate_digest(value: object) -> None:
        if not isinstance(value, str) or len(value) != _HEX_DIGEST_LENGTH:
            raise ValueError("checkpoint digest must be 64 hexadecimal characters")
        try:
            bytes.fromhex(value)
        except ValueError as exc:
            raise ValueError("checkpoint digest must be hexadecimal") from exc

    def _journal_entry_hash(self, entry: JournalEntry) -> str:
        payload = asdict(entry)
        payload.pop("hash")
        return hashlib.sha256(self._canonical_json(payload)).hexdigest()

    @staticmethod
    def _canonical_json(value: object) -> bytes:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")

    @staticmethod
    def _pack_bits_lsb_first(bits: Iterable[int]) -> bytes:
        """Pack the first addressed bit into bit 0 of the first output byte."""

        packed = bytearray()
        value = 0
        width = 0
        for bit in bits:
            value |= (bit & 1) << width
            width += 1
            if width == 8:
                packed.append(value)
                value = 0
                width = 0
        if width:
            packed.append(value)
        return bytes(packed)
