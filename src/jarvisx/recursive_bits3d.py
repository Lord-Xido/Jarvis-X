"""Bounded 3D bitstream adapter for the Jarvis-X volumetric bytecode runtime.

The adapter makes the byte/bit -> (x, y, z, channel) mapping explicit and adds
an unevaluated right-associated power-tower iteration target. Astronomical
iteration metadata is never materialized and is never reported as physical
throughput.

The executor delegates each bounded physical cycle to VolumetricBytecodeVM.
That preserves the existing sparse-state, exact codec, residual/Hamming,
CTR-style verification and permeation semantics.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from typing import Sequence

from .volumetric_bytecode import VolumetricBytecodeError, VolumetricBytecodeVM, VolumetricConfig


@dataclass(frozen=True, slots=True)
class SymbolicPowerTower:
    """Unevaluated right-associated tower a^(a^(...)).

    The default is the requested septillion tower:

        a = 10^24
        T = a^(a^a)

    represented with height=3. Only the small structural metadata is stored.
    The numeric value of T is never constructed.
    """

    base: int = 10**24
    height: int = 3
    symbol: str = "a"

    def __post_init__(self) -> None:
        if self.base < 2:
            raise ValueError("tower base must be >= 2")
        if self.height < 1:
            raise ValueError("tower height must be >= 1")
        if not self.symbol:
            raise ValueError("tower symbol must be non-empty")

    @property
    def definition(self) -> str:
        return f"{self.symbol}={self.base}"

    @property
    def description(self) -> str:
        expr = self.symbol
        for _ in range(self.height - 1):
            expr = f"{self.symbol}^({expr})"
        return expr

    @property
    def decimal_power_description(self) -> str | None:
        """Return a symbolic base-10 form when base is exactly a power of ten."""

        power = 0
        value = self.base
        while value > 1 and value % 10 == 0:
            value //= 10
            power += 1
        if value != 1:
            return None

        exponent = str(power)
        for _ in range(self.height - 1):
            exponent = f"{power}*10^({exponent})"
        return f"10^({exponent})"


@dataclass(frozen=True, slots=True)
class Bits3DConfig:
    """Logical 3D byte lattice plus bounded physical execution policy."""

    side: int = 64
    channels: int = 4
    lane_bits: int = 8
    octree_depth: int = 3
    max_active_cells: int = 1_048_576
    max_iterations: int = 64
    epsilon_active_change: float = 0.0
    symbolic_target: SymbolicPowerTower = SymbolicPowerTower()

    def __post_init__(self) -> None:
        if self.side < 2:
            raise ValueError("side must be >= 2")
        if self.channels < 1:
            raise ValueError("channels must be positive")
        if self.lane_bits != 8:
            raise ValueError("reference adapter currently supports 8-bit lanes")
        if self.max_active_cells <= 0:
            raise ValueError("max_active_cells must be positive")
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if not 0.0 <= self.epsilon_active_change <= 1.0:
            raise ValueError("epsilon_active_change must be within [0, 1]")

    @property
    def scalar_cells(self) -> int:
        return self.side**3 * self.channels

    @property
    def raw_bits(self) -> int:
        return self.scalar_cells * self.lane_bits

    @property
    def raw_bytes(self) -> int:
        return self.scalar_cells


@dataclass(frozen=True, slots=True)
class CellAddress3D:
    x: int
    y: int
    z: int
    channel: int


@dataclass(frozen=True, slots=True)
class BitAddress3D:
    x: int
    y: int
    z: int
    channel: int
    bit: int


def scalar_index_to_address(index: int, config: Bits3DConfig) -> CellAddress3D:
    """Map a linear byte-lane index into (x, y, z, channel)."""

    if not 0 <= index < config.scalar_cells:
        raise IndexError("scalar index outside logical 3D state")
    channel = index % config.channels
    q = index // config.channels
    x = q % config.side
    y = (q // config.side) % config.side
    z = q // (config.side * config.side)
    return CellAddress3D(x=x, y=y, z=z, channel=channel)


def address_to_scalar_index(address: CellAddress3D, config: Bits3DConfig) -> int:
    """Map (x, y, z, channel) back into the linear byte-lane index."""

    if not 0 <= address.x < config.side:
        raise IndexError("x outside logical 3D state")
    if not 0 <= address.y < config.side:
        raise IndexError("y outside logical 3D state")
    if not 0 <= address.z < config.side:
        raise IndexError("z outside logical 3D state")
    if not 0 <= address.channel < config.channels:
        raise IndexError("channel outside logical 3D state")
    return (
        ((address.z * config.side + address.y) * config.side + address.x)
        * config.channels
        + address.channel
    )


def bit_index_to_address(index: int, config: Bits3DConfig) -> BitAddress3D:
    """Map a linear bit index into (x, y, z, channel, bit)."""

    if not 0 <= index < config.raw_bits:
        raise IndexError("bit index outside logical 3D state")
    scalar_index, bit = divmod(index, config.lane_bits)
    cell = scalar_index_to_address(scalar_index, config)
    return BitAddress3D(
        x=cell.x,
        y=cell.y,
        z=cell.z,
        channel=cell.channel,
        bit=bit,
    )


def address_to_bit_index(address: BitAddress3D, config: Bits3DConfig) -> int:
    """Inverse of bit_index_to_address."""

    if not 0 <= address.bit < config.lane_bits:
        raise IndexError("bit lane outside byte")
    scalar = address_to_scalar_index(
        CellAddress3D(address.x, address.y, address.z, address.channel),
        config,
    )
    return scalar * config.lane_bits + address.bit


REFERENCE_BITS3D_CYCLE: tuple[str, ...] = (
    "MAP_3D_ACTIVE_STATE",
    "SPARSE_MASK",
    "TRAVERSE_OCT",
    "SIMD_ENCODE",
    "FOLD_XYZ_MIRROR",
    "INJECT_MIRROR_UNION",
    "SIMD_DECODE",
    "RESIDUAL_XOR",
    "HAMMING_CHECK",
    "CTR_VERIFY",
    "COMMIT_OR_ROLLBACK",
    "PERMEATE_RECEIPT",
    "RECUR",
)


@dataclass(frozen=True, slots=True)
class Bits3DReceipt:
    iteration: int
    active_cells_before: int
    active_cells_after: int
    changed_bits: int
    active_change_fraction: float
    logical_change_fraction: float
    codec_roundtrip_error_bits: int
    committed: bool
    converged: bool
    state_hash: str
    symbolic_iteration_target: str
    executed_ops: tuple[str, ...] = REFERENCE_BITS3D_CYCLE


@dataclass(frozen=True, slots=True)
class Bits3DRunResult:
    receipts: tuple[Bits3DReceipt, ...]
    converged: bool
    budget_exhausted: bool
    physical_iterations: int
    changed_bits_total: int
    symbolic_iteration_target: str
    symbolic_decimal_target: str | None
    final_state_hash: str


class RecursiveBits3DVM:
    """Byte/bit adapter over the bounded sparse volumetric VM.

    Channels are represented as deterministic modality names ch0, ch1, ... in
    the underlying sparse executor. The adapter does not create a second state
    authority: commits and convergence decisions remain those of
    VolumetricBytecodeVM.
    """

    def __init__(self, config: Bits3DConfig | None = None) -> None:
        self.config = config or Bits3DConfig()
        self._vm = VolumetricBytecodeVM(
            VolumetricConfig(
                axis_extent=self.config.side,
                lane_bits=self.config.lane_bits,
                octree_depth=self.config.octree_depth,
                max_active_voxels=self.config.max_active_cells,
                max_iterations=self.config.max_iterations,
                epsilon_active_change=self.config.epsilon_active_change,
            )
        )
        self._loaded_span = 0

    @property
    def active_cells(self) -> int:
        return len(self._vm.state)

    @property
    def physical_iteration(self) -> int:
        return self._vm.iteration

    def load_bytes(self, payload: bytes | bytearray | memoryview) -> None:
        """Load a finite byte prefix into the logical 3D state.

        Zero bytes remain implicit and therefore consume no sparse resident cell.
        """

        data = bytes(payload)
        if len(data) > self.config.scalar_cells:
            raise VolumetricBytecodeError("payload exceeds logical 3D byte capacity")

        modalities: dict[str, dict[tuple[int, int, int], int]] = {
            f"ch{channel}": {} for channel in range(self.config.channels)
        }
        for index, value in enumerate(data):
            if value == 0:
                continue
            address = scalar_index_to_address(index, self.config)
            modalities[f"ch{address.channel}"][(address.x, address.y, address.z)] = value

        self._vm.load_modalities(modalities)
        self._loaded_span = len(data)

    def sparse_bytes(self) -> tuple[tuple[int, int], ...]:
        """Return sorted (linear_byte_index, value) pairs for active cells."""

        output: list[tuple[int, int]] = []
        for key, value in self._vm.state.items():
            modality, x, y, z = key
            if not modality.startswith("ch"):
                raise RuntimeError(f"unexpected channel modality {modality!r}")
            channel = int(modality[2:])
            index = address_to_scalar_index(
                CellAddress3D(x=x, y=y, z=z, channel=channel),
                self.config,
            )
            output.append((index, value))
        return tuple(sorted(output))

    def read_prefix(self, length: int | None = None) -> bytes:
        """Materialize a bounded dense prefix of the current logical state."""

        if length is None:
            length = self._loaded_span
        if not 0 <= length <= self.config.scalar_cells:
            raise ValueError("prefix length outside logical 3D byte capacity")
        data = bytearray(length)
        for index, value in self.sparse_bytes():
            if index < length:
                data[index] = value
        return bytes(data)

    def step(self) -> Bits3DReceipt:
        receipt = self._vm.step()
        return Bits3DReceipt(
            iteration=receipt.iteration,
            active_cells_before=receipt.active_voxels_before,
            active_cells_after=receipt.active_voxels_after,
            changed_bits=receipt.changed_bits,
            active_change_fraction=receipt.active_change_fraction,
            logical_change_fraction=receipt.logical_change_fraction,
            codec_roundtrip_error_bits=receipt.codec_roundtrip_error_bits,
            committed=receipt.committed,
            converged=receipt.converged,
            state_hash=receipt.state_hash,
            symbolic_iteration_target=self.config.symbolic_target.description,
        )

    def run(self) -> Bits3DRunResult:
        receipts: list[Bits3DReceipt] = []
        converged = False
        for _ in range(self.config.max_iterations):
            receipt = self.step()
            receipts.append(receipt)
            if receipt.converged:
                converged = True
                break

        return Bits3DRunResult(
            receipts=tuple(receipts),
            converged=converged,
            budget_exhausted=not converged,
            physical_iterations=len(receipts),
            changed_bits_total=sum(item.changed_bits for item in receipts),
            symbolic_iteration_target=self.config.symbolic_target.description,
            symbolic_decimal_target=self.config.symbolic_target.decimal_power_description,
            final_state_hash=(
                receipts[-1].state_hash
                if receipts
                else self._vm.permeate_snapshot().state_hash
            ),
        )

    def snapshot(self) -> dict[str, object]:
        """Return a serializable, non-authoritative permeation receipt."""

        snapshot = self._vm.permeate_snapshot()
        return {
            "iteration": snapshot.iteration,
            "active_cells": snapshot.active_voxels,
            "state_hash": snapshot.state_hash,
            "logical_side": self.config.side,
            "channels": self.config.channels,
            "lane_bits": self.config.lane_bits,
            "logical_scalar_cells": self.config.scalar_cells,
            "logical_raw_bits": self.config.raw_bits,
            "symbolic_target_definition": self.config.symbolic_target.definition,
            "symbolic_target": self.config.symbolic_target.description,
            "symbolic_decimal_target": self.config.symbolic_target.decimal_power_description,
            "physical_iteration_budget": self.config.max_iterations,
        }


def _demo() -> dict[str, object]:
    vm = RecursiveBits3DVM(Bits3DConfig(side=8, channels=4, max_iterations=4))
    vm.load_bytes(bytes([0x03, 0x00, 0x55, 0x80]))
    result = vm.run()
    return {
        "config": {
            "side": vm.config.side,
            "channels": vm.config.channels,
            "raw_bits": vm.config.raw_bits,
        },
        "result": asdict(result),
        "snapshot": vm.snapshot(),
        "active_sparse_bytes": vm.sparse_bytes(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="bounded recursive 3D bitstream iteration reference"
    )
    parser.add_argument("--demo", action="store_true", help="run the deterministic bounded demo")
    args = parser.parse_args(argv)
    if not args.demo:
        parser.error("use --demo")
    print(json.dumps(_demo(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
