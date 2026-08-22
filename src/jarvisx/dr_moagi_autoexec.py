"""End-to-end sparse 3D auto-encoding/decoding auto-execution engine.

This integration layer composes the existing transactional Dr Moagi field runtime
with a deterministic sparse 3D parser, a bounded block autoencoder, verification,
hash-chained audit records, and conservative policy self-optimization.

The self-optimizer is deliberately bounded. It searches a finite policy
neighbourhood and only promotes a candidate when its measured objective improves
by a configured minimum. It does not rewrite arbitrary code or mutate the host
process.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .dr_moagi_field_runtime import (
    Coordinate,
    DrMoagiFieldConfig,
    DrMoagiFieldRuntime,
    FieldStepMetrics,
    SparseField,
)


@dataclass(frozen=True)
class AutoExecPolicy:
    """Bounded policy that permeates parser, codec and runtime projection."""

    block_size: int = 2
    quantization: float = 0.01
    prune_epsilon: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.block_size, bool) or not isinstance(self.block_size, int):
            raise TypeError("block_size must be an integer")
        if not 1 <= self.block_size <= 16:
            raise ValueError("block_size must be in [1, 16]")
        for name in ("quantization", "prune_epsilon"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if not 1.0e-9 <= self.quantization <= 1.0:
            raise ValueError("quantization must be in [1e-9, 1]")
        if not 0.0 <= self.prune_epsilon <= 1.0:
            raise ValueError("prune_epsilon must be in [0, 1]")


@dataclass(frozen=True)
class LatentCell3D:
    block: Coordinate
    value: float
    population: int
    energy: float


@dataclass(frozen=True)
class SparseLatent3D:
    """Compact sparse 3D latent representation."""

    block_size: int
    support_cells: int
    cells: tuple[LatentCell3D, ...]

    @property
    def latent_cells(self) -> int:
        return len(self.cells)


class SparseParser3D:
    """Validate and sparsify raw coordinate/value input without dense allocation."""

    def __init__(
        self,
        *,
        side: int,
        max_active_cells: int,
        value_min: float,
        value_max: float,
        prune_epsilon: float = 0.0,
    ) -> None:
        self.side = side
        self.max_active_cells = max_active_cells
        self.value_min = value_min
        self.value_max = value_max
        self.prune_epsilon = prune_epsilon

    def set_prune_epsilon(self, value: float) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError("prune_epsilon must be in [0, 1]")
        self.prune_epsilon = float(value)

    def parse(
        self,
        source: Mapping[Coordinate, float] | Iterable[tuple[Coordinate, float]],
    ) -> SparseField:
        items = source.items() if isinstance(source, Mapping) else source
        parsed: SparseField = {}
        for raw_coordinate, raw_value in items:
            coordinate = self._coordinate(raw_coordinate)
            value = self._value(raw_value)
            value = min(self.value_max, max(self.value_min, value))
            if abs(value) <= self.prune_epsilon:
                parsed.pop(coordinate, None)
            else:
                parsed[coordinate] = value
            if len(parsed) > self.max_active_cells:
                raise RuntimeError("active-cell budget exceeded during parsing")
        return parsed

    def _coordinate(self, coordinate: object) -> Coordinate:
        if not isinstance(coordinate, (tuple, list)) or len(coordinate) != 3:
            raise TypeError("coordinate must be a 3-item tuple/list")
        values: list[int] = []
        for axis in coordinate:
            if isinstance(axis, bool) or not isinstance(axis, int):
                raise TypeError("coordinate axes must be integers")
            if not 0 <= axis < self.side:
                raise ValueError("coordinate outside logical 3D lattice")
            values.append(axis)
        return values[0], values[1], values[2]

    @staticmethod
    def _value(value: object) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("field values must be numeric")
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("field values must be finite")
        return result


class SparseBlockCodec3D:
    """Deterministic sparse block-mean autoencoder.

    Coordinates in the same ``block_size`` cube share a quantized latent value.
    Decoding materializes only support explicitly requested by the runtime.
    """

    def __init__(self, policy: AutoExecPolicy | None = None) -> None:
        self.policy = policy or AutoExecPolicy()

    def set_policy(self, policy: AutoExecPolicy) -> None:
        self.policy = policy

    def encode(self, field: Mapping[Coordinate, float]) -> SparseLatent3D:
        block_size = self.policy.block_size
        accum: dict[Coordinate, tuple[float, float, int]] = {}
        for coordinate, raw_value in field.items():
            value = float(raw_value)
            block = (
                coordinate[0] // block_size,
                coordinate[1] // block_size,
                coordinate[2] // block_size,
            )
            total, energy, count = accum.get(block, (0.0, 0.0, 0))
            accum[block] = (total + value, energy + value * value, count + 1)

        cells: list[LatentCell3D] = []
        for block in sorted(accum):
            total, energy, count = accum[block]
            mean = total / count
            q = self.policy.quantization
            quantized = round(mean / q) * q
            if abs(quantized) <= self.policy.prune_epsilon:
                continue
            cells.append(
                LatentCell3D(
                    block=block,
                    value=quantized,
                    population=count,
                    energy=energy,
                )
            )

        return SparseLatent3D(
            block_size=block_size,
            support_cells=len(field),
            cells=tuple(cells),
        )

    def decode(
        self,
        latent: SparseLatent3D,
        support: Sequence[Coordinate],
    ) -> SparseField:
        if latent.block_size != self.policy.block_size:
            raise ValueError("latent block size does not match active codec policy")
        values = {cell.block: cell.value for cell in latent.cells}
        block_size = latent.block_size
        return {
            coordinate: float(
                values.get(
                    (
                        coordinate[0] // block_size,
                        coordinate[1] // block_size,
                        coordinate[2] // block_size,
                    ),
                    0.0,
                )
            )
            for coordinate in support
        }


@dataclass(frozen=True)
class DrMoagiAutoExecConfig:
    """Resource, verification and bounded self-optimization contract."""

    field_config: DrMoagiFieldConfig = field(
        default_factory=lambda: DrMoagiFieldConfig(
            side=64,
            alpha=1.0,
            lambda_residual=0.25,
            eta=0.05,
            dt=0.02,
            max_active_cells=50_000,
            expand_halo=True,
            prune_epsilon=0.0,
        )
    )
    policy: AutoExecPolicy = field(default_factory=AutoExecPolicy)
    cycles: int = 4
    auto_optimize: bool = True
    max_reconstruction_mse: float = 1.0
    min_policy_improvement: float = 1.0e-6
    max_block_size: int = 8
    quantization_multiplier: float = 2.0
    prune_step: float = 0.01
    fidelity_weight: float = 0.72
    compression_weight: float = 0.20
    execution_weight: float = 0.08

    def __post_init__(self) -> None:
        if isinstance(self.cycles, bool) or not isinstance(self.cycles, int) or self.cycles <= 0:
            raise ValueError("cycles must be a positive integer")
        if not 1 <= self.max_block_size <= 16:
            raise ValueError("max_block_size must be in [1, 16]")
        if self.policy.block_size > self.max_block_size:
            raise ValueError("policy.block_size exceeds max_block_size")
        for name in (
            "max_reconstruction_mse",
            "min_policy_improvement",
            "quantization_multiplier",
            "prune_step",
            "fidelity_weight",
            "compression_weight",
            "execution_weight",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if self.max_reconstruction_mse < 0.0:
            raise ValueError("max_reconstruction_mse must be non-negative")
        if self.min_policy_improvement < 0.0:
            raise ValueError("min_policy_improvement must be non-negative")
        if self.quantization_multiplier <= 1.0:
            raise ValueError("quantization_multiplier must be > 1")
        if self.prune_step <= 0.0:
            raise ValueError("prune_step must be positive")
        if min(self.fidelity_weight, self.compression_weight, self.execution_weight) < 0.0:
            raise ValueError("objective weights must be non-negative")
        if self.fidelity_weight + self.compression_weight + self.execution_weight <= 0.0:
            raise ValueError("at least one objective weight must be positive")


@dataclass(frozen=True)
class CycleReport:
    cycle: int
    committed: bool
    active_cells_before: int
    active_cells_after: int
    latent_cells: int
    compression_ratio: float
    reconstruction_mse: float
    objective_before: float
    objective_after: float
    policy_promoted: bool
    policy_before: AutoExecPolicy
    policy_after: AutoExecPolicy
    rejection_reason: str | None
    journal_hash: str = ""

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class HashChainJournal:
    """Small append-only SHA-256 chain for deterministic audit evidence."""

    _GENESIS = "0" * 64

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else None
        self.entries: list[dict[str, object]] = []
        self.head = self._GENESIS
        if self.path is not None and self.path.exists():
            self._load()

    @staticmethod
    def _digest(prev_hash: str, record: Mapping[str, object]) -> str:
        payload = json.dumps(
            {"prev_hash": prev_hash, "record": record},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def append(self, record: Mapping[str, object]) -> str:
        clean_record = dict(record)
        digest = self._digest(self.head, clean_record)
        envelope: dict[str, object] = {
            "prev_hash": self.head,
            "record": clean_record,
            "hash": digest,
        }
        self.entries.append(envelope)
        self.head = digest
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(envelope, sort_keys=True) + "\n")
        return digest

    def verify(self) -> bool:
        previous = self._GENESIS
        for envelope in self.entries:
            if envelope.get("prev_hash") != previous:
                return False
            record = envelope.get("record")
            if not isinstance(record, Mapping):
                return False
            expected = self._digest(previous, record)
            if envelope.get("hash") != expected:
                return False
            previous = expected
        return previous == self.head

    def _load(self) -> None:
        assert self.path is not None
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            envelope = json.loads(line)
            if not isinstance(envelope, dict):
                raise ValueError("journal contains a non-object record")
            self.entries.append(envelope)
        if self.entries:
            final_hash = self.entries[-1].get("hash")
            if not isinstance(final_hash, str):
                raise ValueError("journal contains an invalid hash")
            self.head = final_hash
        if not self.verify():
            raise ValueError("journal hash chain verification failed")


class DrMoagiAutoExecutionEngine:
    """Parse -> encode -> decode -> execute -> verify -> optimize -> permeate."""

    def __init__(
        self,
        config: DrMoagiAutoExecConfig | None = None,
        *,
        journal_path: str | Path | None = None,
    ) -> None:
        self.config = config or DrMoagiAutoExecConfig()
        self.policy = self.config.policy
        fc = self.config.field_config
        self.parser = SparseParser3D(
            side=fc.side,
            max_active_cells=fc.max_active_cells,
            value_min=fc.value_min,
            value_max=fc.value_max,
            prune_epsilon=self.policy.prune_epsilon,
        )
        self.codec = SparseBlockCodec3D(self.policy)
        self.runtime = DrMoagiFieldRuntime(
            self.codec,
            replace(fc, prune_epsilon=self.policy.prune_epsilon),
        )
        self.journal = HashChainJournal(journal_path)
        self.reports: list[CycleReport] = []
        self._loaded = False
        self._permeation_generation = 0

    @property
    def permeation_generation(self) -> int:
        return self._permeation_generation

    def load(
        self,
        source: Mapping[Coordinate, float] | Iterable[tuple[Coordinate, float]],
    ) -> SparseField:
        parsed = self.parser.parse(source)
        self.runtime.load(parsed)
        self.reports.clear()
        self._loaded = True
        return self.runtime.snapshot()

    def encode(self) -> SparseLatent3D:
        self._require_loaded()
        return self.codec.encode(self.runtime.snapshot())

    def decode(
        self,
        latent: SparseLatent3D | None = None,
        support: Sequence[Coordinate] | None = None,
    ) -> SparseField:
        self._require_loaded()
        state = self.runtime.snapshot()
        latent = latent or self.codec.encode(state)
        selected_support = tuple(sorted(state)) if support is None else tuple(support)
        return self.codec.decode(latent, selected_support)

    def run(self, cycles: int | None = None) -> tuple[CycleReport, ...]:
        self._require_loaded()
        count = self.config.cycles if cycles is None else cycles
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("cycles must be a positive integer")
        for _ in range(count):
            self.step()
        return tuple(self.reports)

    def step(self) -> CycleReport:
        self._require_loaded()
        state_before = self.runtime.snapshot()
        policy_before = self.policy
        latent = self.codec.encode(state_before)
        compression_ratio = latent.latent_cells / max(1, len(state_before))

        objective_before = self._score_policy(state_before, policy_before)
        metrics = self.runtime.step(validator=self._runtime_validator)
        state_after = self.runtime.snapshot()

        objective_after = self._score_policy(state_after, self.policy)
        promoted = False
        if metrics.committed and self.config.auto_optimize and state_after:
            candidate, candidate_score = self._best_policy(state_after)
            if (
                candidate != self.policy
                and candidate_score > objective_after + self.config.min_policy_improvement
            ):
                self._permeate_policy(candidate)
                objective_after = candidate_score
                promoted = True

        provisional = CycleReport(
            cycle=metrics.cycle,
            committed=metrics.committed,
            active_cells_before=metrics.active_cells_before,
            active_cells_after=metrics.active_cells_after,
            latent_cells=latent.latent_cells,
            compression_ratio=compression_ratio,
            reconstruction_mse=metrics.reconstruction_mse,
            objective_before=objective_before,
            objective_after=objective_after,
            policy_promoted=promoted,
            policy_before=policy_before,
            policy_after=self.policy,
            rejection_reason=metrics.rejection_reason,
        )
        payload = provisional.as_dict()
        payload.pop("journal_hash", None)
        digest = self.journal.append(payload)
        report = replace(provisional, journal_hash=digest)
        self.reports.append(report)
        return report

    def status(self) -> dict[str, object]:
        self._require_loaded()
        latent = self.codec.encode(self.runtime.snapshot())
        active = self.runtime.active_cell_count
        return {
            "cycle": self.runtime.cycle,
            "virtual_cells": self.runtime.virtual_cell_count,
            "active_cells": active,
            "latent_cells": latent.latent_cells,
            "compression_ratio": latent.latent_cells / max(1, active),
            "policy": asdict(self.policy),
            "permeation_generation": self._permeation_generation,
            "journal_head": self.journal.head,
            "journal_valid": self.journal.verify(),
        }

    def _runtime_validator(
        self,
        candidate: Mapping[Coordinate, float],
        metrics: FieldStepMetrics,
    ) -> bool:
        if metrics.reconstruction_mse > self.config.max_reconstruction_mse:
            return False
        if not all(
            math.isfinite(value)
            for value in (
                metrics.reconstruction_mse,
                metrics.anchor_mse,
                metrics.max_abs_residual,
                metrics.max_abs_rhs,
            )
        ):
            return False
        return len(candidate) <= self.config.field_config.max_active_cells

    def _best_policy(self, field: Mapping[Coordinate, float]) -> tuple[AutoExecPolicy, float]:
        current = self.policy
        best = current
        best_score = self._score_policy(field, current)
        q_values = {
            max(1.0e-9, current.quantization / self.config.quantization_multiplier),
            current.quantization,
            min(1.0, current.quantization * self.config.quantization_multiplier),
        }
        p_values = {
            max(0.0, current.prune_epsilon - self.config.prune_step),
            current.prune_epsilon,
            min(1.0, current.prune_epsilon + self.config.prune_step),
        }
        b_values = {
            max(1, current.block_size - 1),
            current.block_size,
            min(self.config.max_block_size, current.block_size + 1),
        }
        for block_size in sorted(b_values):
            for quantization in sorted(q_values):
                for prune_epsilon in sorted(p_values):
                    candidate = AutoExecPolicy(
                        block_size=block_size,
                        quantization=quantization,
                        prune_epsilon=prune_epsilon,
                    )
                    score = self._score_policy(field, candidate)
                    if score > best_score + self.config.min_policy_improvement:
                        best = candidate
                        best_score = score
        return best, best_score

    def _score_policy(
        self,
        field: Mapping[Coordinate, float],
        policy: AutoExecPolicy,
    ) -> float:
        if not field:
            return (
                self.config.fidelity_weight
                + self.config.compression_weight
                + self.config.execution_weight
            )

        filtered = {
            coordinate: float(value)
            for coordinate, value in field.items()
            if abs(float(value)) > policy.prune_epsilon
        }
        codec = SparseBlockCodec3D(policy)
        latent = codec.encode(filtered)
        support = tuple(sorted(field))
        reconstruction = codec.decode(latent, support)
        mse = sum(
            (float(field[coordinate]) - reconstruction.get(coordinate, 0.0)) ** 2
            for coordinate in support
        ) / len(support)

        fidelity = 1.0 / (1.0 + mse)
        compression_gain = 1.0 - latent.latent_cells / max(1, len(field))
        execution_saving = 1.0 - len(filtered) / max(1, len(field))
        return (
            self.config.fidelity_weight * fidelity
            + self.config.compression_weight * compression_gain
            + self.config.execution_weight * execution_saving
        )

    def _permeate_policy(self, policy: AutoExecPolicy) -> None:
        """Apply one verified policy coherently across parser, codec and runtime."""

        self.policy = policy
        self.parser.set_prune_epsilon(policy.prune_epsilon)
        self.codec.set_policy(policy)
        self.runtime.config = replace(
            self.runtime.config,
            prune_epsilon=policy.prune_epsilon,
        )
        self._permeation_generation += 1

    def _require_loaded(self) -> None:
        if not self._loaded:
            raise RuntimeError("load a sparse 3D field before execution")
