"""Operational auto-encoding/decoding control loop for the Dr Moagi field runtime.

The loop is deliberately bounded: it repeatedly executes the existing
transactional encode -> decode -> residual -> projected-field step, records a
hash-chained Omega journal receipt, and stops only on an explicit convergence,
fixed-point, rejection-budget, or cycle-budget condition.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from typing import Any

from .dr_moagi_field_runtime import (
    Coordinate,
    DrMoagiFieldRuntime,
    FieldStepMetrics,
    SparseField,
    Validator,
)
from .ledger import OmegaLedger

OPCODE_AUTO_CODEC_LOAD = 0xA0
OPCODE_AUTO_CODEC_CYCLE = 0xA1
OPCODE_AUTO_CODEC_STOP = 0xA2


@dataclass(frozen=True)
class UniformQuantizedLatent:
    """Sparse quantized latent representation used by the reference codec."""

    step: float
    codes: dict[Coordinate, int]


class UniformQuantizedFieldCodec:
    """Deterministic sparse scalar codec with explicit quantization error.

    This is a reference codec, not a learned neural compressor. It exists so
    the operational loop has a real, bounded encode/decode transform whose
    reconstruction error can be measured and driven toward a fixed point.
    """

    def __init__(self, step: float = 0.05, *, prune_zero_codes: bool = True) -> None:
        if isinstance(step, bool) or not isinstance(step, (int, float)):
            raise TypeError("step must be numeric")
        step = float(step)
        if not math.isfinite(step) or step <= 0.0:
            raise ValueError("step must be finite and positive")
        self.step = step
        self.prune_zero_codes = bool(prune_zero_codes)

    def encode(self, field: Mapping[Coordinate, float]) -> UniformQuantizedLatent:
        codes: dict[Coordinate, int] = {}
        for coordinate, raw_value in field.items():
            if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                raise TypeError("field values must be numeric")
            value = float(raw_value)
            if not math.isfinite(value):
                raise ValueError("field values must be finite")
            code = int(round(value / self.step))
            if code != 0 or not self.prune_zero_codes:
                codes[coordinate] = code
        return UniformQuantizedLatent(step=self.step, codes=codes)

    def decode(
        self,
        latent: UniformQuantizedLatent,
        support: Sequence[Coordinate],
    ) -> Mapping[Coordinate, float]:
        if not isinstance(latent, UniformQuantizedLatent):
            raise TypeError("latent must be a UniformQuantizedLatent")
        if latent.step != self.step:
            raise ValueError("latent quantization step does not match codec")
        return {
            coordinate: float(latent.codes.get(coordinate, 0)) * self.step
            for coordinate in support
        }


@dataclass(frozen=True)
class AutoCodecLoopConfig:
    """Termination and verification contract for one auto-codec run."""

    max_cycles: int = 64
    min_cycles: int = 1
    reconstruction_mse_target: float = 1e-8
    max_consecutive_rejections: int = 3
    stop_on_fixed_point: bool = True

    def __post_init__(self) -> None:
        for name in ("max_cycles", "min_cycles", "max_consecutive_rejections"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.min_cycles > self.max_cycles:
            raise ValueError("min_cycles cannot exceed max_cycles")
        target = self.reconstruction_mse_target
        if isinstance(target, bool) or not isinstance(target, (int, float)):
            raise TypeError("reconstruction_mse_target must be numeric")
        if not math.isfinite(float(target)) or float(target) < 0.0:
            raise ValueError("reconstruction_mse_target must be finite and non-negative")


@dataclass(frozen=True)
class AutoCodecCycleReceipt:
    cycle: int
    committed: bool
    reconstruction_mse: float
    anchor_mse: float
    active_cells: int
    state_digest: str
    rejection_reason: str | None


@dataclass(frozen=True)
class AutoCodecRunSummary:
    stop_reason: str
    cycles_executed: int
    committed_cycles: int
    rejected_cycles: int
    converged: bool
    final_cycle: int
    final_state_digest: str
    final_reconstruction_mse: float | None
    journal_verified: bool
    journal_entries: int
    journal_head_hash: str | None
    final_state: SparseField

    def to_dict(self) -> dict[str, Any]:
        return {
            "stop_reason": self.stop_reason,
            "cycles_executed": self.cycles_executed,
            "committed_cycles": self.committed_cycles,
            "rejected_cycles": self.rejected_cycles,
            "converged": self.converged,
            "final_cycle": self.final_cycle,
            "final_state_digest": self.final_state_digest,
            "final_reconstruction_mse": self.final_reconstruction_mse,
            "journal_verified": self.journal_verified,
            "journal_entries": self.journal_entries,
            "journal_head_hash": self.journal_head_hash,
            "final_state": field_to_cells(self.final_state),
        }


class AutoCodecLoop:
    """Bounded closed-loop controller around :class:`DrMoagiFieldRuntime`."""

    def __init__(
        self,
        runtime: DrMoagiFieldRuntime,
        config: AutoCodecLoopConfig | None = None,
        *,
        ledger: OmegaLedger | None = None,
    ) -> None:
        self.runtime = runtime
        self.config = config or AutoCodecLoopConfig()
        self.ledger = ledger or OmegaLedger()
        self._loaded = False

    def load(self, field: Mapping[Coordinate, float]) -> None:
        self.runtime.load(field)
        self._loaded = True
        snapshot = self.runtime.snapshot()
        self.ledger.log(
            {
                "event": "auto_codec_load",
                "cycle": self.runtime.cycle,
                "active_cells": len(snapshot),
                "state_digest": digest_field(snapshot),
            },
            OPCODE_AUTO_CODEC_LOAD,
        )

    def step(self, validator: Validator | None = None) -> AutoCodecCycleReceipt:
        if not self._loaded:
            raise RuntimeError("load a field before stepping the auto-codec loop")
        metrics = self.runtime.step(validator=validator)
        snapshot = self.runtime.snapshot()
        receipt = self._receipt(metrics, snapshot)
        self.ledger.log(
            {
                "event": "auto_codec_cycle",
                **asdict(receipt),
            },
            OPCODE_AUTO_CODEC_CYCLE,
        )
        return receipt

    def run(self, validator: Validator | None = None) -> AutoCodecRunSummary:
        if not self._loaded:
            raise RuntimeError("load a field before running the auto-codec loop")

        start_cycle = self.runtime.cycle
        committed = 0
        rejected = 0
        consecutive_rejections = 0
        last_receipt: AutoCodecCycleReceipt | None = None
        previous_committed_digest: str | None = None
        stop_reason = "cycle_limit"
        converged = False

        while self.runtime.cycle - start_cycle < self.config.max_cycles:
            receipt = self.step(validator=validator)
            last_receipt = receipt
            executed = self.runtime.cycle - start_cycle

            if receipt.committed:
                committed += 1
                consecutive_rejections = 0
                if (
                    executed >= self.config.min_cycles
                    and receipt.reconstruction_mse <= self.config.reconstruction_mse_target
                ):
                    stop_reason = "reconstruction_target"
                    converged = True
                    break
                if (
                    self.config.stop_on_fixed_point
                    and executed >= self.config.min_cycles
                    and previous_committed_digest == receipt.state_digest
                ):
                    stop_reason = "fixed_point"
                    converged = True
                    break
                previous_committed_digest = receipt.state_digest
            else:
                rejected += 1
                consecutive_rejections += 1
                if consecutive_rejections >= self.config.max_consecutive_rejections:
                    stop_reason = "rejection_limit"
                    break

        final_state = self.runtime.snapshot()
        summary = AutoCodecRunSummary(
            stop_reason=stop_reason,
            cycles_executed=self.runtime.cycle - start_cycle,
            committed_cycles=committed,
            rejected_cycles=rejected,
            converged=converged,
            final_cycle=self.runtime.cycle,
            final_state_digest=digest_field(final_state),
            final_reconstruction_mse=(
                None if last_receipt is None else last_receipt.reconstruction_mse
            ),
            journal_verified=self.ledger.verify(),
            journal_entries=len(self.ledger.chain),
            journal_head_hash=(self.ledger.chain[-1]["hash"] if self.ledger.chain else None),
            final_state=final_state,
        )
        self.ledger.log(
            {
                "event": "auto_codec_stop",
                "stop_reason": summary.stop_reason,
                "cycles_executed": summary.cycles_executed,
                "committed_cycles": summary.committed_cycles,
                "rejected_cycles": summary.rejected_cycles,
                "converged": summary.converged,
                "final_cycle": summary.final_cycle,
                "final_state_digest": summary.final_state_digest,
            },
            OPCODE_AUTO_CODEC_STOP,
        )
        return replace(
            summary,
            journal_verified=self.ledger.verify(),
            journal_entries=len(self.ledger.chain),
            journal_head_hash=(self.ledger.chain[-1]["hash"] if self.ledger.chain else None),
        )

    @staticmethod
    def _receipt(
        metrics: FieldStepMetrics, snapshot: Mapping[Coordinate, float]
    ) -> AutoCodecCycleReceipt:
        return AutoCodecCycleReceipt(
            cycle=metrics.cycle,
            committed=metrics.committed,
            reconstruction_mse=metrics.reconstruction_mse,
            anchor_mse=metrics.anchor_mse,
            active_cells=len(snapshot),
            state_digest=digest_field(snapshot),
            rejection_reason=metrics.rejection_reason,
        )


def field_to_cells(field: Mapping[Coordinate, float]) -> list[dict[str, int | float]]:
    """Return a deterministic JSON-native representation of a sparse field."""

    return [
        {"x": x, "y": y, "z": z, "value": float(value)}
        for (x, y, z), value in sorted(field.items())
    ]


def digest_field(field: Mapping[Coordinate, float]) -> str:
    payload = json.dumps(
        field_to_cells(field),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
