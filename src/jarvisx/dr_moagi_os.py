"""Full-stack orchestration kernel for the bounded Dr Moagi 3D runtime.

This module is an operating-system *control plane* for Jarvis-X rather than a
replacement for a host kernel.  It owns the authoritative sparse 3D state,
coordinates bit-plane packing, inward folding, auto-encoding/decoding,
fixed-point stabilization, transaction verification, checkpoint persistence,
metrics, and a bounded auto-execution scheduler.

No arbitrary host commands or self-rewriting code are executed.  The scheduler
only advances the internal Dr Moagi state machine.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from time import monotonic
from typing import Mapping

from .dm_vomegaxi_fixed_point import (
    DMvOmegaXiFixedPointConfig,
    DMvOmegaXiFixedPointEngine,
    FixedPointReport,
)
from .dr_moagi_autoexec import (
    AutoExecPolicy,
    DrMoagiAutoExecConfig,
    DrMoagiAutoExecutionEngine,
    HashChainJournal,
)
from .dr_moagi_bitplane import BitPlaneMetrics, SparseBitPlane3D, fold_and_attenuate
from .dr_moagi_field_runtime import Coordinate, DrMoagiFieldConfig, SparseField


class OSLifecycle(str, Enum):
    OFFLINE = "offline"
    READY = "ready"
    RUNNING = "running"
    HALTED = "halted"


@dataclass(frozen=True)
class DrMoagiOSConfig:
    side: int = 64
    max_active_cells: int = 50_000
    activation_threshold: float = 0.5
    contraction: float = 0.08
    attenuation: float = 0.10
    prune_epsilon: float = 0.0
    block_size: int = 2
    quantization: float = 0.01
    auto_optimize: bool = True
    max_reconstruction_mse: float = 1.0
    fixed_point_passes: int = 1
    fixed_point_tolerance: float = 1.0e-6
    autorun_interval_seconds: float = 0.5
    snapshot_limit: int = 2_048
    state_dir: Path | None = field(default_factory=lambda: Path("state/dr-moagi-os"))

    def __post_init__(self) -> None:
        if isinstance(self.side, bool) or not isinstance(self.side, int) or self.side <= 0:
            raise ValueError("side must be a positive integer")
        if (
            isinstance(self.max_active_cells, bool)
            or not isinstance(self.max_active_cells, int)
            or self.max_active_cells <= 0
        ):
            raise ValueError("max_active_cells must be a positive integer")
        if (
            isinstance(self.fixed_point_passes, bool)
            or not isinstance(self.fixed_point_passes, int)
            or self.fixed_point_passes < 0
        ):
            raise ValueError("fixed_point_passes must be a non-negative integer")
        if (
            isinstance(self.snapshot_limit, bool)
            or not isinstance(self.snapshot_limit, int)
            or self.snapshot_limit <= 0
        ):
            raise ValueError("snapshot_limit must be a positive integer")
        for name in (
            "activation_threshold",
            "contraction",
            "attenuation",
            "prune_epsilon",
            "quantization",
            "max_reconstruction_mse",
            "fixed_point_tolerance",
            "autorun_interval_seconds",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if self.activation_threshold < 0.0:
            raise ValueError("activation_threshold must be non-negative")
        if not 0.0 <= self.contraction < 1.0:
            raise ValueError("contraction must be in [0, 1)")
        if self.attenuation < 0.0:
            raise ValueError("attenuation must be non-negative")
        if not 0.0 <= self.prune_epsilon <= 1.0:
            raise ValueError("prune_epsilon must be in [0, 1]")
        if not 1 <= self.block_size <= 16:
            raise ValueError("block_size must be in [1, 16]")
        if not 1.0e-9 <= self.quantization <= 1.0:
            raise ValueError("quantization must be in [1e-9, 1]")
        if self.max_reconstruction_mse < 0.0:
            raise ValueError("max_reconstruction_mse must be non-negative")
        if self.fixed_point_tolerance < 0.0:
            raise ValueError("fixed_point_tolerance must be non-negative")
        if self.autorun_interval_seconds <= 0.0:
            raise ValueError("autorun_interval_seconds must be positive")


@dataclass(frozen=True)
class OSCycleReport:
    cycle: int
    committed: bool
    active_cells_before: int
    active_cells_after: int
    latent_cells: int
    packed_words: int
    logical_words: int
    bit_density: float
    phase_velocity: float
    spatial_entropy: float
    kinetic_energy: float
    reconstruction_mse: float
    fixed_point_residual: float | None
    fixed_point_converged: bool
    policy_promoted: bool
    policy: AutoExecPolicy
    rejection_reason: str | None
    state_hash: str
    bitplane_hash: str
    journal_hash: str = ""

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class DrMoagiOSKernel:
    """Authoritative sparse-state kernel and bounded auto-execution scheduler."""

    CHECKPOINT_VERSION = 1

    def __init__(self, config: DrMoagiOSConfig | None = None) -> None:
        self.config = config or DrMoagiOSConfig()
        self.policy = AutoExecPolicy(
            block_size=self.config.block_size,
            quantization=self.config.quantization,
            prune_epsilon=self.config.prune_epsilon,
        )
        self._lock = threading.RLock()
        self._state: SparseField = {}
        self._bitplane = SparseBitPlane3D(side=self.config.side, words=())
        self._cycle = 0
        self._loaded = False
        self._lifecycle = OSLifecycle.OFFLINE
        self._halt_reason: str | None = None
        self._last_report: OSCycleReport | None = None
        self._boot_time = monotonic()
        self._autorun_stop = threading.Event()
        self._autorun_thread: threading.Thread | None = None
        journal_path = None
        if self.config.state_dir is not None:
            journal_path = self.config.state_dir / "os-journal.jsonl"
        self.journal = HashChainJournal(journal_path)

    @property
    def lifecycle(self) -> OSLifecycle:
        with self._lock:
            return self._lifecycle

    @property
    def loaded(self) -> bool:
        with self._lock:
            return self._loaded

    @property
    def cycle(self) -> int:
        with self._lock:
            return self._cycle

    def boot(self, *, restore: bool = True) -> dict[str, object]:
        with self._lock:
            if self._lifecycle is not OSLifecycle.OFFLINE:
                return self.status()
            self._lifecycle = OSLifecycle.READY
            self._halt_reason = None
            self._boot_time = monotonic()
            if restore:
                self._restore_checkpoint_if_present()
            return self.status()

    def shutdown(self) -> dict[str, object]:
        self.stop_autorun()
        with self._lock:
            self._persist_checkpoint()
            self._lifecycle = OSLifecycle.OFFLINE
            return self.status()

    def load(self, source: Mapping[Coordinate, float]) -> dict[str, object]:
        with self._lock:
            self._require_booted()
            field_config = self._field_config(self.policy)
            engine = DrMoagiAutoExecutionEngine(
                DrMoagiAutoExecConfig(
                    field_config=field_config,
                    policy=self.policy,
                    cycles=1,
                    auto_optimize=False,
                    max_reconstruction_mse=self.config.max_reconstruction_mse,
                )
            )
            parsed = engine.parser.parse(source)
            if len(parsed) > self.config.max_active_cells:
                raise RuntimeError("active-cell budget exceeded")
            self._state = dict(parsed)
            self._bitplane = SparseBitPlane3D.from_scalar_field(
                self._state,
                side=self.config.side,
                activation_threshold=self.config.activation_threshold,
            )
            self._cycle = 0
            self._loaded = True
            self._halt_reason = None
            self._last_report = None
            self._lifecycle = OSLifecycle.READY
            self._persist_checkpoint()
            return self.status()

    def snapshot(self, limit: int | None = None) -> dict[str, object]:
        with self._lock:
            self._require_loaded()
            selected_limit = self.config.snapshot_limit if limit is None else limit
            if (
                isinstance(selected_limit, bool)
                or not isinstance(selected_limit, int)
                or selected_limit <= 0
            ):
                raise ValueError("limit must be a positive integer")
            cells = [
                {"x": x, "y": y, "z": z, "value": float(self._state[(x, y, z)])}
                for x, y, z in sorted(self._state)[:selected_limit]
            ]
            return {
                "cycle": self._cycle,
                "total_active_cells": len(self._state),
                "returned_cells": len(cells),
                "truncated": len(cells) < len(self._state),
                "cells": cells,
                "state_hash": self._state_hash(self._state),
            }

    def bitplane(self, limit: int = 256) -> dict[str, object]:
        with self._lock:
            self._require_loaded()
            metrics = self._bitplane.metrics()
            return {
                "side": self._bitplane.side,
                "words_per_column": self._bitplane.words_per_column,
                "checksum_sha256": self._bitplane.checksum_sha256,
                "metrics": metrics.as_dict(),
                "words": self._bitplane.sample_words(limit),
            }

    def step(self) -> OSCycleReport:
        with self._lock:
            self._require_loaded()
            if self._lifecycle is OSLifecycle.HALTED:
                raise RuntimeError(f"kernel halted: {self._halt_reason or 'unknown reason'}")

            before = dict(self._state)
            previous_plane = self._bitplane
            active_before = len(before)
            rejection_reason: str | None = None
            fixed_report: FixedPointReport | None = None

            folded = fold_and_attenuate(
                before,
                side=self.config.side,
                contraction=self.config.contraction,
                attenuation=self.config.attenuation,
                prune_epsilon=self.policy.prune_epsilon,
            )
            if not folded and before:
                rejection_reason = "inward fold removed all active state"

            candidate = before
            auto_report = None
            candidate_policy = self.policy
            if rejection_reason is None:
                engine = DrMoagiAutoExecutionEngine(
                    self._autoexec_config(self.policy),
                    journal_path=None,
                )
                engine.load(folded)
                auto_report = engine.step()
                if not auto_report.committed:
                    rejection_reason = auto_report.rejection_reason or "autoexec rejected candidate"
                else:
                    candidate = engine.runtime.snapshot()
                    candidate_policy = engine.policy

            if rejection_reason is None and self.config.fixed_point_passes > 0:
                fixed_engine = DMvOmegaXiFixedPointEngine(
                    self._fixed_point_config(candidate_policy),
                    journal_path=None,
                )
                fixed_engine.load(candidate)
                for _ in range(self.config.fixed_point_passes):
                    fixed_report = fixed_engine.step()
                    if not fixed_report.committed or fixed_report.converged:
                        break
                if fixed_report is not None and not fixed_report.committed:
                    rejection_reason = fixed_report.rejection_reason or "fixed-point gate rejected candidate"
                else:
                    candidate = fixed_engine.snapshot()

            if rejection_reason is None:
                rejection_reason = self._validate_candidate(candidate)

            committed = rejection_reason is None
            if committed:
                self._state = dict(candidate)
                self.policy = candidate_policy
                self._cycle += 1
                self._bitplane = SparseBitPlane3D.from_scalar_field(
                    self._state,
                    side=self.config.side,
                    activation_threshold=self.config.activation_threshold,
                )
                self._lifecycle = (
                    OSLifecycle.RUNNING if self._autorun_thread is not None else OSLifecycle.READY
                )
            else:
                self._state = before
                self._bitplane = previous_plane
                self._lifecycle = OSLifecycle.HALTED
                self._halt_reason = rejection_reason

            metrics: BitPlaneMetrics = self._bitplane.metrics(previous_plane)
            reconstruction_mse = (
                float(auto_report.reconstruction_mse) if auto_report is not None else 0.0
            )
            policy_promoted = bool(auto_report.policy_promoted) if auto_report is not None else False
            fixed_residual = (
                float(fixed_report.fixed_point_residual) if fixed_report is not None else None
            )
            fixed_converged = bool(fixed_report.converged) if fixed_report is not None else False
            provisional = OSCycleReport(
                cycle=self._cycle,
                committed=committed,
                active_cells_before=active_before,
                active_cells_after=len(self._state),
                latent_cells=(int(auto_report.latent_cells) if auto_report is not None else 0),
                packed_words=metrics.packed_words,
                logical_words=metrics.logical_words,
                bit_density=metrics.density,
                phase_velocity=metrics.phase_velocity,
                spatial_entropy=metrics.entropy,
                kinetic_energy=metrics.kinetic_energy,
                reconstruction_mse=reconstruction_mse,
                fixed_point_residual=fixed_residual,
                fixed_point_converged=fixed_converged,
                policy_promoted=policy_promoted,
                policy=self.policy,
                rejection_reason=rejection_reason,
                state_hash=self._state_hash(self._state),
                bitplane_hash=self._bitplane.checksum_sha256,
            )
            record = provisional.as_dict()
            record.pop("journal_hash", None)
            digest = self.journal.append(record)
            report = OSCycleReport(**{**provisional.__dict__, "journal_hash": digest})
            self._last_report = report
            self._persist_checkpoint()
            return report

    def run(self, cycles: int) -> tuple[OSCycleReport, ...]:
        if isinstance(cycles, bool) or not isinstance(cycles, int) or cycles <= 0:
            raise ValueError("cycles must be a positive integer")
        reports: list[OSCycleReport] = []
        for _ in range(cycles):
            report = self.step()
            reports.append(report)
            if not report.committed or report.fixed_point_converged:
                break
        return tuple(reports)

    def start_autorun(self, interval_seconds: float | None = None) -> dict[str, object]:
        interval = (
            self.config.autorun_interval_seconds if interval_seconds is None else interval_seconds
        )
        if not math.isfinite(float(interval)) or interval <= 0.0:
            raise ValueError("interval_seconds must be finite and positive")
        with self._lock:
            self._require_loaded()
            if self._autorun_thread is not None and self._autorun_thread.is_alive():
                return self.status()
            if self._lifecycle is OSLifecycle.HALTED:
                raise RuntimeError("reset or reload state before restarting a halted kernel")
            self._autorun_stop.clear()
            thread = threading.Thread(
                target=self._autorun_worker,
                args=(float(interval),),
                daemon=True,
                name="dr-moagi-os-autorun",
            )
            self._autorun_thread = thread
            self._lifecycle = OSLifecycle.RUNNING
            thread.start()
            return self.status()

    def stop_autorun(self) -> dict[str, object]:
        with self._lock:
            thread = self._autorun_thread
            self._autorun_stop.set()
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        with self._lock:
            self._autorun_thread = None
            if self._lifecycle is OSLifecycle.RUNNING:
                self._lifecycle = OSLifecycle.READY
            return self.status()

    def reset_halt(self) -> dict[str, object]:
        with self._lock:
            self._require_booted()
            self._halt_reason = None
            self._lifecycle = OSLifecycle.READY
            return self.status()

    def status(self) -> dict[str, object]:
        with self._lock:
            scheduler_running = bool(
                self._autorun_thread is not None and self._autorun_thread.is_alive()
            )
            metrics = self._bitplane.metrics()
            return {
                "system": "Dr Moagi 3D OS",
                "mode": "bounded sparse auto-encoding/decoding control plane",
                "lifecycle": self._lifecycle.value,
                "loaded": self._loaded,
                "cycle": self._cycle,
                "uptime_seconds": max(0.0, monotonic() - self._boot_time),
                "side": self.config.side,
                "logical_cells": self.config.side**3,
                "active_cells": len(self._state),
                "policy": asdict(self.policy),
                "bitplane": metrics.as_dict(),
                "state_hash": self._state_hash(self._state),
                "journal_head": self.journal.head,
                "journal_valid": self.journal.verify(),
                "scheduler_running": scheduler_running,
                "halt_reason": self._halt_reason,
                "last_report": self._last_report.as_dict() if self._last_report else None,
            }

    def _autorun_worker(self, interval: float) -> None:
        try:
            while not self._autorun_stop.wait(interval):
                try:
                    report = self.step()
                except (RuntimeError, ValueError):
                    break
                if not report.committed or report.fixed_point_converged:
                    break
        finally:
            with self._lock:
                self._autorun_thread = None
                if self._lifecycle is OSLifecycle.RUNNING:
                    self._lifecycle = OSLifecycle.READY

    def _autoexec_config(self, policy: AutoExecPolicy) -> DrMoagiAutoExecConfig:
        return DrMoagiAutoExecConfig(
            field_config=self._field_config(policy),
            policy=policy,
            cycles=1,
            auto_optimize=self.config.auto_optimize,
            max_reconstruction_mse=self.config.max_reconstruction_mse,
        )

    def _field_config(self, policy: AutoExecPolicy) -> DrMoagiFieldConfig:
        return DrMoagiFieldConfig(
            side=self.config.side,
            alpha=1.0,
            lambda_residual=0.25,
            eta=0.05,
            dt=0.02,
            max_active_cells=self.config.max_active_cells,
            expand_halo=True,
            prune_epsilon=policy.prune_epsilon,
        )

    def _fixed_point_config(self, policy: AutoExecPolicy) -> DMvOmegaXiFixedPointConfig:
        return DMvOmegaXiFixedPointConfig(
            side=self.config.side,
            max_active_cells=self.config.max_active_cells,
            policy=policy,
            fixed_point_tolerance=self.config.fixed_point_tolerance,
            max_iterations=max(1, self.config.fixed_point_passes),
        )

    def _validate_candidate(self, candidate: Mapping[Coordinate, float]) -> str | None:
        if len(candidate) > self.config.max_active_cells:
            return "active-cell budget exceeded"
        if not candidate and self._state:
            return "candidate removed all active state"
        for coordinate, value in candidate.items():
            if any(axis < 0 or axis >= self.config.side for axis in coordinate):
                return "candidate coordinate outside logical lattice"
            if not math.isfinite(float(value)):
                return "candidate contains non-finite value"
        return None

    @staticmethod
    def _state_hash(field: Mapping[Coordinate, float]) -> str:
        canonical = [
            [x, y, z, float(field[(x, y, z)])]
            for x, y, z in sorted(field)
        ]
        payload = json.dumps(canonical, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @property
    def _checkpoint_path(self) -> Path | None:
        if self.config.state_dir is None:
            return None
        return self.config.state_dir / "checkpoint.json"

    def _persist_checkpoint(self) -> None:
        path = self._checkpoint_path
        if path is None or not self._loaded:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": self.CHECKPOINT_VERSION,
            "cycle": self._cycle,
            "policy": asdict(self.policy),
            "state": [
                [x, y, z, float(self._state[(x, y, z)])]
                for x, y, z in sorted(self._state)
            ],
            "journal_head": self.journal.head,
            "state_hash": self._state_hash(self._state),
        }
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        temp.replace(path)

    def _restore_checkpoint_if_present(self) -> None:
        path = self._checkpoint_path
        if path is None or not path.exists():
            return
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or raw.get("version") != self.CHECKPOINT_VERSION:
            raise ValueError("unsupported Dr Moagi OS checkpoint")
        if raw.get("journal_head") != self.journal.head:
            raise ValueError("checkpoint/journal head mismatch")
        state_rows = raw.get("state")
        if not isinstance(state_rows, list):
            raise ValueError("checkpoint state must be a list")
        state: SparseField = {}
        for row in state_rows:
            if not isinstance(row, list) or len(row) != 4:
                raise ValueError("invalid checkpoint state row")
            x, y, z, value = row
            if any(isinstance(axis, bool) or not isinstance(axis, int) for axis in (x, y, z)):
                raise ValueError("invalid checkpoint coordinate")
            state[(x, y, z)] = float(value)
        policy_raw = raw.get("policy")
        if not isinstance(policy_raw, dict):
            raise ValueError("checkpoint policy must be an object")
        restored_policy = AutoExecPolicy(
            block_size=int(policy_raw["block_size"]),
            quantization=float(policy_raw["quantization"]),
            prune_epsilon=float(policy_raw["prune_epsilon"]),
        )
        parser_engine = DrMoagiAutoExecutionEngine(
            DrMoagiAutoExecConfig(
                field_config=self._field_config(restored_policy),
                policy=restored_policy,
                cycles=1,
                auto_optimize=False,
            )
        )
        validated = parser_engine.parser.parse(state)
        expected_hash = raw.get("state_hash")
        if expected_hash != self._state_hash(validated):
            raise ValueError("checkpoint state hash mismatch")
        cycle_raw = raw.get("cycle", 0)
        if isinstance(cycle_raw, bool) or not isinstance(cycle_raw, int) or cycle_raw < 0:
            raise ValueError("checkpoint cycle is invalid")
        self.policy = restored_policy
        self._state = dict(validated)
        self._cycle = cycle_raw
        self._loaded = True
        self._bitplane = SparseBitPlane3D.from_scalar_field(
            self._state,
            side=self.config.side,
            activation_threshold=self.config.activation_threshold,
        )

    def _require_booted(self) -> None:
        if self._lifecycle is OSLifecycle.OFFLINE:
            raise RuntimeError("boot the Dr Moagi OS kernel before use")

    def _require_loaded(self) -> None:
        self._require_booted()
        if not self._loaded:
            raise RuntimeError("load a sparse 3D field before execution")


def demo_field(side: int) -> SparseField:
    """Return a deterministic 3D cross used by the CLI, API and smoke tests."""
    if isinstance(side, bool) or not isinstance(side, int) or side <= 0:
        raise ValueError("side must be a positive integer")
    center = side // 2
    field: SparseField = {(center, center, center): 1.0}
    for distance, value in ((1, 0.85), (2, 0.65), (3, 0.50)):
        for dx, dy, dz in (
            (distance, 0, 0),
            (-distance, 0, 0),
            (0, distance, 0),
            (0, -distance, 0),
            (0, 0, distance),
            (0, 0, -distance),
        ):
            coordinate = center + dx, center + dy, center + dz
            if all(0 <= axis < side for axis in coordinate):
                field[coordinate] = value
    return field
