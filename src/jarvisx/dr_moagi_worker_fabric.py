"""Sparse virtual 6400^3 logical worker fabric for the Dr Moagi runtime.

The fabric exposes 262,144,000,000 addressable logical workers without
materializing one Python object or native thread per worker. Logical workers are
partitioned into 64^3-worker bricks and scheduled through a bounded native
thread pool over an explicitly measured active set.

This module is a reference scheduler/control-plane implementation. Logical
worker count, resident working set, executed active work and physical worker
threads are deliberately reported as separate quantities.
"""
from __future__ import annotations

import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any

MASK64 = (1 << 64) - 1


@dataclass(frozen=True)
class FabricConfig:
    side: int = 6_400
    brick_side: int = 64
    physical_workers: int = field(
        default_factory=lambda: max(1, min(os.cpu_count() or 1, 64))
    )
    active_worker_budget: int = 4_096
    max_active_workers: int = 100_000
    resident_brick_limit: int = 128
    seed: int = 41

    def validate(self) -> None:
        if self.side != 6_400:
            raise ValueError("the canonical worker-fabric side is fixed at 6400")
        if self.brick_side <= 0 or self.side % self.brick_side != 0:
            raise ValueError("brick_side must divide side exactly")
        if not (1 <= self.physical_workers <= 256):
            raise ValueError("physical_workers must be in [1, 256]")
        if not (1 <= self.active_worker_budget <= self.max_active_workers):
            raise ValueError("active_worker_budget must be within the active-work bound")
        if not (self.active_worker_budget <= self.max_active_workers <= 1_000_000):
            raise ValueError("max_active_workers must be in [active_worker_budget, 1000000]")
        if self.resident_brick_limit <= 0:
            raise ValueError("resident_brick_limit must be positive")


class LogicalWorkerFabric:
    """Bounded executor over a 6400 x 6400 x 6400 logical worker lattice."""

    def __init__(self, config: FabricConfig | None = None) -> None:
        self.config = config or FabricConfig()
        self.config.validate()
        self._cycle = 0
        self._lock = threading.RLock()
        self._last_receipt: dict[str, Any] | None = None
        self._pool = ThreadPoolExecutor(
            max_workers=self.config.physical_workers,
            thread_name_prefix="dm6400cube",
        )

    @property
    def logical_workers(self) -> int:
        return self.config.side ** 3

    @property
    def bricks_per_axis(self) -> int:
        return self.config.side // self.config.brick_side

    @property
    def logical_bricks(self) -> int:
        return self.bricks_per_axis ** 3

    @property
    def workers_per_brick(self) -> int:
        return self.config.brick_side ** 3

    def linear_id(self, x: int, y: int, z: int) -> int:
        side = self.config.side
        if not (0 <= x < side and 0 <= y < side and 0 <= z < side):
            raise ValueError("worker coordinate outside 6400^3 lattice")
        return x + side * y + side * side * z

    def coordinate(self, worker_id: int) -> tuple[int, int, int]:
        if not (0 <= worker_id < self.logical_workers):
            raise ValueError("worker_id outside logical worker domain")
        side = self.config.side
        plane = side * side
        z, rem = divmod(worker_id, plane)
        y, x = divmod(rem, side)
        return x, y, z

    def brick_coordinate(self, x: int, y: int, z: int) -> tuple[int, int, int]:
        self.linear_id(x, y, z)
        b = self.config.brick_side
        return x // b, y // b, z // b

    def brick_id(self, x: int, y: int, z: int) -> int:
        bx, by, bz = self.brick_coordinate(x, y, z)
        n = self.bricks_per_axis
        return bx + n * by + n * n * bz

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "mode": "sparse-virtualized",
                "logical_dimensions": [self.config.side] * 3,
                "logical_workers": self.logical_workers,
                "brick_side": self.config.brick_side,
                "brick_dimensions": [self.bricks_per_axis] * 3,
                "logical_bricks": self.logical_bricks,
                "workers_per_brick": self.workers_per_brick,
                "physical_workers": self.config.physical_workers,
                "active_worker_budget": self.config.active_worker_budget,
                "max_active_workers": self.config.max_active_workers,
                "resident_brick_limit": self.config.resident_brick_limit,
                "cycle": self._cycle,
                "last_receipt": self._last_receipt,
                "semantics": {
                    "logical_worker": "addressable work-domain coordinate, not a native thread",
                    "physical_worker": "bounded host thread used by the reference scheduler",
                    "resident_brick": "bounded active working-set partition",
                },
            }

    @staticmethod
    def _splitmix64(value: int) -> int:
        value = (value + 0x9E3779B97F4A7C15) & MASK64
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
        return (value ^ (value >> 31)) & MASK64

    def _worker_id_for(self, cycle: int, ordinal: int) -> int:
        key = (
            (self.config.seed & MASK64)
            ^ ((cycle + 1) * 0xD6E8FEB86659FD93)
            ^ ((ordinal + 1) * 0xA0761D6478BD642F)
        ) & MASK64
        return self._splitmix64(key) % self.logical_workers

    def _kernel(self, worker_id: int, cycle: int) -> tuple[float, float, int]:
        x, y, z = self.coordinate(worker_id)
        side = float(self.config.side)
        gx = (x + 0.5) / side - 0.5
        gy = (y + 0.5) / side - 0.5
        gz = (z + 0.5) / side - 0.5

        wave = (
            math.sin((x + 1) * 0.0017 + cycle * 0.031)
            + math.cos((y + 1) * 0.0013 - cycle * 0.019)
            + math.sin((z + 1) * 0.0011 + cycle * 0.023)
        )
        target = math.tanh(0.36 * wave + 0.12 * (gx - gy + gz))
        encoded = math.tanh(0.72 * target + 0.18 * (gx + gy - gz))
        folded = 0.82 * encoded + 0.18 * math.tanh(1.10 * encoded)
        decoded = math.tanh(1.08 * folded - 0.045 * (gx + gy + gz))
        residual = target - decoded
        checksum = decoded * (1.0 + ((worker_id & 0xFFFF) / 65535.0))
        return residual * residual, checksum, self.brick_id(x, y, z)

    def _run_chunk(self, cycle: int, begin: int, end: int) -> tuple[int, float, float, set[int]]:
        sum_sq = 0.0
        checksum = 0.0
        bricks: set[int] = set()
        for ordinal in range(begin, end):
            worker_id = self._worker_id_for(cycle, ordinal)
            sq, value, brick = self._kernel(worker_id, cycle)
            sum_sq += sq
            checksum += value
            bricks.add(brick)
        return end - begin, sum_sq, checksum, bricks

    def _sample(self, cycle: int, sample_size: int) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for ordinal in range(sample_size):
            worker_id = self._worker_id_for(cycle, ordinal)
            x, y, z = self.coordinate(worker_id)
            sq, checksum, brick = self._kernel(worker_id, cycle)
            out.append(
                {
                    "worker_id": worker_id,
                    "xyz": [x, y, z],
                    "brick_id": brick,
                    "residual_abs": math.sqrt(sq),
                    "state": checksum / (1.0 + ((worker_id & 0xFFFF) / 65535.0)),
                }
            )
        return out

    def step(
        self,
        *,
        active_workers: int | None = None,
        sample_size: int = 256,
    ) -> dict[str, Any]:
        active = self.config.active_worker_budget if active_workers is None else int(active_workers)
        if not (1 <= active <= self.config.max_active_workers):
            raise ValueError(
                f"active_workers must be in [1, {self.config.max_active_workers}]"
            )
        if not (0 <= sample_size <= min(active, 2_048)):
            raise ValueError("sample_size must be in [0, min(active_workers, 2048)]")

        with self._lock:
            cycle = self._cycle
            started = time.perf_counter()
            task_count = min(self.config.physical_workers, active)
            base, remainder = divmod(active, task_count)
            futures = []
            begin = 0
            for task in range(task_count):
                width = base + (1 if task < remainder else 0)
                end = begin + width
                futures.append(self._pool.submit(self._run_chunk, cycle, begin, end))
                begin = end

            executed = 0
            sum_sq = 0.0
            checksum = 0.0
            bricks: set[int] = set()
            for future in futures:
                count, chunk_sq, chunk_checksum, chunk_bricks = future.result()
                executed += count
                sum_sq += chunk_sq
                checksum += chunk_checksum
                bricks.update(chunk_bricks)

            elapsed_ms = (time.perf_counter() - started) * 1000.0
            unique_bricks = len(bricks)
            receipt: dict[str, Any] = {
                "cycle": cycle,
                "logical_dimensions": [self.config.side] * 3,
                "logical_workers": self.logical_workers,
                "logical_bricks": self.logical_bricks,
                "active_workers_executed": executed,
                "execution_fraction": executed / self.logical_workers,
                "active_bricks_touched": unique_bricks,
                "resident_bricks": min(unique_bricks, self.config.resident_brick_limit),
                "resident_brick_limit": self.config.resident_brick_limit,
                "physical_workers": self.config.physical_workers,
                "residual_rms": math.sqrt(sum_sq / executed),
                "checksum": checksum,
                "elapsed_ms": elapsed_ms,
                "sample": self._sample(cycle, sample_size),
            }
            self._cycle += 1
            self._last_receipt = {k: v for k, v in receipt.items() if k != "sample"}
            return receipt

    def config_dict(self) -> dict[str, Any]:
        return asdict(self.config)
