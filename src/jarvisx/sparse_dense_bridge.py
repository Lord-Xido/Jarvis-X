"""Transactional sparse-to-dense bridge for the Dr Moagi 3D runtimes.

The sparse 1000x1000 lattice remains the logical scheduling/authority surface.
Dense backends receive bounded tiles and return candidates.  Candidates are
scattered only onto already-authoritative sparse coordinates, then admitted or
rolled back through an explicit numerical/epistemic/authority gate.

The bridge deliberately keeps two geometries distinct:

* sparse X/Y coordinates identify logical processing loops;
* dense D/H/W coordinates identify a bounded numerical tile presented to a
  neural or other dense backend.

The default embedding places sparse loop values on the centre depth plane.  It
does not identify recursive sparse encode/decode depth with physical world Z.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Mapping, Protocol, Sequence

from .candidate_receipt import (
    CandidateReceipt,
    VerificationVector,
    build_candidate_receipt,
    hash_sparse_field,
)
from .dr_moagi_multiparallel import (
    DrMoagiMultiparallel3D,
    LoopCoordinate,
    SparseLoopField,
    Vector,
)


@dataclass(frozen=True)
class ExternalCandidateMetrics:
    cycle: int
    active_loops_before: int
    active_loops_after: int
    channel_count: int
    candidate_mse: float
    max_abs_delta: float
    state_delta_rms: float
    global_core_delta_rms: float
    committed: bool
    rejection_reason: str | None = None


ExternalValidator = Callable[[Mapping[LoopCoordinate, Vector], ExternalCandidateMetrics], bool]
EpistemicValidator = Callable[[Mapping[LoopCoordinate, Vector]], bool]


class SparseDenseTransactionalRuntime(DrMoagiMultiparallel3D):
    """Sparse runtime extended with candidate-first external backend admission."""

    def commit_external_candidate(
        self,
        field: Mapping[LoopCoordinate, Sequence[float]],
        *,
        max_candidate_mse: float | None = None,
        validator: ExternalValidator | None = None,
    ) -> ExternalCandidateMetrics:
        self._require_loaded()
        before = self.snapshot_surface()
        parsed: SparseLoopField = {}
        for raw_coordinate, raw_vector in field.items():
            coordinate = self._validate_coordinate(raw_coordinate)
            vector = self._validate_vector(raw_vector)
            if len(vector) != self.channel_count:
                raise ValueError("external candidate channel count must match the loaded runtime")
            if self._active(vector):
                parsed[coordinate] = vector
            if len(parsed) > self.config.max_active_loops:
                raise RuntimeError("external candidate active-loop budget exceeded")

        if max_candidate_mse is None:
            budget = self.config.max_reconstruction_mse
        else:
            if isinstance(max_candidate_mse, bool) or not isinstance(max_candidate_mse, (int, float)):
                raise TypeError("max_candidate_mse must be numeric")
            budget = float(max_candidate_mse)
            if not math.isfinite(budget) or budget < 0.0:
                raise ValueError("max_candidate_mse must be finite and non-negative")

        coordinates = sorted(set(before) | set(parsed), key=self._linear_address)
        zero = self._zero()
        squared_delta = 0.0
        max_abs_delta = 0.0
        samples = 0
        for coordinate in coordinates:
            prior = before.get(coordinate, zero)
            proposed = parsed.get(coordinate, zero)
            for left, right in zip(prior, proposed):
                delta = right - left
                squared_delta += delta * delta
                max_abs_delta = max(max_abs_delta, abs(delta))
                samples += 1
        candidate_mse = squared_delta / samples if samples else 0.0
        state_delta_rms = math.sqrt(candidate_mse)

        volume, global_core = self._fold_volume(parsed)
        old_core = self.global_core if self.global_core else zero
        new_core = global_core if global_core else zero
        core_sq = sum((a - b) ** 2 for a, b in zip(old_core, new_core))
        global_core_delta_rms = math.sqrt(core_sq / self.channel_count) if self.channel_count else 0.0
        cycle = self.cycle + 1
        provisional = ExternalCandidateMetrics(
            cycle=cycle,
            active_loops_before=len(before),
            active_loops_after=len(parsed),
            channel_count=self.channel_count,
            candidate_mse=candidate_mse,
            max_abs_delta=max_abs_delta,
            state_delta_rms=state_delta_rms,
            global_core_delta_rms=global_core_delta_rms,
            committed=False,
        )

        if candidate_mse > budget:
            self._cycle = cycle
            return ExternalCandidateMetrics(
                **{
                    **provisional.__dict__,
                    "rejection_reason": "candidate distortion budget exceeded",
                }
            )
        if validator is not None and not bool(validator(parsed, provisional)):
            self._cycle = cycle
            return ExternalCandidateMetrics(
                **{
                    **provisional.__dict__,
                    "rejection_reason": "external validator rejected candidate",
                }
            )

        new_memory: SparseLoopField = {}
        new_error: SparseLoopField = {}
        new_velocity: SparseLoopField = {}
        for coordinate, value in parsed.items():
            prior = before.get(coordinate, zero)
            delta = self._sub(value, prior)
            new_error[coordinate] = delta
            previous_memory = self._memory.get(coordinate, zero)
            memory = self._add(
                self._scale(previous_memory, self.config.memory_decay),
                self._scale(delta, 1.0 - self.config.memory_decay),
            )
            if self._active(memory):
                new_memory[coordinate] = memory
            new_velocity[coordinate] = self._velocity.get(coordinate, zero)

        self._surface = parsed
        self._velocity = new_velocity
        self._memory = new_memory
        self._error = new_error
        self._volume = volume
        self._global_core = global_core
        self._cycle = cycle
        return ExternalCandidateMetrics(
            cycle=cycle,
            active_loops_before=len(before),
            active_loops_after=len(parsed),
            channel_count=self.channel_count,
            candidate_mse=candidate_mse,
            max_abs_delta=max_abs_delta,
            state_delta_rms=state_delta_rms,
            global_core_delta_rms=global_core_delta_rms,
            committed=True,
        )


@dataclass(frozen=True)
class DenseTile:
    tile_key: tuple[int, int]
    origin: tuple[int, int]
    side: int
    depth: int
    channels: int
    centre_depth: int
    active_coordinates: tuple[LoopCoordinate, ...]
    values: tuple[float, ...]

    @property
    def element_count(self) -> int:
        return self.channels * self.depth * self.side * self.side


@dataclass(frozen=True)
class DenseTileResult:
    values: tuple[float, ...]
    metrics: tuple[tuple[str, float | int | bool], ...] = ()


class DenseTileBackend(Protocol):
    def process(self, tile: DenseTile) -> DenseTileResult:
        """Transform one bounded dense tile without mutating sparse authority."""


@dataclass(frozen=True)
class SparseDenseBridgeConfig:
    tile_side: int = 16
    tile_depth: int = 16
    max_tiles: int = 256
    max_candidate_mse: float = 0.25
    output_blend: float = 1.0
    convergence_tol: float = 1.0e-5

    def __post_init__(self) -> None:
        if isinstance(self.tile_side, bool) or not isinstance(self.tile_side, int) or self.tile_side <= 0:
            raise ValueError("tile_side must be a positive integer")
        if isinstance(self.tile_depth, bool) or not isinstance(self.tile_depth, int) or self.tile_depth <= 0:
            raise ValueError("tile_depth must be a positive integer")
        if isinstance(self.max_tiles, bool) or not isinstance(self.max_tiles, int) or self.max_tiles <= 0:
            raise ValueError("max_tiles must be a positive integer")
        for name in ("max_candidate_mse", "output_blend", "convergence_tol"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.max_candidate_mse < 0.0:
            raise ValueError("max_candidate_mse must be non-negative")
        if not 0.0 <= self.output_blend <= 1.0:
            raise ValueError("output_blend must be in [0, 1]")
        if self.convergence_tol < 0.0:
            raise ValueError("convergence_tol must be non-negative")


@dataclass(frozen=True)
class SparseDenseBridgeRun:
    receipt: CandidateReceipt
    tile_count: int
    candidate_mse: float
    state_delta_rms: float
    global_core_delta_rms: float
    converged: bool


class SparseDenseTileBridge:
    """Gather sparse loops, execute dense tiles, scatter a guarded candidate."""

    def __init__(self, config: SparseDenseBridgeConfig | None = None) -> None:
        self.config = config or SparseDenseBridgeConfig()

    def run(
        self,
        runtime: SparseDenseTransactionalRuntime,
        backend: DenseTileBackend,
        *,
        epistemic_validator: EpistemicValidator | None = None,
        authority_validator: ExternalValidator | None = None,
    ) -> SparseDenseBridgeRun:
        parent = runtime.snapshot_surface()
        if not parent:
            raise RuntimeError("sparse-dense bridge requires at least one active sparse loop")
        groups = self._group_coordinates(parent)
        if len(groups) > self.config.max_tiles:
            raise RuntimeError("dense tile budget exceeded")

        candidate: SparseLoopField = dict(parent)
        backend_metric_count = 0
        for tile_key in sorted(groups):
            tile = self._pack_tile(runtime, parent, tile_key, groups[tile_key])
            result = backend.process(tile)
            self._validate_result(tile, result)
            backend_metric_count += len(result.metrics)
            self._scatter_tile(candidate, parent, tile, result)

        epistemic_pass = True
        epistemic_status = "NOT_APPLICABLE"
        if epistemic_validator is not None:
            epistemic_pass = bool(epistemic_validator(candidate))
            epistemic_status = "PASS" if epistemic_pass else "FAIL"

        authority_pass = True

        def combined_validator(
            proposed: Mapping[LoopCoordinate, Vector], metrics: ExternalCandidateMetrics
        ) -> bool:
            nonlocal authority_pass
            if not epistemic_pass:
                authority_pass = False
                return False
            if authority_validator is None:
                return True
            authority_pass = bool(authority_validator(proposed, metrics))
            return authority_pass

        parent_hash = hash_sparse_field(parent)
        candidate_hash = hash_sparse_field(candidate)
        external = runtime.commit_external_candidate(
            candidate,
            max_candidate_mse=self.config.max_candidate_mse,
            validator=combined_validator,
        )
        after = runtime.snapshot_surface()
        after_hash = hash_sparse_field(after)
        numerical_pass = external.candidate_mse <= self.config.max_candidate_mse
        if not external.committed and external.rejection_reason == "candidate distortion budget exceeded":
            numerical_pass = False
        if not external.committed and external.rejection_reason == "external validator rejected candidate":
            authority_pass = False

        verification = VerificationVector(
            integrity="PASS",
            numerical="PASS" if numerical_pass else "FAIL",
            epistemic=epistemic_status,  # type: ignore[arg-type]
            authority="PASS" if external.committed else "FAIL",
        )
        hard_constraints = {
            "tile_budget": len(groups) <= self.config.max_tiles,
            "candidate_distortion": numerical_pass,
            "epistemic_gate": epistemic_pass,
            "authority_gate": authority_pass,
        }
        receipt = build_candidate_receipt(
            subsystem="dr-moagi-sparse-dense",
            operator="gather->dense-backend->scatter->verify->commit",
            state_scope="sparse_surface",
            parent_state_hash=parent_hash,
            candidate_state_hash=candidate_hash,
            after_state_hash=after_hash,
            objective_before=0.0,
            candidate_objective=external.candidate_mse,
            objective_after=external.candidate_mse if external.committed else 0.0,
            metrics={
                "cycle": external.cycle,
                "tile_count": len(groups),
                "backend_metric_count": backend_metric_count,
                "active_loops_before": external.active_loops_before,
                "active_loops_after": external.active_loops_after,
                "candidate_mse": external.candidate_mse,
                "max_abs_delta": external.max_abs_delta,
                "state_delta_rms": external.state_delta_rms,
                "global_core_delta_rms": external.global_core_delta_rms,
            },
            hard_constraints=hard_constraints,
            verification=verification,
            committed=external.committed,
            rejection_reason=external.rejection_reason,
        )
        converged = bool(
            external.committed
            and external.state_delta_rms <= self.config.convergence_tol
            and external.global_core_delta_rms <= self.config.convergence_tol
        )
        return SparseDenseBridgeRun(
            receipt=receipt,
            tile_count=len(groups),
            candidate_mse=external.candidate_mse,
            state_delta_rms=external.state_delta_rms,
            global_core_delta_rms=external.global_core_delta_rms,
            converged=converged,
        )

    def run_until_converged(
        self,
        runtime: SparseDenseTransactionalRuntime,
        backend: DenseTileBackend,
        *,
        max_rounds: int = 4,
        epistemic_validator: EpistemicValidator | None = None,
        authority_validator: ExternalValidator | None = None,
    ) -> tuple[SparseDenseBridgeRun, ...]:
        if isinstance(max_rounds, bool) or not isinstance(max_rounds, int) or max_rounds <= 0:
            raise ValueError("max_rounds must be a positive integer")
        runs: list[SparseDenseBridgeRun] = []
        for _ in range(max_rounds):
            result = self.run(
                runtime,
                backend,
                epistemic_validator=epistemic_validator,
                authority_validator=authority_validator,
            )
            runs.append(result)
            if not result.receipt.committed or result.converged:
                break
        return tuple(runs)

    def _group_coordinates(
        self, field: Mapping[LoopCoordinate, Vector]
    ) -> dict[tuple[int, int], tuple[LoopCoordinate, ...]]:
        grouped: dict[tuple[int, int], list[LoopCoordinate]] = {}
        side = self.config.tile_side
        for coordinate in sorted(field):
            key = coordinate[0] // side, coordinate[1] // side
            grouped.setdefault(key, []).append(coordinate)
        return {key: tuple(value) for key, value in grouped.items()}

    def _pack_tile(
        self,
        runtime: SparseDenseTransactionalRuntime,
        field: Mapping[LoopCoordinate, Vector],
        tile_key: tuple[int, int],
        coordinates: Sequence[LoopCoordinate],
    ) -> DenseTile:
        side = self.config.tile_side
        depth = self.config.tile_depth
        channels = runtime.channel_count
        centre = depth // 2
        values = [0.0] * (channels * depth * side * side)
        origin = tile_key[0] * side, tile_key[1] * side
        for coordinate in coordinates:
            local_x = coordinate[0] - origin[0]
            local_y = coordinate[1] - origin[1]
            vector = field[coordinate]
            for channel, value in enumerate(vector):
                values[self._flat_index(channel, centre, local_y, local_x, depth, side)] = value
        return DenseTile(
            tile_key=tile_key,
            origin=origin,
            side=side,
            depth=depth,
            channels=channels,
            centre_depth=centre,
            active_coordinates=tuple(coordinates),
            values=tuple(values),
        )

    def _scatter_tile(
        self,
        candidate: SparseLoopField,
        parent: Mapping[LoopCoordinate, Vector],
        tile: DenseTile,
        result: DenseTileResult,
    ) -> None:
        blend = self.config.output_blend
        for coordinate in tile.active_coordinates:
            local_x = coordinate[0] - tile.origin[0]
            local_y = coordinate[1] - tile.origin[1]
            output: list[float] = []
            for channel in range(tile.channels):
                index = self._flat_index(
                    channel,
                    tile.centre_depth,
                    local_y,
                    local_x,
                    tile.depth,
                    tile.side,
                )
                value = float(result.values[index])
                source = parent[coordinate][channel]
                output.append((1.0 - blend) * source + blend * value)
            candidate[coordinate] = tuple(output)

    @staticmethod
    def _validate_result(tile: DenseTile, result: DenseTileResult) -> None:
        if len(result.values) != tile.element_count:
            raise ValueError("dense backend result shape does not match input tile")
        if any(not math.isfinite(float(value)) for value in result.values):
            raise ValueError("dense backend returned non-finite values")
        for key, value in result.metrics:
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"dense backend metric {key!r} is non-finite")

    @staticmethod
    def _flat_index(channel: int, depth: int, row: int, column: int, depth_size: int, side: int) -> int:
        return (((channel * depth_size + depth) * side + row) * side) + column
