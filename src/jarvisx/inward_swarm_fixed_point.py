"""Monotone inward-swarm fixed-point operator for the Dr Moagi runtime.

This module is the bounded, deterministic reference for the discrete inward
contraction discussed by ADR-014.  It deliberately separates a *logical*
``scale**3`` coordinate domain from the finite set of agents that are actually
materialized.

Two invariants are enforced on every admitted step:

1. spatial L1 distance to the declared core never increases; and
2. packed-state Hamming distance to the declared latent target never increases.

If the swarm is not already at the joint fixed point, their positive weighted
objective must decrease strictly.  This prevents the overshoot/periodic behaviour
of an un-clipped ``+/- step`` spatial update and prevents a reversible Gray-code
mix such as ``s ^ (s >> 1)`` from being mislabeled as contraction.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Sequence

Coordinate3D = tuple[int, int, int]
PackedWords = tuple[int, ...]


def _require_plain_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def clipped_contract_axis(value: int, center: int, max_step: int) -> int:
    """Move one integer coordinate toward ``center`` without overshooting."""

    value = _require_plain_int("value", value)
    center = _require_plain_int("center", center)
    max_step = _require_plain_int("max_step", max_step)
    if max_step <= 0:
        raise ValueError("max_step must be positive")

    delta = center - value
    if abs(delta) <= max_step:
        return center
    return value + (max_step if delta > 0 else -max_step)


def clipped_contract_position(
    position: Coordinate3D,
    center: Coordinate3D,
    max_step: int,
) -> Coordinate3D:
    """Apply :func:`clipped_contract_axis` independently on x, y and z."""

    if len(position) != 3 or len(center) != 3:
        raise ValueError("position and center must be 3D coordinates")
    return tuple(
        clipped_contract_axis(int(value), int(target), max_step)
        for value, target in zip(position, center)
    )  # type: ignore[return-value]


def hamming_distance_word(left: int, right: int, word_bits: int = 31) -> int:
    """Return Hamming distance between two bounded unsigned packed words."""

    left = _require_plain_int("left", left)
    right = _require_plain_int("right", right)
    word_bits = _require_plain_int("word_bits", word_bits)
    if word_bits <= 0 or word_bits > 63:
        raise ValueError("word_bits must be in [1, 63]")
    limit = 1 << word_bits
    if not 0 <= left < limit or not 0 <= right < limit:
        raise ValueError("packed word is outside configured word_bits")
    return (left ^ right).bit_count()


def refine_word_toward_target(current: int, target: int, word_bits: int = 31) -> int:
    """Correct exactly one differing bit, making Hamming error strictly contractive.

    ``diff & -diff`` isolates the least-significant bit on which ``current`` and
    ``target`` disagree.  XORing that bit into ``current`` makes it agree with
    ``target``.  A non-fixed word therefore loses exactly one Hamming-error bit
    per call and converges in at most ``word_bits`` calls.
    """

    distance = hamming_distance_word(current, target, word_bits)
    if distance == 0:
        return current
    diff = current ^ target
    correction = diff & -diff
    refined = current ^ correction
    if hamming_distance_word(refined, target, word_bits) != distance - 1:
        raise RuntimeError("bit contraction invariant violated")
    return refined


@dataclass(frozen=True)
class InwardSwarmConfig:
    """Finite execution contract for a sparse swarm in a large logical domain."""

    scale: int = 1_000_000
    spatial_step: int = 1_000
    word_bits: int = 31
    words_per_agent: int = 64
    max_iterations: int = 512
    spatial_weight: float = 1.0
    latent_weight: float = 1.0

    def __post_init__(self) -> None:
        for name in ("scale", "spatial_step", "word_bits", "words_per_agent", "max_iterations"):
            value = _require_plain_int(name, getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.scale < 2:
            raise ValueError("scale must be at least 2")
        if self.word_bits > 31:
            raise ValueError("word_bits must be <= 31 for the int32 accelerator contract")
        for name in ("spatial_weight", "latent_weight"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")

    @property
    def center(self) -> Coordinate3D:
        c = self.scale // 2
        return c, c, c

    @property
    def domain_worst_case_iterations(self) -> int:
        """Finite upper bound for any state admitted by this configuration."""

        c = self.scale // 2
        farthest = max(c, (self.scale - 1) - c)
        spatial = math.ceil(farthest / self.spatial_step)
        return max(spatial, self.word_bits)


@dataclass(frozen=True)
class InwardSwarmReport:
    iteration: int
    spatial_l1_before: int
    spatial_l1_after: int
    latent_hamming_before: int
    latent_hamming_after: int
    objective_before: float
    objective_after: float
    strict_contraction: bool
    spatial_converged: bool
    latent_converged: bool
    converged: bool
    state_hash: str


class InwardSwarmFixedPointEngine:
    """Sparse joint spatial/latent contraction engine.

    The encoded/decoded multimodal runtime may supply an arbitrary finite swarm
    and explicit packed latent targets.  This class does not infer that target;
    it guarantees only that the discrete inward operator reaches the supplied
    target monotonically under the configured bounds.
    """

    OPERATOR = "T_in"
    EQUATION = (
        "P[k+1]=P[k]+sign(C-P[k])*min(abs(C-P[k]),Delta); "
        "S[k+1]=S[k] xor LSB(S[k] xor S*)"
    )

    def __init__(self, config: InwardSwarmConfig | None = None) -> None:
        self.config = config or InwardSwarmConfig()
        self._positions: tuple[Coordinate3D, ...] = ()
        self._states: tuple[PackedWords, ...] = ()
        self._targets: tuple[PackedWords, ...] = ()
        self._iteration = 0
        self._loaded = False
        self.reports: list[InwardSwarmReport] = []

    def load(
        self,
        positions: Sequence[Coordinate3D],
        states: Sequence[Sequence[int]],
        targets: Sequence[Sequence[int]],
    ) -> None:
        if not positions:
            raise ValueError("at least one active agent is required")
        if len(positions) != len(states) or len(states) != len(targets):
            raise ValueError("positions, states and targets must have the same agent count")

        normalized_positions: list[Coordinate3D] = []
        normalized_states: list[PackedWords] = []
        normalized_targets: list[PackedWords] = []
        limit = 1 << self.config.word_bits

        for agent, (position, state, target) in enumerate(zip(positions, states, targets)):
            if len(position) != 3:
                raise ValueError(f"agent {agent} position must contain exactly three coordinates")
            coordinate = tuple(_require_plain_int("coordinate", axis) for axis in position)
            if any(axis < 0 or axis >= self.config.scale for axis in coordinate):
                raise ValueError(f"agent {agent} position is outside the logical domain")
            if len(state) != self.config.words_per_agent or len(target) != self.config.words_per_agent:
                raise ValueError(
                    f"agent {agent} requires exactly {self.config.words_per_agent} packed words"
                )
            state_words = tuple(_require_plain_int("state word", word) for word in state)
            target_words = tuple(_require_plain_int("target word", word) for word in target)
            if any(word < 0 or word >= limit for word in state_words + target_words):
                raise ValueError(f"agent {agent} packed word is outside configured word_bits")
            normalized_positions.append(coordinate)  # type: ignore[arg-type]
            normalized_states.append(state_words)
            normalized_targets.append(target_words)

        self._positions = tuple(normalized_positions)
        self._states = tuple(normalized_states)
        self._targets = tuple(normalized_targets)
        self._iteration = 0
        self._loaded = True
        self.reports.clear()

    def positions(self) -> tuple[Coordinate3D, ...]:
        self._require_loaded()
        return self._positions

    def states(self) -> tuple[PackedWords, ...]:
        self._require_loaded()
        return self._states

    def targets(self) -> tuple[PackedWords, ...]:
        self._require_loaded()
        return self._targets

    def _errors(self) -> tuple[int, int]:
        center = self.config.center
        spatial = sum(
            abs(position[0] - center[0])
            + abs(position[1] - center[1])
            + abs(position[2] - center[2])
            for position in self._positions
        )
        latent = sum(
            hamming_distance_word(current, target, self.config.word_bits)
            for state, target_state in zip(self._states, self._targets)
            for current, target in zip(state, target_state)
        )
        return spatial, latent

    def _objective(self, spatial: int, latent: int) -> float:
        return self.config.spatial_weight * spatial + self.config.latent_weight * latent

    def theoretical_iterations_remaining(self) -> int:
        """Return an exact upper bound for the currently loaded finite swarm."""

        self._require_loaded()
        center = self.config.center
        max_spatial_distance = max(
            abs(axis - target)
            for position in self._positions
            for axis, target in zip(position, center)
        )
        spatial_steps = math.ceil(max_spatial_distance / self.config.spatial_step)
        max_word_hamming = max(
            (
                hamming_distance_word(current, target, self.config.word_bits)
                for state, target_state in zip(self._states, self._targets)
                for current, target in zip(state, target_state)
            ),
            default=0,
        )
        return max(spatial_steps, max_word_hamming)

    def step(self) -> InwardSwarmReport:
        self._require_loaded()
        spatial_before, latent_before = self._errors()
        objective_before = self._objective(spatial_before, latent_before)
        was_converged = spatial_before == 0 and latent_before == 0

        center = self.config.center
        positions = tuple(
            clipped_contract_position(position, center, self.config.spatial_step)
            for position in self._positions
        )
        states = tuple(
            tuple(
                refine_word_toward_target(current, target, self.config.word_bits)
                for current, target in zip(state, target_state)
            )
            for state, target_state in zip(self._states, self._targets)
        )

        self._positions = positions
        self._states = states
        self._iteration += 1

        spatial_after, latent_after = self._errors()
        objective_after = self._objective(spatial_after, latent_after)
        if spatial_after > spatial_before:
            raise RuntimeError("spatial contraction invariant violated")
        if latent_after > latent_before:
            raise RuntimeError("latent contraction invariant violated")
        strict = objective_after < objective_before
        if not was_converged and not strict:
            raise RuntimeError("joint Lyapunov objective failed to decrease strictly")

        spatial_converged = spatial_after == 0
        latent_converged = latent_after == 0
        converged = spatial_converged and latent_converged
        report = InwardSwarmReport(
            iteration=self._iteration,
            spatial_l1_before=spatial_before,
            spatial_l1_after=spatial_after,
            latent_hamming_before=latent_before,
            latent_hamming_after=latent_after,
            objective_before=objective_before,
            objective_after=objective_after,
            strict_contraction=strict,
            spatial_converged=spatial_converged,
            latent_converged=latent_converged,
            converged=converged,
            state_hash=self._state_hash(),
        )
        self.reports.append(report)
        return report

    def run_until_fixed_point(self, max_iterations: int | None = None) -> tuple[InwardSwarmReport, ...]:
        self._require_loaded()
        limit = self.config.max_iterations if max_iterations is None else max_iterations
        limit = _require_plain_int("max_iterations", limit)
        if limit <= 0:
            raise ValueError("max_iterations must be positive")
        for _ in range(limit):
            report = self.step()
            if report.converged:
                break
        return tuple(self.reports)

    def status(self) -> dict[str, object]:
        self._require_loaded()
        spatial, latent = self._errors()
        return {
            "operator": self.OPERATOR,
            "equation": self.EQUATION,
            "iteration": self._iteration,
            "agents": len(self._positions),
            "logical_domain": f"{self.config.scale}^3",
            "materialization": "sparse-active-agents-only",
            "center": self.config.center,
            "spatial_l1": spatial,
            "latent_hamming": latent,
            "objective": self._objective(spatial, latent),
            "theoretical_iterations_remaining": self.theoretical_iterations_remaining(),
            "converged": spatial == 0 and latent == 0,
            "state_hash": self._state_hash(),
        }

    def _state_hash(self) -> str:
        canonical = {
            "positions": self._positions,
            "states": self._states,
            "targets": self._targets,
        }
        payload = json.dumps(canonical, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _require_loaded(self) -> None:
        if not self._loaded:
            raise RuntimeError("load a finite swarm before inward execution")
