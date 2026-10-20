"""Suspensory consideration algebra over bounded sparse 3D arithmetic states.

A proposed idea can be evaluated and iteratively refined without becoming an
executable decision.  ``consider`` mutates only the proposal ledger, never the
authoritative state. ``commit`` requires an explicit gate, the candidate digest,
and the revision from which that candidate was derived.

This is a deterministic research control-plane abstraction. The SHA-256 digests
are integrity checks, not digital signatures, evidence of truth, or permission
to perform external side effects.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Mapping

Coordinate = tuple[int, int, int]
SparseField = dict[Coordinate, float]


def _hash_field(field: Mapping[Coordinate, float]) -> str:
    """Canonical digest for a finite coordinate/value map."""
    records = [[list(coordinate), value] for coordinate, value in sorted(field.items())]
    payload = json.dumps(records, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _finite_number(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


@dataclass(frozen=True)
class AffineHypothesis:
    """f(v) = scale*v + bias, evaluated only on materialized 3D support."""

    name: str
    scale: float
    bias: float

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("hypothesis name must be non-empty")
        object.__setattr__(self, "scale", _finite_number(self.scale, "scale"))
        object.__setattr__(self, "bias", _finite_number(self.bias, "bias"))


@dataclass(frozen=True)
class ConsideredIdea:
    """Immutable hypothetical outcome, separate from authoritative state."""

    idea_id: str
    parent_id: str | None
    base_revision: int
    base_state_hash: str
    hypothesis: AffineHypothesis
    candidate: tuple[tuple[Coordinate, float], ...]
    candidate_hash: str
    delta_l2: float
    sum_before: float
    sum_after: float

    def snapshot(self) -> SparseField:
        return dict(self.candidate)


@dataclass(frozen=True)
class Decision:
    idea_id: str
    committed: bool
    reason: str
    previous_revision: int
    current_revision: int
    state_hash: str


class SuspensoryAlgebra:
    """Consider branches of arithmetic ideas without applying their outcomes.

    The state is a finite sparse field of 3D coordinates. No dense 3D tensor is
    allocated. Logical domain size never implies available physical memory.
    """

    def __init__(
        self,
        initial: Mapping[Coordinate, float],
        *,
        side: int = 100,
        max_active_cells: int = 100_000,
        max_candidates: int = 128,
        max_abs_value: float = 1.0e9,
    ) -> None:
        for name, integer in (
            ("side", side),
            ("max_active_cells", max_active_cells),
            ("max_candidates", max_candidates),
        ):
            if isinstance(integer, bool) or not isinstance(integer, int) or integer <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.side = side
        self.max_active_cells = max_active_cells
        self.max_candidates = max_candidates
        self.max_abs_value = _finite_number(max_abs_value, "max_abs_value")
        if self.max_abs_value <= 0.0:
            raise ValueError("max_abs_value must be positive")
        self._state = self._validate(initial)
        self._revision = 0
        self._ideas: dict[str, ConsideredIdea] = {}

    def _validate(self, source: Mapping[Coordinate, float]) -> SparseField:
        if len(source) > self.max_active_cells:
            raise ValueError("active-cell budget exceeded")
        result: SparseField = {}
        for coordinate, raw in source.items():
            if (
                not isinstance(coordinate, tuple)
                or len(coordinate) != 3
                or any(isinstance(axis, bool) or not isinstance(axis, int) for axis in coordinate)
                or any(axis < 0 or axis >= self.side for axis in coordinate)
            ):
                raise ValueError("coordinates must be integer triples within the 3D domain")
            value = _finite_number(raw, "field value")
            if abs(value) > self.max_abs_value:
                raise ValueError("field value exceeds configured bound")
            result[coordinate] = value
        return result

    @property
    def revision(self) -> int:
        return self._revision

    @property
    def state_hash(self) -> str:
        return _hash_field(self._state)

    def snapshot(self) -> SparseField:
        """Return a copy of the authoritative state."""
        return dict(self._state)

    def idea(self, idea_id: str) -> ConsideredIdea:
        return self._ideas[idea_id]

    def consider(
        self,
        idea_id: str,
        hypothesis: AffineHypothesis,
        *,
        parent_id: str | None = None,
    ) -> ConsideredIdea:
        """Evaluate a hypothesis on a shadow state; never modify the real state.

        A ``parent_id`` composes with a prior hypothetical result, not the
        authoritative field. Branches remain bound to their starting revision.
        """
        if not isinstance(idea_id, str) or not idea_id.strip():
            raise ValueError("idea_id must be non-empty")
        if idea_id in self._ideas:
            raise ValueError("duplicate idea_id")
        if len(self._ideas) >= self.max_candidates:
            raise ValueError("candidate budget exceeded")
        if not isinstance(hypothesis, AffineHypothesis):
            raise TypeError("hypothesis must be an AffineHypothesis")
        if parent_id is None:
            source = self.snapshot()
            base_revision = self._revision
            base_hash = self.state_hash
        else:
            parent = self._ideas[parent_id]
            if parent.base_revision != self._revision or parent.base_state_hash != self.state_hash:
                raise ValueError("cannot branch from a stale candidate")
            source = parent.snapshot()
            base_revision = parent.base_revision
            base_hash = parent.base_state_hash

        shadow = {
            coordinate: hypothesis.scale * value + hypothesis.bias
            for coordinate, value in sorted(source.items())
        }
        shadow = self._validate(shadow)
        delta = [shadow[key] - source[key] for key in source]
        # hypot() avoids overflow while accumulating the L2 norm; reject infinity.
        norm = math.hypot(*delta)
        if not math.isfinite(norm):
            raise ValueError("hypothesis produced an unrepresentable displacement")
        receipt = ConsideredIdea(
            idea_id=idea_id,
            parent_id=parent_id,
            base_revision=base_revision,
            base_state_hash=base_hash,
            hypothesis=hypothesis,
            candidate=tuple(sorted(shadow.items())),
            candidate_hash=_hash_field(shadow),
            delta_l2=norm,
            sum_before=math.fsum(source.values()),
            sum_after=math.fsum(shadow.values()),
        )
        self._ideas[idea_id] = receipt
        return receipt

    def commit(
        self,
        idea_id: str,
        *,
        authorization: bool = False,
        expected_hash: str | None = None,
    ) -> Decision:
        """Commit a reviewed shadow state into this in-memory instance only.

        An unauthorized, mismatched, or stale proposal cannot change the state.
        This method has no external IO or callback execution capabilities.
        """
        idea = self._ideas[idea_id]
        before = self._revision
        if not authorization:
            reason = "execution_gate_closed"
        elif expected_hash != idea.candidate_hash:
            reason = "candidate_hash_mismatch"
        elif idea.base_revision != before or idea.base_state_hash != self.state_hash:
            reason = "stale_base_state"
        elif _hash_field(idea.snapshot()) != idea.candidate_hash:
            reason = "candidate_integrity_failed"
        else:
            # Recheck bounds at the authority boundary.
            self._state = self._validate(idea.snapshot())
            self._revision += 1
            return Decision(idea_id, True, "committed", before, self._revision, self.state_hash)
        return Decision(idea_id, False, reason, before, self._revision, self.state_hash)
