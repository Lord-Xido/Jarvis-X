"""Epistemic admission layer for the Dr Moagi 3D Codex.

The core invariant is deliberately stronger than numerical boundedness:

    hypotheses may recurse, but they cannot become authoritative observations
    without independent external verification.

This module wraps the pure Layer-5 Codex executor with provenance-aware input
classification, immutable run anchoring, evidence consensus, fail-closed
commit semantics, verified-only learning, and verified-only permeation release.
It does not mutate the canonical Jarvis-X VM.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence

from .dr_moagi_codex import (
    Coordinate,
    DrMoagiCodexResult,
    Latent,
    ScalarField,
    SceneLike,
    update_parameters,
)


class EvidenceKind(str, Enum):
    SENSOR = "sensor"
    INSTRUMENT = "instrument"
    RETRIEVAL = "retrieval"
    USER = "user"
    MODEL = "model"
    SIMULATION = "simulation"


DEFAULT_EXTERNAL_KINDS = frozenset(
    {
        EvidenceKind.SENSOR,
        EvidenceKind.INSTRUMENT,
        EvidenceKind.RETRIEVAL,
        EvidenceKind.USER,
    }
)


@dataclass(frozen=True)
class ObservationPacket:
    """Externally sourced observation presented to the epistemic boundary."""

    scene: SceneLike
    source_id: str
    kind: EvidenceKind
    confidence: float = 1.0
    derived_from_hypothesis: bool = False


@dataclass(frozen=True)
class EvidencePacket:
    """Independent evidence used to verify a generated hypothesis."""

    scene: SceneLike
    source_id: str
    kind: EvidenceKind
    confidence: float = 1.0
    independence_key: str | None = None
    derived_from_hypothesis: bool = False


@dataclass(frozen=True)
class ObservationAnchor:
    scene: ScalarField
    source_id: str
    digest: str


@dataclass(frozen=True)
class EpistemicGateConfig:
    """Fail-closed thresholds for promotion from hypothesis to committed state."""

    max_observation_nrmse: float = 0.10
    max_anchor_nrmse: float | None = 0.25
    max_evidence_nrmse: float = 0.10
    min_independent_evidence: int = 1
    min_source_confidence: float = 0.75
    require_exact_support: bool = True
    allowed_external_kinds: frozenset[EvidenceKind] = DEFAULT_EXTERNAL_KINDS

    def __post_init__(self) -> None:
        for name in ("max_observation_nrmse", "max_evidence_nrmse"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.max_anchor_nrmse is not None:
            value = self.max_anchor_nrmse
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError("max_anchor_nrmse must be numeric or None")
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError("max_anchor_nrmse must be finite and non-negative")
        if (
            isinstance(self.min_independent_evidence, bool)
            or not isinstance(self.min_independent_evidence, int)
            or self.min_independent_evidence < 0
        ):
            raise ValueError("min_independent_evidence must be a non-negative integer")
        if (
            isinstance(self.min_source_confidence, bool)
            or not isinstance(self.min_source_confidence, (int, float))
            or not math.isfinite(float(self.min_source_confidence))
            or not 0.0 <= float(self.min_source_confidence) <= 1.0
        ):
            raise ValueError("min_source_confidence must lie in [0, 1]")
        if not self.allowed_external_kinds:
            raise ValueError("allowed_external_kinds must not be empty")
        if EvidenceKind.MODEL in self.allowed_external_kinds:
            raise ValueError("model output cannot be configured as independent external evidence")


@dataclass(frozen=True)
class EpistemicVerdict:
    admitted: bool
    reasons: tuple[str, ...]
    observation_nrmse: float
    anchor_nrmse: float | None
    max_evidence_nrmse: float | None
    independent_evidence_count: int
    observation_digest: str
    anchor_digest: str
    evidence_digests: tuple[str, ...]


@dataclass(frozen=True)
class EpistemicExecutionResult:
    """Authoritative outputs are populated only when the gate admits them."""

    verdict: EpistemicVerdict
    hypothesis_scene: ScalarField
    hypothesis_latent: Latent
    committed_scene: ScalarField | None
    theta_before: Latent | None
    theta_after: Latent | None
    learning_committed: bool
    released_source_charge: ScalarField
    released_permeation_field: dict[Coordinate, complex]
    fixed_point_iterations: int
    fixed_point_converged: bool
    virtual_depth_label: str


class CodexCandidateExecutor(Protocol):
    config: Any

    def execute(self, scene: SceneLike, **kwargs: Any) -> DrMoagiCodexResult: ...


def _field(name: str, scene: SceneLike) -> ScalarField:
    result: ScalarField = {}
    for raw_coordinate, raw_value in scene.items():
        if not isinstance(raw_coordinate, tuple) or len(raw_coordinate) != 3:
            raise TypeError(f"{name} coordinates must be 3-tuples")
        coordinate = tuple(float(axis) for axis in raw_coordinate)
        if not all(math.isfinite(axis) for axis in coordinate):
            raise ValueError(f"{name} coordinates must be finite")
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            raise TypeError(f"{name} values must be numeric")
        value = float(raw_value)
        if not math.isfinite(value):
            raise ValueError(f"{name} contains a non-finite value")
        result[coordinate] = value
    if not result:
        raise ValueError(f"{name} must not be empty")
    return result


def scene_digest(scene: SceneLike) -> str:
    field = _field("scene", scene)
    canonical = [
        [coordinate[0], coordinate[1], coordinate[2], field[coordinate]]
        for coordinate in sorted(field)
    ]
    encoded = json.dumps(canonical, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def scene_nrmse(
    candidate: SceneLike,
    reference: SceneLike,
    *,
    require_exact_support: bool = True,
) -> float:
    hypothesis = _field("candidate scene", candidate)
    truth = _field("reference scene", reference)
    if require_exact_support and set(hypothesis) != set(truth):
        return math.inf
    support = set(hypothesis) | set(truth)
    mse = sum((hypothesis.get(point, 0.0) - truth.get(point, 0.0)) ** 2 for point in support)
    mse /= len(support)
    reference_power = sum(truth.get(point, 0.0) ** 2 for point in support) / len(support)
    scale = max(math.sqrt(reference_power), 1.0e-12)
    return math.sqrt(mse) / scale


def _packet_is_external(
    *,
    kind: EvidenceKind,
    confidence: float,
    derived_from_hypothesis: bool,
    config: EpistemicGateConfig,
) -> bool:
    return (
        kind in config.allowed_external_kinds
        and kind is not EvidenceKind.MODEL
        and not derived_from_hypothesis
        and confidence >= config.min_source_confidence
    )


def evaluate_epistemic_gate(
    hypothesis: SceneLike,
    observation: ObservationPacket,
    evidence: Sequence[EvidencePacket],
    *,
    anchor: ObservationAnchor,
    config: EpistemicGateConfig,
) -> EpistemicVerdict:
    candidate = _field("hypothesis scene", hypothesis)
    observed = _field("observation scene", observation.scene)
    reasons: list[str] = []

    observation_error = scene_nrmse(
        candidate,
        observed,
        require_exact_support=config.require_exact_support,
    )
    if observation_error > config.max_observation_nrmse:
        reasons.append("hypothesis exceeds observation error threshold")

    anchor_error: float | None = None
    if config.max_anchor_nrmse is not None:
        anchor_error = scene_nrmse(
            candidate,
            anchor.scene,
            require_exact_support=config.require_exact_support,
        )
        if anchor_error > config.max_anchor_nrmse:
            reasons.append("hypothesis exceeds immutable-anchor drift threshold")

    evidence_errors: list[float] = []
    evidence_digests: list[str] = []
    independence_keys: set[str] = set()
    source_ids: set[str] = set()
    for packet in evidence:
        if not packet.source_id:
            reasons.append("evidence source_id must not be empty")
            continue
        if packet.source_id in source_ids:
            reasons.append("duplicate evidence source_id")
            continue
        source_ids.add(packet.source_id)
        if not _packet_is_external(
            kind=packet.kind,
            confidence=packet.confidence,
            derived_from_hypothesis=packet.derived_from_hypothesis,
            config=config,
        ):
            reasons.append(f"inadmissible evidence source: {packet.source_id}")
            continue
        key = packet.independence_key or packet.source_id
        independence_keys.add(key)
        packet_scene = _field(f"evidence[{packet.source_id}]", packet.scene)
        evidence_digests.append(scene_digest(packet_scene))
        error = scene_nrmse(
            candidate,
            packet_scene,
            require_exact_support=config.require_exact_support,
        )
        evidence_errors.append(error)
        if error > config.max_evidence_nrmse:
            reasons.append(f"hypothesis conflicts with evidence: {packet.source_id}")

    independent_count = len(independence_keys)
    if independent_count < config.min_independent_evidence:
        reasons.append("insufficient independent external evidence")

    return EpistemicVerdict(
        admitted=not reasons,
        reasons=tuple(reasons),
        observation_nrmse=observation_error,
        anchor_nrmse=anchor_error,
        max_evidence_nrmse=max(evidence_errors) if evidence_errors else None,
        independent_evidence_count=independent_count,
        observation_digest=scene_digest(observed),
        anchor_digest=anchor.digest,
        evidence_digests=tuple(evidence_digests),
    )


class EpistemicDrMoagiCodex:
    """Candidate-first Codex wrapper with an explicit epistemic commit gate."""

    def __init__(
        self,
        codex: CodexCandidateExecutor,
        config: EpistemicGateConfig | None = None,
    ) -> None:
        self.codex = codex
        self.config = config or EpistemicGateConfig()
        self._anchor: ObservationAnchor | None = None

    @property
    def anchor(self) -> ObservationAnchor | None:
        return self._anchor

    def begin_run(self, observation: ObservationPacket) -> ObservationAnchor:
        if self._anchor is not None:
            raise RuntimeError("run anchor is immutable; call reset_run before beginning another run")
        self._validate_observation(observation)
        scene = _field("observation scene", observation.scene)
        self._anchor = ObservationAnchor(
            scene=scene,
            source_id=observation.source_id,
            digest=scene_digest(scene),
        )
        return self._anchor

    def reset_run(self) -> None:
        self._anchor = None

    def _validate_observation(self, observation: ObservationPacket) -> None:
        if not observation.source_id:
            raise ValueError("observation source_id must not be empty")
        if not _packet_is_external(
            kind=observation.kind,
            confidence=observation.confidence,
            derived_from_hypothesis=observation.derived_from_hypothesis,
            config=self.config,
        ):
            raise ValueError("observation is not admissible independent external input")
        _field("observation scene", observation.scene)

    def execute(
        self,
        observation: ObservationPacket,
        *,
        evidence: Sequence[EvidencePacket],
        theta: Sequence[float] | None = None,
        theta_gradient: Sequence[float] | None = None,
        **codex_kwargs: Any,
    ) -> EpistemicExecutionResult:
        self._validate_observation(observation)
        if self._anchor is None:
            self.begin_run(observation)
        assert self._anchor is not None

        if "theta" in codex_kwargs or "theta_gradient" in codex_kwargs:
            raise ValueError("theta updates are controlled by the epistemic wrapper")
        if (theta is None) != (theta_gradient is None):
            raise ValueError("theta and theta_gradient must be supplied together")

        # Generate a hypothesis only. The base Codex remains pure, but its
        # candidate learning output is deliberately disabled at this stage.
        candidate = self.codex.execute(
            observation.scene,
            theta=None,
            theta_gradient=None,
            **codex_kwargs,
        )
        verdict = evaluate_epistemic_gate(
            candidate.decoded_scene,
            observation,
            evidence,
            anchor=self._anchor,
            config=self.config,
        )

        theta_before = tuple(float(value) for value in theta) if theta is not None else None
        theta_after = theta_before
        learning_committed = False
        if verdict.admitted and theta_before is not None and theta_gradient is not None:
            eta_theta = float(getattr(self.codex.config, "eta_theta"))
            theta_after = update_parameters(theta_before, theta_gradient, eta_theta=eta_theta)
            learning_committed = True

        if verdict.admitted:
            committed_scene: ScalarField | None = dict(candidate.decoded_scene)
            released_source = dict(candidate.source_charge)
            released_phi = dict(candidate.permeation_field)
        else:
            committed_scene = None
            released_source = {}
            released_phi = {}

        return EpistemicExecutionResult(
            verdict=verdict,
            hypothesis_scene=dict(candidate.decoded_scene),
            hypothesis_latent=tuple(candidate.projected_latent),
            committed_scene=committed_scene,
            theta_before=theta_before,
            theta_after=theta_after,
            learning_committed=learning_committed,
            released_source_charge=released_source,
            released_permeation_field=released_phi,
            fixed_point_iterations=candidate.fixed_point.iterations,
            fixed_point_converged=candidate.fixed_point.converged,
            virtual_depth_label=candidate.virtual_depth_label,
        )