"""Multimodal evidence retrieval and active verification runtime.

This module turns retrieval from a one-shot text lookup into a bounded,
multimodal evidence-acquisition loop:

query/state
    -> candidate retrieval across modalities
    -> temporal/spatial localization
    -> shared evidence embedding
    -> 3D latent contraction
    -> CTR verification
    -> uncertainty estimate
    -> targeted re-retrieval when evidence is insufficient
    -> verified evidence bundle

The reference implementation is deterministic and dependency-free. It does not
decode arbitrary media files by itself; production audio/video/image/3D/code
adapters can populate EvidenceSegment objects with native features and precise
regions. That boundary is explicit so the core does not pretend metadata is the
same thing as media access.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable, Sequence


class Modality(str, Enum):
    TEXT = "text"
    AUDIO = "audio"
    VIDEO = "video"
    IMAGE = "image"
    VOLUME3D = "3d"
    CODE = "code"
    SENSOR = "sensor"
    STRUCTURED = "structured"


@dataclass(frozen=True)
class TimeSpan:
    start_seconds: float
    end_seconds: float

    def validate(self) -> None:
        if self.start_seconds < 0.0:
            raise ValueError("start_seconds must be non-negative")
        if self.end_seconds < self.start_seconds:
            raise ValueError("end_seconds must be >= start_seconds")


@dataclass(frozen=True)
class Region2D:
    x0: float
    y0: float
    x1: float
    y1: float

    def validate(self) -> None:
        values = (self.x0, self.y0, self.x1, self.y1)
        if not all(0.0 <= value <= 1.0 for value in values):
            raise ValueError("2D region coordinates must be normalized to [0, 1]")
        if self.x1 < self.x0 or self.y1 < self.y0:
            raise ValueError("2D region max bounds must be >= min bounds")


@dataclass(frozen=True)
class Region3D:
    x0: float
    y0: float
    z0: float
    x1: float
    y1: float
    z1: float

    def validate(self) -> None:
        values = (self.x0, self.y0, self.z0, self.x1, self.y1, self.z1)
        if not all(0.0 <= value <= 1.0 for value in values):
            raise ValueError("3D region coordinates must be normalized to [0, 1]")
        if self.x1 < self.x0 or self.y1 < self.y0 or self.z1 < self.z0:
            raise ValueError("3D region max bounds must be >= min bounds")


@dataclass(frozen=True)
class EvidenceSegment:
    evidence_id: str
    object_id: str
    modality: Modality
    summary: str
    provenance: str
    confidence: float = 1.0
    time_span: TimeSpan | None = None
    region_2d: Region2D | None = None
    region_3d: Region3D | None = None
    tags: tuple[str, ...] = ()
    native_feature: tuple[float, ...] = ()

    def validate(self) -> None:
        if not self.evidence_id:
            raise ValueError("evidence_id is required")
        if not self.object_id:
            raise ValueError("object_id is required")
        if not self.provenance:
            raise ValueError("provenance is required")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if self.time_span is not None:
            self.time_span.validate()
        if self.region_2d is not None:
            self.region_2d.validate()
        if self.region_3d is not None:
            self.region_3d.validate()


@dataclass(frozen=True)
class RetrievalQuery:
    text: str
    top_k: int = 5
    required_modalities: tuple[Modality, ...] = ()
    preferred_modalities: tuple[Modality, ...] = ()
    uncertainty_threshold: float = 0.30
    max_rounds: int = 3

    def validate(self) -> None:
        if not self.text.strip():
            raise ValueError("query text is required")
        if self.top_k < 1:
            raise ValueError("top_k must be >= 1")
        if not 0.0 <= self.uncertainty_threshold <= 1.0:
            raise ValueError("uncertainty_threshold must be in [0, 1]")
        if self.max_rounds < 1:
            raise ValueError("max_rounds must be >= 1")


@dataclass(frozen=True)
class RetrievedEvidence:
    segment: EvidenceSegment
    similarity: float
    score: float


@dataclass(frozen=True)
class CTRReceipt:
    generated: int
    contrasted: int
    reckoned: int
    verified: int
    corrected: int
    modality_coverage: float
    provenance_diversity: float
    agreement: float
    uncertainty: float
    accepted: bool


@dataclass(frozen=True)
class RetrievalRound:
    round_index: int
    target_modalities: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    latent_initial: tuple[float, float, float]
    latent_fixed_point: tuple[float, float, float]
    fixed_point_iterations: int
    fixed_point_residual: float
    ctr: CTRReceipt


@dataclass(frozen=True)
class RetrievalResult:
    query: str
    evidence: tuple[RetrievedEvidence, ...]
    rounds: tuple[RetrievalRound, ...]
    latent: tuple[float, float, float]
    uncertainty: float
    accepted: bool
    unresolved_modalities: tuple[str, ...]


class SharedHasher:
    """Deterministic shared embedding used by the reference runtime."""

    def __init__(self, dimensions: int = 64) -> None:
        if dimensions < 8:
            raise ValueError("dimensions must be >= 8")
        self.dimensions = dimensions

    @staticmethod
    def _tokens(text: str) -> tuple[str, ...]:
        clean = "".join(ch.lower() if ch.isalnum() else " " for ch in text)
        return tuple(token for token in clean.split() if token)

    def embed_text(self, text: str) -> tuple[float, ...]:
        vector = [0.0] * self.dimensions
        tokens = self._tokens(text)
        if not tokens:
            return tuple(vector)

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm <= 1.0e-12:
            return tuple(vector)
        return tuple(value / norm for value in vector)

    def embed_segment(self, segment: EvidenceSegment) -> tuple[float, ...]:
        text = " ".join(
            (
                segment.modality.value,
                segment.summary,
                " ".join(segment.tags),
                segment.object_id,
            )
        )
        base = list(self.embed_text(text))

        for index, value in enumerate(segment.native_feature[: self.dimensions]):
            base[index] += 0.35 * float(value)

        norm = math.sqrt(sum(value * value for value in base))
        if norm <= 1.0e-12:
            return tuple(base)
        return tuple(value / norm for value in base)


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError("embedding dimensions do not match")
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na <= 1.0e-12 or nb <= 1.0e-12:
        return 0.0
    return max(-1.0, min(1.0, dot / (na * nb)))


class EvidenceStore:
    """In-memory evidence index for deterministic tests and reference execution."""

    def __init__(self, segments: Iterable[EvidenceSegment] = ()) -> None:
        self._segments: dict[str, EvidenceSegment] = {}
        for segment in segments:
            self.add(segment)

    def add(self, segment: EvidenceSegment) -> None:
        segment.validate()
        self._segments[segment.evidence_id] = segment

    def segments(self) -> tuple[EvidenceSegment, ...]:
        return tuple(self._segments[key] for key in sorted(self._segments))


class MultimodalRetriever:
    def __init__(self, store: EvidenceStore, hasher: SharedHasher | None = None) -> None:
        self.store = store
        self.hasher = hasher or SharedHasher()
        self._embedding_cache: dict[str, tuple[float, ...]] = {}

    def _embedding(self, segment: EvidenceSegment) -> tuple[float, ...]:
        if segment.evidence_id not in self._embedding_cache:
            self._embedding_cache[segment.evidence_id] = self.hasher.embed_segment(segment)
        return self._embedding_cache[segment.evidence_id]

    def retrieve(
        self,
        query: RetrievalQuery,
        modalities: Sequence[Modality] | None = None,
    ) -> tuple[RetrievedEvidence, ...]:
        query.validate()
        query_vector = self.hasher.embed_text(query.text)
        allowed = set(modalities) if modalities else None
        preferred = set(query.preferred_modalities)

        rows: list[RetrievedEvidence] = []
        for segment in self.store.segments():
            if allowed is not None and segment.modality not in allowed:
                continue
            similarity = cosine(query_vector, self._embedding(segment))
            normalized_similarity = 0.5 * (similarity + 1.0)
            modality_bonus = 0.08 if segment.modality in preferred else 0.0
            score = (
                0.72 * normalized_similarity
                + 0.20 * segment.confidence
                + modality_bonus
            )
            rows.append(
                RetrievedEvidence(
                    segment=segment,
                    similarity=similarity,
                    score=max(0.0, min(1.0, score)),
                )
            )

        rows.sort(key=lambda row: (-row.score, row.segment.evidence_id))
        return tuple(rows[: query.top_k])


def evidence_to_latent(
    query: RetrievalQuery,
    evidence: Sequence[RetrievedEvidence],
    hasher: SharedHasher,
) -> tuple[float, float, float]:
    query_vector = hasher.embed_text(query.text)
    if not evidence:
        return (0.0, 0.0, 0.0)

    accumulator = [0.0, 0.0, 0.0]
    weight_total = 0.0

    for row in evidence:
        vector = hasher.embed_segment(row.segment)
        weight = max(row.score, 1.0e-6)
        for axis in range(3):
            q_component = query_vector[axis]
            e_component = vector[axis]
            accumulator[axis] += weight * (0.45 * q_component + 0.55 * e_component)
        weight_total += weight

    return tuple(value / weight_total for value in accumulator)  # type: ignore[return-value]


def latent_echo(
    latent: tuple[float, float, float],
    evidence: Sequence[RetrievedEvidence],
    hasher: SharedHasher,
) -> tuple[float, float, float]:
    if not evidence:
        return latent

    target = [0.0, 0.0, 0.0]
    total = 0.0
    for row in evidence:
        vector = hasher.embed_segment(row.segment)
        weight = max(row.score * row.segment.confidence, 1.0e-6)
        for axis in range(3):
            target[axis] += weight * vector[axis]
        total += weight

    target = [value / total for value in target]
    return tuple(
        math.tanh(0.62 * latent[axis] + 0.38 * target[axis])
        for axis in range(3)
    )  # type: ignore[return-value]


def inward_fixed_point(
    initial: tuple[float, float, float],
    evidence: Sequence[RetrievedEvidence],
    hasher: SharedHasher,
    *,
    damping: float = 0.65,
    tolerance: float = 1.0e-6,
    max_iterations: int = 32,
) -> tuple[tuple[float, float, float], int, float]:
    latent = initial
    residual = float("inf")

    for iteration in range(1, max_iterations + 1):
        echo = latent_echo(latent, evidence, hasher)
        candidate = (
            (1.0 - damping) * latent[0] + damping * echo[0],
            (1.0 - damping) * latent[1] + damping * echo[1],
            (1.0 - damping) * latent[2] + damping * echo[2],
        )
        residual = math.sqrt(
            sum((candidate[axis] - latent[axis]) ** 2 for axis in range(3))
        )
        latent = candidate
        if residual <= tolerance:
            return latent, iteration, residual

    return latent, max_iterations, residual


def _agreement(evidence: Sequence[RetrievedEvidence]) -> float:
    if len(evidence) < 2:
        return 0.5 if evidence else 0.0

    values = [row.score for row in evidence]
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return max(0.0, min(1.0, 1.0 - math.sqrt(variance)))


def ctr_verify(
    query: RetrievalQuery,
    evidence: Sequence[RetrievedEvidence],
) -> CTRReceipt:
    modalities = {row.segment.modality for row in evidence}
    required = set(query.required_modalities)

    if required:
        coverage = len(modalities & required) / len(required)
    else:
        coverage = min(1.0, len(modalities) / 3.0)

    provenances = {row.segment.provenance for row in evidence}
    provenance_diversity = min(1.0, len(provenances) / 3.0)
    agreement = _agreement(evidence)

    if evidence:
        mean_confidence = sum(
            row.segment.confidence * row.score for row in evidence
        ) / len(evidence)
    else:
        mean_confidence = 0.0

    evidence_strength = (
        0.34 * coverage
        + 0.22 * provenance_diversity
        + 0.22 * agreement
        + 0.22 * mean_confidence
    )
    uncertainty = max(0.0, min(1.0, 1.0 - evidence_strength))
    accepted = uncertainty <= query.uncertainty_threshold and coverage >= 1.0

    generated = len(evidence)
    contrasted = generated
    reckoned = sum(row.score >= 0.35 for row in evidence)
    verified = sum(
        row.score >= 0.45 and row.segment.confidence >= 0.50 for row in evidence
    )
    corrected = 0 if accepted else 1

    return CTRReceipt(
        generated=generated,
        contrasted=contrasted,
        reckoned=reckoned,
        verified=verified,
        corrected=corrected,
        modality_coverage=coverage,
        provenance_diversity=provenance_diversity,
        agreement=agreement,
        uncertainty=uncertainty,
        accepted=accepted,
    )


class ActiveMultimodalEvidenceEngine:
    """Closed-loop multimodal retrieval with uncertainty-triggered reacquisition."""

    def __init__(self, retriever: MultimodalRetriever) -> None:
        self.retriever = retriever

    @staticmethod
    def _target_modalities(
        query: RetrievalQuery,
        accumulated: Sequence[RetrievedEvidence],
        round_index: int,
    ) -> tuple[Modality, ...]:
        covered = {row.segment.modality for row in accumulated}
        required_missing = [
            modality
            for modality in query.required_modalities
            if modality not in covered
        ]
        if required_missing:
            return tuple(required_missing)

        if round_index == 0 and query.preferred_modalities:
            return query.preferred_modalities

        expansion_order = (
            Modality.TEXT,
            Modality.AUDIO,
            Modality.VIDEO,
            Modality.IMAGE,
            Modality.CODE,
            Modality.VOLUME3D,
            Modality.SENSOR,
            Modality.STRUCTURED,
        )
        unseen = [modality for modality in expansion_order if modality not in covered]
        return tuple(unseen[:3])

    @staticmethod
    def _merge(
        accumulated: Sequence[RetrievedEvidence],
        incoming: Sequence[RetrievedEvidence],
    ) -> tuple[RetrievedEvidence, ...]:
        by_id = {row.segment.evidence_id: row for row in accumulated}
        for row in incoming:
            previous = by_id.get(row.segment.evidence_id)
            if previous is None or row.score > previous.score:
                by_id[row.segment.evidence_id] = row
        return tuple(
            sorted(by_id.values(), key=lambda row: (-row.score, row.segment.evidence_id))
        )

    def run(self, query: RetrievalQuery) -> RetrievalResult:
        query.validate()
        accumulated: tuple[RetrievedEvidence, ...] = ()
        rounds: list[RetrievalRound] = []
        latent = (0.0, 0.0, 0.0)
        last_ctr = ctr_verify(query, accumulated)

        for round_index in range(query.max_rounds):
            targets = self._target_modalities(query, accumulated, round_index)
            incoming = self.retriever.retrieve(
                query,
                targets if targets else None,
            )
            accumulated = self._merge(accumulated, incoming)

            initial = evidence_to_latent(
                query,
                accumulated,
                self.retriever.hasher,
            )
            latent, iterations, residual = inward_fixed_point(
                initial,
                accumulated,
                self.retriever.hasher,
            )
            last_ctr = ctr_verify(query, accumulated)

            rounds.append(
                RetrievalRound(
                    round_index=round_index,
                    target_modalities=tuple(modality.value for modality in targets),
                    evidence_ids=tuple(row.segment.evidence_id for row in accumulated),
                    latent_initial=initial,
                    latent_fixed_point=latent,
                    fixed_point_iterations=iterations,
                    fixed_point_residual=residual,
                    ctr=last_ctr,
                )
            )

            if last_ctr.accepted:
                break

        covered = {row.segment.modality for row in accumulated}
        unresolved = tuple(
            modality.value
            for modality in query.required_modalities
            if modality not in covered
        )

        return RetrievalResult(
            query=query.text,
            evidence=accumulated,
            rounds=tuple(rounds),
            latent=latent,
            uncertainty=last_ctr.uncertainty,
            accepted=last_ctr.accepted,
            unresolved_modalities=unresolved,
        )


def result_to_json(result: RetrievalResult) -> dict[str, object]:
    return {
        "schema_version": "jarvisx.multimodal-evidence-runtime.v1",
        "provenance": "simulated-reference-runtime",
        "query": result.query,
        "accepted": result.accepted,
        "uncertainty": result.uncertainty,
        "unresolved_modalities": list(result.unresolved_modalities),
        "latent": list(result.latent),
        "evidence": [
            {
                "segment": {
                    "evidence_id": row.segment.evidence_id,
                    "object_id": row.segment.object_id,
                    "modality": row.segment.modality.value,
                    "summary": row.segment.summary,
                    "provenance": row.segment.provenance,
                    "confidence": row.segment.confidence,
                    "time_span": (
                        asdict(row.segment.time_span)
                        if row.segment.time_span is not None
                        else None
                    ),
                    "region_2d": (
                        asdict(row.segment.region_2d)
                        if row.segment.region_2d is not None
                        else None
                    ),
                    "region_3d": (
                        asdict(row.segment.region_3d)
                        if row.segment.region_3d is not None
                        else None
                    ),
                    "tags": list(row.segment.tags),
                    "native_feature": list(row.segment.native_feature),
                },
                "similarity": row.similarity,
                "score": row.score,
            }
            for row in result.evidence
        ],
        "rounds": [asdict(row) for row in result.rounds],
        "claim_boundary": {
            "implemented": (
                "multimodal evidence objects, temporal/spatial localization metadata, "
                "shared deterministic retrieval, 3D latent fixed-point refinement, "
                "CTR verification, provenance tracking, and uncertainty-triggered "
                "re-retrieval"
            ),
            "adapter_boundary": (
                "production media decoding, ASR, OCR, vision, video understanding, "
                "3D perception, and external search connectors remain pluggable native "
                "adapters that populate EvidenceSegment instances"
            ),
        },
    }


def demo_store() -> EvidenceStore:
    return EvidenceStore(
        (
            EvidenceSegment(
                evidence_id="text-1",
                object_id="track-42",
                modality=Modality.TEXT,
                summary="catalog metadata for a song about returning home",
                provenance="catalog",
                confidence=0.72,
                tags=("song", "home", "partner"),
            ),
            EvidenceSegment(
                evidence_id="audio-1",
                object_id="track-42",
                modality=Modality.AUDIO,
                summary="vocal segment where a partner asks the singer to come home",
                provenance="audio-index",
                confidence=0.93,
                time_span=TimeSpan(41.2, 52.8),
                tags=("vocals", "partner", "come home"),
                native_feature=(0.4, 0.7, 0.3),
            ),
            EvidenceSegment(
                evidence_id="video-1",
                object_id="track-42",
                modality=Modality.VIDEO,
                summary="music video segment aligned to the home-request lyric",
                provenance="video-index",
                confidence=0.84,
                time_span=TimeSpan(40.8, 53.1),
                region_2d=Region2D(0.1, 0.1, 0.9, 0.9),
                tags=("music video", "home"),
                native_feature=(0.3, 0.6, 0.2),
            ),
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Jarvis-X multimodal active evidence reference runtime."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/multimodal-evidence-runtime.json"),
    )
    args = parser.parse_args()

    engine = ActiveMultimodalEvidenceEngine(MultimodalRetriever(demo_store()))
    query = RetrievalQuery(
        text="song where his partner asks him to come home",
        top_k=3,
        required_modalities=(Modality.TEXT, Modality.AUDIO),
        preferred_modalities=(Modality.AUDIO, Modality.VIDEO),
        uncertainty_threshold=0.45,
        max_rounds=3,
    )
    result = engine.run(query)
    payload = result_to_json(result)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    print("Jarvis-X multimodal evidence runtime")
    print(f"accepted={result.accepted}")
    print(f"uncertainty={result.uncertainty:.6f}")
    print(f"evidence={len(result.evidence)}")
    print(f"rounds={len(result.rounds)}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
