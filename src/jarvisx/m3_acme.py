"""Deterministic reference runtime for the Moagi 3D AutoCodec Manifold Equation (M³-ACME).

This module operationalizes the user-facing equation as a bounded research adapter:

    X --C--> X_c --E_Theta--> Z --A--> Z* --D_Phi--> X_hat

It deliberately does not crawl the web and does not infer legal permission. Upstream callers
must provide explicit compliance/provenance assertions. The reference encoder is canonical
JSON + zlib, making the allowed-record round trip deterministic and exactly reversible.
Learned encoders/decoders can replace the codec behind the same runtime contract later.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import zlib
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlparse


class M3ACMEError(ValueError):
    """Base error for M³-ACME validation and integrity failures."""


class ComplianceRejected(M3ACMEError):
    """Raised when the compliance/provenance gate rejects a record."""


class IntegrityError(M3ACMEError):
    """Raised when an encoded latent packet fails integrity verification."""


@dataclass(frozen=True)
class ComplianceContext:
    """Explicit upstream authorization assertions consumed by the compliance gate."""

    authorized: bool
    robots_permitted: bool = True
    terms_permitted: bool = True
    restricted_data: bool = False
    permission_basis: str = ""


@dataclass(frozen=True)
class WebBit:
    """One bounded observation in the structural/semantic/provenance/time field."""

    structure: Mapping[str, Any]
    semantics: Mapping[str, Any]
    provenance_url: str
    observed_at: str
    trust: float
    payload: Mapping[str, Any]
    compliance: ComplianceContext


@dataclass(frozen=True)
class LatentPacket:
    """Deterministic compressed representation of one admitted WebBit."""

    codec_version: str
    digest_sha256: str
    compressed: bytes
    source_url: str
    observed_at: str
    trust: float
    semantic_projection: Mapping[str, Any]


@dataclass(frozen=True)
class AbstractObject:
    """Higher-order projection copied from supported semantic metadata only."""

    entities: tuple[str, ...] = ()
    relations: tuple[tuple[str, str, str], ...] = ()
    topics: tuple[str, ...] = ()
    sentiment: str | None = None
    confidence: float | None = None
    provenance_url: str = ""
    observed_at: str = ""
    trust: float = 0.0


@dataclass(frozen=True)
class LossComponents:
    reconstruction: float
    semantic: float
    provenance: float
    temporal: float
    compliance: float
    graph: float
    total: float


@dataclass(frozen=True)
class RejectedRecord:
    index: int
    reasons: tuple[str, ...]
    provenance_url: str


@dataclass(frozen=True)
class ProcessedRecord:
    index: int
    latent_digest: str
    compressed_bytes: int
    abstraction: AbstractObject
    reconstructed: Mapping[str, Any]
    loss: LossComponents


@dataclass(frozen=True)
class RuntimeReport:
    accepted: tuple[ProcessedRecord, ...]
    rejected: tuple[RejectedRecord, ...]

    @property
    def accepted_count(self) -> int:
        return len(self.accepted)

    @property
    def rejected_count(self) -> int:
        return len(self.rejected)

    @property
    def mean_loss(self) -> float:
        if not self.accepted:
            return 0.0
        return sum(record.loss.total for record in self.accepted) / len(self.accepted)


@dataclass(frozen=True)
class M3ACMEConfig:
    """Resource and loss configuration for the bounded reference runtime."""

    codec_version: str = "m3-acme-json-zlib-v1"
    compression_level: int = 9
    max_record_bytes: int = 1_000_000
    max_age_seconds: float = 30 * 24 * 3600.0
    require_https: bool = True
    lambda_semantic: float = 1.0
    lambda_provenance: float = 1.0
    lambda_temporal: float = 1.0
    lambda_compliance: float = 1.0
    lambda_graph: float = 1.0

    def __post_init__(self) -> None:
        if not 0 <= self.compression_level <= 9:
            raise M3ACMEError("compression_level must be in [0, 9]")
        if self.max_record_bytes <= 0:
            raise M3ACMEError("max_record_bytes must be positive")
        if self.max_age_seconds <= 0:
            raise M3ACMEError("max_age_seconds must be positive")
        weights = (
            self.lambda_semantic,
            self.lambda_provenance,
            self.lambda_temporal,
            self.lambda_compliance,
            self.lambda_graph,
        )
        if any((not math.isfinite(value) or value < 0.0) for value in weights):
            raise M3ACMEError("loss weights must be finite and non-negative")


class M3ACMERuntime:
    """Reference compiler/runtime for the M³-ACME operational identity."""

    def __init__(self, config: M3ACMEConfig | None = None) -> None:
        self.config = config or M3ACMEConfig()

    @staticmethod
    def _parse_time(value: str) -> datetime:
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError as exc:
            raise M3ACMEError(f"invalid observed_at timestamp: {value!r}") from exc
        if parsed.tzinfo is None:
            raise M3ACMEError("observed_at must include an explicit timezone")
        return parsed.astimezone(timezone.utc)

    @staticmethod
    def _canonical_json(value: Mapping[str, Any]) -> bytes:
        try:
            text = json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise M3ACMEError("record is not canonical-JSON serializable") from exc
        return text.encode("utf-8")

    def compliance_reasons(self, bit: WebBit) -> tuple[str, ...]:
        reasons: list[str] = []
        ctx = bit.compliance
        parsed = urlparse(bit.provenance_url)

        if not ctx.authorized:
            reasons.append("authorization_missing")
        if not ctx.robots_permitted:
            reasons.append("robots_disallowed")
        if not ctx.terms_permitted:
            reasons.append("terms_disallowed")
        if ctx.restricted_data:
            reasons.append("restricted_data")
        if not ctx.permission_basis.strip():
            reasons.append("permission_basis_missing")
        if not parsed.scheme or not parsed.netloc:
            reasons.append("provenance_url_invalid")
        elif self.config.require_https and parsed.scheme.lower() != "https":
            reasons.append("https_required")
        if not math.isfinite(bit.trust) or not 0.0 <= bit.trust <= 1.0:
            reasons.append("trust_out_of_range")
        try:
            self._parse_time(bit.observed_at)
        except M3ACMEError:
            reasons.append("timestamp_invalid")
        return tuple(reasons)

    def _record_mapping(self, bit: WebBit) -> Mapping[str, Any]:
        return {
            "structure": dict(bit.structure),
            "semantics": dict(bit.semantics),
            "provenance": {
                "url": bit.provenance_url,
                "observed_at": bit.observed_at,
                "trust": bit.trust,
                "permission_basis": bit.compliance.permission_basis,
            },
            "payload": dict(bit.payload),
        }

    def encode(self, bit: WebBit) -> LatentPacket:
        """Apply C then E_Theta: fail closed, canonicalize, compress, and integrity-bind."""

        reasons = self.compliance_reasons(bit)
        if reasons:
            raise ComplianceRejected(",".join(reasons))

        raw = self._canonical_json(self._record_mapping(bit))
        if len(raw) > self.config.max_record_bytes:
            raise ComplianceRejected("record_size_limit")
        compressed = zlib.compress(raw, level=self.config.compression_level)
        digest = hashlib.sha256(raw).hexdigest()
        projection = {
            key: bit.semantics[key]
            for key in ("entities", "relations", "topics", "sentiment", "confidence")
            if key in bit.semantics
        }
        return LatentPacket(
            codec_version=self.config.codec_version,
            digest_sha256=digest,
            compressed=compressed,
            source_url=bit.provenance_url,
            observed_at=bit.observed_at,
            trust=bit.trust,
            semantic_projection=projection,
        )

    @staticmethod
    def _string_tuple(value: Any) -> tuple[str, ...]:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
            return ()
        return tuple(str(item) for item in value)

    def abstract(self, packet: LatentPacket) -> AbstractObject:
        """Apply A without fabricating semantics absent from the admitted source record."""

        semantics = packet.semantic_projection
        relations: list[tuple[str, str, str]] = []
        raw_relations = semantics.get("relations", ())
        if isinstance(raw_relations, Sequence) and not isinstance(raw_relations, (str, bytes)):
            for relation in raw_relations:
                if isinstance(relation, Sequence) and not isinstance(relation, (str, bytes)):
                    parts = tuple(str(part) for part in relation)
                    if len(parts) == 3:
                        relations.append((parts[0], parts[1], parts[2]))

        confidence_value = semantics.get("confidence")
        confidence: float | None = None
        if isinstance(confidence_value, (int, float)) and math.isfinite(float(confidence_value)):
            confidence = float(confidence_value)

        sentiment_value = semantics.get("sentiment")
        sentiment = str(sentiment_value) if sentiment_value is not None else None

        return AbstractObject(
            entities=self._string_tuple(semantics.get("entities", ())),
            relations=tuple(relations),
            topics=self._string_tuple(semantics.get("topics", ())),
            sentiment=sentiment,
            confidence=confidence,
            provenance_url=packet.source_url,
            observed_at=packet.observed_at,
            trust=packet.trust,
        )

    def decode(self, packet: LatentPacket) -> Mapping[str, Any]:
        """Apply D_Phi and verify the latent packet before returning structured X_hat."""

        if packet.codec_version != self.config.codec_version:
            raise IntegrityError("codec_version_mismatch")
        try:
            raw = zlib.decompress(packet.compressed)
        except zlib.error as exc:
            raise IntegrityError("compressed_payload_corrupt") from exc
        if len(raw) > self.config.max_record_bytes:
            raise IntegrityError("decoded_record_size_limit")
        if hashlib.sha256(raw).hexdigest() != packet.digest_sha256:
            raise IntegrityError("digest_mismatch")
        try:
            decoded = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise IntegrityError("decoded_payload_invalid_json") from exc
        if not isinstance(decoded, dict):
            raise IntegrityError("decoded_payload_not_object")
        return decoded

    @staticmethod
    def _semantic_loss(source: Mapping[str, Any], decoded: Mapping[str, Any]) -> float:
        reconstructed = decoded.get("semantics")
        return 0.0 if reconstructed == dict(source) else 1.0

    @staticmethod
    def _provenance_loss(bit: WebBit, decoded: Mapping[str, Any]) -> float:
        provenance = decoded.get("provenance")
        if not isinstance(provenance, Mapping):
            return 1.0
        expected = {
            "url": bit.provenance_url,
            "observed_at": bit.observed_at,
            "trust": bit.trust,
            "permission_basis": bit.compliance.permission_basis,
        }
        return 0.0 if dict(provenance) == expected else 1.0

    def _temporal_loss(self, bit: WebBit, now: datetime) -> float:
        observed = self._parse_time(bit.observed_at)
        normalized_now = now.astimezone(timezone.utc)
        age = max(0.0, (normalized_now - observed).total_seconds())
        return min(1.0, age / self.config.max_age_seconds)

    @staticmethod
    def _graph_loss(abstraction: AbstractObject) -> float:
        if not abstraction.relations:
            return 0.0
        entities = set(abstraction.entities)
        broken = sum(
            1
            for source, _predicate, target in abstraction.relations
            if source not in entities or target not in entities
        )
        return broken / len(abstraction.relations)

    def loss(
        self,
        bit: WebBit,
        decoded: Mapping[str, Any],
        abstraction: AbstractObject,
        *,
        now: datetime | None = None,
    ) -> LossComponents:
        """Evaluate the operational Moagi composite loss over an admitted record."""

        expected = self._record_mapping(bit)
        reconstruction = 0.0 if decoded == expected else 1.0
        semantic = self._semantic_loss(bit.semantics, decoded)
        provenance = self._provenance_loss(bit, decoded)
        temporal = self._temporal_loss(bit, now or datetime.now(timezone.utc))
        compliance = 0.0 if not self.compliance_reasons(bit) else 1.0
        graph = self._graph_loss(abstraction)
        total = (
            reconstruction
            + self.config.lambda_semantic * semantic
            + self.config.lambda_provenance * provenance
            + self.config.lambda_temporal * temporal
            + self.config.lambda_compliance * compliance
            + self.config.lambda_graph * graph
        )
        return LossComponents(
            reconstruction=reconstruction,
            semantic=semantic,
            provenance=provenance,
            temporal=temporal,
            compliance=compliance,
            graph=graph,
            total=total,
        )

    def process(self, bits: Iterable[WebBit], *, now: datetime | None = None) -> RuntimeReport:
        """Execute C -> E -> A -> D -> L for each input and emit a bounded receipt."""

        accepted: list[ProcessedRecord] = []
        rejected: list[RejectedRecord] = []
        evaluation_time = now or datetime.now(timezone.utc)

        for index, bit in enumerate(bits):
            reasons = self.compliance_reasons(bit)
            if reasons:
                rejected.append(
                    RejectedRecord(index=index, reasons=reasons, provenance_url=bit.provenance_url)
                )
                continue
            try:
                packet = self.encode(bit)
            except ComplianceRejected as exc:
                rejected.append(
                    RejectedRecord(
                        index=index,
                        reasons=tuple(str(exc).split(",")),
                        provenance_url=bit.provenance_url,
                    )
                )
                continue
            abstraction = self.abstract(packet)
            decoded = self.decode(packet)
            metrics = self.loss(bit, decoded, abstraction, now=evaluation_time)
            accepted.append(
                ProcessedRecord(
                    index=index,
                    latent_digest=packet.digest_sha256,
                    compressed_bytes=len(packet.compressed),
                    abstraction=abstraction,
                    reconstructed=decoded,
                    loss=metrics,
                )
            )

        return RuntimeReport(accepted=tuple(accepted), rejected=tuple(rejected))


def webbit_from_mapping(value: Mapping[str, Any]) -> WebBit:
    """Parse a JSON-compatible mapping into a strongly validated WebBit input."""

    compliance_value = value.get("compliance", {})
    if not isinstance(compliance_value, Mapping):
        raise M3ACMEError("compliance must be an object")
    return WebBit(
        structure=dict(value.get("structure", {})),
        semantics=dict(value.get("semantics", {})),
        provenance_url=str(value.get("provenance_url", "")),
        observed_at=str(value.get("observed_at", "")),
        trust=float(value.get("trust", 0.0)),
        payload=dict(value.get("payload", {})),
        compliance=ComplianceContext(
            authorized=bool(compliance_value.get("authorized", False)),
            robots_permitted=bool(compliance_value.get("robots_permitted", True)),
            terms_permitted=bool(compliance_value.get("terms_permitted", True)),
            restricted_data=bool(compliance_value.get("restricted_data", False)),
            permission_basis=str(compliance_value.get("permission_basis", "")),
        ),
    )


def _report_to_json(report: RuntimeReport) -> str:
    data = {
        "accepted_count": report.accepted_count,
        "rejected_count": report.rejected_count,
        "mean_loss": report.mean_loss,
        "accepted": [asdict(item) for item in report.accepted],
        "rejected": [asdict(item) for item in report.rejected],
    }
    return json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the deterministic M³-ACME reference pipeline")
    parser.add_argument("input", type=Path, help="JSON file containing a list of WebBit records")
    args = parser.parse_args(argv)

    raw = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise M3ACMEError("input JSON must contain a list")
    bits = [webbit_from_mapping(item) for item in raw]
    report = M3ACMERuntime().process(bits)
    print(_report_to_json(report))
    return 0 if report.rejected_count == 0 else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
