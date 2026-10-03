from pathlib import Path

import pytest

from jarvisx.dr_moagi_multimodal_loop import Modality as RuntimeModality
from jarvisx.dr_moagi_operational_runtime import CandidateConfig, RecursiveOperationalController
from jarvisx.evidence_grounded_operational import (
    EvidenceGateRejected,
    EvidenceGroundedOperationalController,
    evidence_gate_receipt,
)
from jarvisx.multimodal_evidence_runtime import (
    ActiveMultimodalEvidenceEngine,
    EvidenceSegment,
    EvidenceStore,
    Modality,
    MultimodalRetriever,
    RetrievalQuery,
    demo_store,
)


def _config() -> CandidateConfig:
    return CandidateConfig(
        edge=4,
        temporal_depth=2,
        mix=0.35,
        cycles=2,
        seed=101,
        deterministic_stride=0,
    )


def test_evidence_gate_blocks_operational_promotion_when_audio_is_missing(
    tmp_path: Path,
) -> None:
    store = EvidenceStore(
        (
            EvidenceSegment(
                evidence_id="text-only",
                object_id="object",
                modality=Modality.TEXT,
                summary="metadata says a partner asks him to come home",
                provenance="catalog",
                confidence=0.95,
            ),
        )
    )
    evidence_engine = ActiveMultimodalEvidenceEngine(MultimodalRetriever(store))
    operational = RecursiveOperationalController(
        payloads={RuntimeModality.TEXT: "bounded test payload"}
    )
    controller = EvidenceGroundedOperationalController(evidence_engine, operational)
    query = RetrievalQuery(
        text="song where his partner asks him to come home",
        required_modalities=(Modality.TEXT, Modality.AUDIO),
        uncertainty_threshold=0.95,
        max_rounds=2,
    )

    with pytest.raises(EvidenceGateRejected, match="unresolved_modalities=audio"):
        controller.optimize(query, (_config(),), output_dir=tmp_path / "rejected")

    assert not (tmp_path / "rejected" / "operational-report.json").exists()


def test_verified_multimodal_evidence_allows_operational_promotion(
    tmp_path: Path,
) -> None:
    evidence_engine = ActiveMultimodalEvidenceEngine(
        MultimodalRetriever(demo_store())
    )
    operational = RecursiveOperationalController(
        payloads={RuntimeModality.TEXT: "bounded test payload"}
    )
    controller = EvidenceGroundedOperationalController(evidence_engine, operational)
    query = RetrievalQuery(
        text="song where his partner asks him to come home",
        top_k=3,
        required_modalities=(Modality.TEXT, Modality.AUDIO),
        uncertainty_threshold=0.60,
        max_rounds=3,
    )

    result = controller.optimize(
        query,
        (_config(),),
        output_dir=tmp_path / "accepted",
    )

    assert result.evidence.accepted is True
    assert result.evidence.unresolved_modalities == ()
    assert result.operational.selected.verification.valid is True
    assert (tmp_path / "accepted" / "evidence-gate.json").exists()
    assert (tmp_path / "accepted" / "evidence-grounded-summary.json").exists()
    assert (tmp_path / "accepted" / "operational-report.json").exists()


def test_gate_receipt_preserves_inward_evidence_fixed_point() -> None:
    engine = ActiveMultimodalEvidenceEngine(MultimodalRetriever(demo_store()))
    result = engine.run(
        RetrievalQuery(
            text="partner asks singer to return home",
            required_modalities=(Modality.TEXT, Modality.AUDIO),
            uncertainty_threshold=0.60,
            max_rounds=3,
        )
    )

    receipt = evidence_gate_receipt(result)

    assert receipt.accepted is True
    assert receipt.evidence_count >= 2
    assert receipt.rounds >= 1
    assert receipt.fixed_point_residual is not None
    assert receipt.fixed_point_residual >= 0.0
