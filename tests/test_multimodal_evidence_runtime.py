import json

import pytest

from jarvisx.multimodal_evidence_runtime import (
    ActiveMultimodalEvidenceEngine,
    EvidenceSegment,
    EvidenceStore,
    Modality,
    MultimodalRetriever,
    Region2D,
    Region3D,
    RetrievalQuery,
    TimeSpan,
    demo_store,
    result_to_json,
)


def test_evidence_segments_validate_temporal_and_spatial_regions() -> None:
    segment = EvidenceSegment(
        evidence_id="video-7",
        object_id="object-7",
        modality=Modality.VIDEO,
        summary="localized audiovisual evidence",
        provenance="unit-test",
        confidence=0.9,
        time_span=TimeSpan(12.5, 18.0),
        region_2d=Region2D(0.1, 0.2, 0.8, 0.9),
        region_3d=Region3D(0.1, 0.1, 0.1, 0.9, 0.9, 0.9),
    )
    segment.validate()

    with pytest.raises(ValueError):
        EvidenceSegment(
            evidence_id="bad",
            object_id="object",
            modality=Modality.AUDIO,
            summary="bad range",
            provenance="unit-test",
            time_span=TimeSpan(5.0, 4.0),
        ).validate()


def test_cross_modal_retrieval_localizes_audio_evidence() -> None:
    retriever = MultimodalRetriever(demo_store())
    query = RetrievalQuery(
        text="partner asks singer to come home",
        top_k=3,
        preferred_modalities=(Modality.AUDIO,),
    )

    rows = retriever.retrieve(query, (Modality.AUDIO,))

    assert rows
    assert rows[0].segment.evidence_id == "audio-1"
    assert rows[0].segment.modality is Modality.AUDIO
    assert rows[0].segment.time_span == TimeSpan(41.2, 52.8)


def test_active_runtime_expands_modalities_when_uncertainty_remains() -> None:
    engine = ActiveMultimodalEvidenceEngine(MultimodalRetriever(demo_store()))
    query = RetrievalQuery(
        text="song where his partner asks him to come home",
        top_k=3,
        preferred_modalities=(Modality.AUDIO,),
        uncertainty_threshold=0.35,
        max_rounds=3,
    )

    result = engine.run(query)

    assert len(result.rounds) >= 2
    assert result.rounds[0].target_modalities == ("audio",)
    assert Modality.AUDIO in {row.segment.modality for row in result.evidence}
    assert len({row.segment.modality for row in result.evidence}) >= 2
    assert all(round_.fixed_point_iterations >= 1 for round_ in result.rounds)


def test_required_modalities_are_accounted_by_ctr() -> None:
    engine = ActiveMultimodalEvidenceEngine(MultimodalRetriever(demo_store()))
    query = RetrievalQuery(
        text="partner asks singer to return home",
        top_k=3,
        required_modalities=(Modality.TEXT, Modality.AUDIO),
        uncertainty_threshold=0.60,
        max_rounds=2,
    )

    result = engine.run(query)

    assert result.unresolved_modalities == ()
    assert result.rounds[-1].ctr.modality_coverage == 1.0
    assert result.rounds[-1].ctr.generated >= 2
    assert result.rounds[-1].fixed_point_residual >= 0.0


def test_missing_required_modality_remains_explicit() -> None:
    store = EvidenceStore(
        (
            EvidenceSegment(
                evidence_id="text-only",
                object_id="object",
                modality=Modality.TEXT,
                summary="textual claim",
                provenance="text-source",
                confidence=0.9,
            ),
        )
    )
    engine = ActiveMultimodalEvidenceEngine(MultimodalRetriever(store))
    query = RetrievalQuery(
        text="claim requiring direct audio verification",
        required_modalities=(Modality.TEXT, Modality.AUDIO),
        uncertainty_threshold=0.9,
        max_rounds=2,
    )

    result = engine.run(query)

    assert result.accepted is False
    assert result.unresolved_modalities == ("audio",)


def test_report_is_json_serializable_and_preserves_media_boundary() -> None:
    engine = ActiveMultimodalEvidenceEngine(MultimodalRetriever(demo_store()))
    result = engine.run(
        RetrievalQuery(
            text="song where partner asks him to come home",
            required_modalities=(Modality.TEXT, Modality.AUDIO),
            uncertainty_threshold=0.60,
            max_rounds=2,
        )
    )
    report = result_to_json(result)
    encoded = json.dumps(report)

    assert "jarvisx.multimodal-evidence-runtime.v1" in encoded
    assert "production media decoding" in report["claim_boundary"]["adapter_boundary"]
    audio = next(
        row["segment"]
        for row in report["evidence"]
        if row["segment"]["modality"] == "audio"
    )
    assert audio["time_span"]["start_seconds"] == 41.2


def test_runtime_is_deterministic() -> None:
    query = RetrievalQuery(
        text="partner asks him to come home",
        top_k=3,
        preferred_modalities=(Modality.AUDIO, Modality.VIDEO),
        max_rounds=3,
    )
    left = ActiveMultimodalEvidenceEngine(MultimodalRetriever(demo_store())).run(query)
    right = ActiveMultimodalEvidenceEngine(MultimodalRetriever(demo_store())).run(query)

    assert left == right
