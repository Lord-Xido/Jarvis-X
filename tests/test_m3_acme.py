from dataclasses import replace
from datetime import datetime, timezone

import pytest

from jarvisx.m3_acme import (
    ComplianceContext,
    IntegrityError,
    M3ACMEConfig,
    M3ACMERuntime,
    WebBit,
)


def fixture_bit(**overrides):
    values = {
        "structure": {"tag": "article", "depth": 3},
        "semantics": {
            "entities": ["Jarvis-X", "M3-ACME"],
            "relations": [["Jarvis-X", "implements", "M3-ACME"]],
            "topics": ["autoencoding", "provenance"],
            "sentiment": "neutral",
            "confidence": 0.98,
        },
        "provenance_url": "https://example.org/spec",
        "observed_at": "2026-08-13T10:00:00+00:00",
        "trust": 0.9,
        "payload": {"text": "bounded source content"},
        "compliance": ComplianceContext(
            authorized=True,
            robots_permitted=True,
            terms_permitted=True,
            restricted_data=False,
            permission_basis="public-test-fixture",
        ),
    }
    values.update(overrides)
    return WebBit(**values)


def test_round_trip_is_deterministic_and_exact():
    runtime = M3ACMERuntime()
    bit = fixture_bit()

    first = runtime.encode(bit)
    second = runtime.encode(bit)

    assert first.digest_sha256 == second.digest_sha256
    assert first.compressed == second.compressed
    assert runtime.decode(first) == runtime.decode(second)
    assert runtime.decode(first)["payload"] == bit.payload


def test_process_executes_full_mask_encode_abstract_decode_loss_cycle():
    runtime = M3ACMERuntime(M3ACMEConfig(max_age_seconds=86_400.0))
    bit = fixture_bit(observed_at="2026-08-13T12:00:00+00:00")
    report = runtime.process([bit], now=datetime(2026, 8, 13, 12, 0, tzinfo=timezone.utc))

    assert report.accepted_count == 1
    assert report.rejected_count == 0
    record = report.accepted[0]
    assert record.abstraction.entities == ("Jarvis-X", "M3-ACME")
    assert record.loss.reconstruction == 0.0
    assert record.loss.semantic == 0.0
    assert record.loss.provenance == 0.0
    assert record.loss.temporal == 0.0
    assert record.loss.graph == 0.0
    assert record.loss.total == 0.0


def test_compliance_gate_is_fail_closed_and_reports_reasons():
    runtime = M3ACMERuntime()
    denied = fixture_bit(
        compliance=ComplianceContext(
            authorized=False,
            robots_permitted=False,
            terms_permitted=True,
            permission_basis="",
        )
    )

    report = runtime.process([denied])

    assert report.accepted_count == 0
    assert report.rejected_count == 1
    assert set(report.rejected[0].reasons) >= {
        "authorization_missing",
        "robots_disallowed",
        "permission_basis_missing",
    }


def test_decoder_rejects_corrupted_latent_packet():
    runtime = M3ACMERuntime()
    packet = runtime.encode(fixture_bit())
    corrupted = replace(packet, compressed=packet.compressed[:-1] + b"x")

    with pytest.raises(IntegrityError):
        runtime.decode(corrupted)


def test_abstraction_does_not_invent_missing_semantics():
    runtime = M3ACMERuntime()
    packet = runtime.encode(fixture_bit(semantics={"topics": ["codec"]}))
    abstraction = runtime.abstract(packet)

    assert abstraction.entities == ()
    assert abstraction.relations == ()
    assert abstraction.topics == ("codec",)
    assert abstraction.sentiment is None
    assert abstraction.confidence is None


def test_temporal_and_graph_losses_are_independent_components():
    runtime = M3ACMERuntime(M3ACMEConfig(max_age_seconds=3600.0))
    bit = fixture_bit(
        observed_at="2026-08-13T10:00:00+00:00",
        semantics={
            "entities": ["A"],
            "relations": [["A", "links", "B"]],
        },
    )
    packet = runtime.encode(bit)
    abstraction = runtime.abstract(packet)
    decoded = runtime.decode(packet)
    loss = runtime.loss(
        bit,
        decoded,
        abstraction,
        now=datetime(2026, 8, 13, 12, 0, tzinfo=timezone.utc),
    )

    assert loss.temporal == 1.0
    assert loss.graph == 1.0
    assert loss.reconstruction == 0.0
    assert loss.total == 2.0


def test_http_provenance_is_rejected_when_https_is_required():
    runtime = M3ACMERuntime()
    report = runtime.process([fixture_bit(provenance_url="http://example.org/spec")])
    assert report.rejected[0].reasons == ("https_required",)
