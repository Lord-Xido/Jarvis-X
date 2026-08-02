import json
from dataclasses import asdict
from pathlib import Path

import pytest

from jarvisx.hyperion import HyperionConfig, HyperionEngine, Observation, ScoreModel
from jarvisx.hyperion_cli import main
from jarvisx.hyperion_service import (
    AtomicReportStore,
    HyperionRuntime,
    HyperionRuntimeSettings,
    build_evidence_bundle,
    create_hyperion_app,
    verify_evidence_bundle,
)


def observation(source: str, timestamp: int, value: float, event: str) -> Observation:
    return Observation(
        source=source,
        timestamp_ms=timestamp,
        value=value,
        quantity="amount",
        unit="ZAR",
        correlation_id=event,
        label="known",
    )


def fixture() -> list[Observation]:
    return [
        observation("csv", 0, 100.0, "e0"),
        observation("cpu", 1, 100.0, "e0"),
        observation("csv", 1000, 110.0, "e1"),
        observation("cpu", 1001, 110.0, "e1"),
    ]


def test_evidence_bundle_is_deterministic_and_replayable() -> None:
    engine = HyperionEngine()
    first = build_evidence_bundle(engine, fixture())
    second = build_evidence_bundle(engine, list(reversed(fixture())))

    assert first == second
    assert verify_evidence_bundle(first, engine) == (True, "verified")


def test_tampered_bundle_fails_before_replay() -> None:
    engine = HyperionEngine()
    bundle = build_evidence_bundle(engine, fixture())
    bundle["report_digest"] = "0" * 64

    verified, reason = verify_evidence_bundle(bundle, engine)

    assert not verified
    assert reason == "bundle digest mismatch"


def test_atomic_store_is_idempotent_and_detects_digest_collision(tmp_path: Path) -> None:
    store = AtomicReportStore(tmp_path)
    bundle = build_evidence_bundle(HyperionEngine(), fixture())

    first = store.put(bundle)
    second = store.put(bundle)

    assert first == second
    assert store.count() == 1
    conflicting = dict(bundle)
    conflicting["bundle_digest"] = "f" * 64
    with pytest.raises(RuntimeError, match="collision"):
        store.put(conflicting)


def test_runtime_persists_and_reverifies_reports(tmp_path: Path) -> None:
    runtime = HyperionRuntime(
        HyperionEngine(),
        HyperionRuntimeSettings(data_dir=tmp_path, max_observations=10),
    )

    result = runtime.audit(fixture(), request_id="request-1")
    verification = runtime.verify(str(result["report_digest"]))

    assert result["verified"] is True
    assert verification["verified"] is True
    assert runtime.store.count() == 1
    assert "hyperion_audits_total 1" in runtime.metrics_text()
    assert "hyperion_observations_total 4" in runtime.metrics_text()


def test_runtime_enforces_batch_limit(tmp_path: Path) -> None:
    runtime = HyperionRuntime(
        HyperionEngine(),
        HyperionRuntimeSettings(data_dir=tmp_path, max_observations=1),
    )

    with pytest.raises(ValueError, match="exceeds limit"):
        runtime.audit(fixture())


def test_authentication_cannot_be_required_without_a_key(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="API_KEY"):
        HyperionRuntimeSettings(
            data_dir=tmp_path,
            require_api_key=True,
            api_key=None,
        )


def test_app_registers_operational_routes_without_replacing_root(tmp_path: Path) -> None:
    app = create_hyperion_app(
        settings=HyperionRuntimeSettings(data_dir=tmp_path),
        engine=HyperionEngine(),
    )
    paths = {route.path for route in app.routes}

    assert "/healthz" in paths
    assert "/readyz" in paths
    assert "/metrics" in paths
    assert "/v1/hyperion/audits" in paths
    assert "/v1/hyperion/audits/{report_digest}/verify" in paths
    assert "/" not in paths


def test_bundle_rejects_different_frozen_model() -> None:
    bundle = build_evidence_bundle(HyperionEngine(), fixture())
    different = HyperionEngine(model=ScoreModel(bias=-2.0))

    verified, reason = verify_evidence_bundle(bundle, different)

    assert not verified
    assert reason == "model hash mismatch"


def test_cli_audit_and_verify_round_trip(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    input_path = tmp_path / "observations.json"
    bundle_path = tmp_path / "bundle.json"
    input_path.write_text(
        json.dumps(
            {
                "observations": [
                    {
                        "source": item.source,
                        "timestamp_ms": item.timestamp_ms,
                        "value": item.value,
                        "quantity": item.quantity,
                        "unit": item.unit,
                        "correlation_id": item.correlation_id,
                        "confidence": item.confidence,
                        "available": item.available,
                        "label": item.label,
                        "metadata": {},
                    }
                    for item in fixture()
                ]
            }
        ),
        encoding="utf-8",
    )

    assert main(["audit", str(input_path), "--output", str(bundle_path)]) == 0
    audit_output = json.loads(capsys.readouterr().out)
    assert audit_output["verified"] is True
    assert bundle_path.exists()

    assert main(["verify", str(bundle_path)]) == 0
    verify_output = json.loads(capsys.readouterr().out)
    assert verify_output == {"reason": "verified", "verified": True}


def test_cli_uses_explicit_configuration_file(tmp_path: Path) -> None:
    input_path = tmp_path / "observations.json"
    output_path = tmp_path / "bundle.json"
    config_path = tmp_path / "config.json"
    input_path.write_text(
        json.dumps(
            [
                {
                    "source": "csv",
                    "timestamp_ms": 0,
                    "value": 5.0,
                    "quantity": "amount",
                    "unit": "USD",
                    "correlation_id": "e0",
                }
            ]
        ),
        encoding="utf-8",
    )
    config_path.write_text(
        json.dumps(
            {
                **asdict(HyperionConfig(target_unit="USD")),
                "source_weights": {"csv": 1.0},
            }
        ),
        encoding="utf-8",
    )

    assert (
        main(
            [
                "audit",
                str(input_path),
                "--output",
                str(output_path),
                "--config",
                str(config_path),
            ]
        )
        == 0
    )
    assert output_path.exists()
