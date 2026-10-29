"""Real Jarvis-X lexical 3D encode/decode A/B research checks."""
import pytest

from jarvisx.dr_moagi_phi_m_codec_probe import ProbeConfig, _corpus, _run_once, run_probe
from jarvisx.dr_moagi_phi_m_3d import PhiMConfig


def test_lexical_baseline_roundtrip_is_exact():
    cfg = ProbeConfig(word_count=48, repetitions=1)
    codec, text = _corpus(cfg)
    result = _run_once(codec, text, cfg, "baseline", PhiMConfig())
    assert result["exact_text"]
    assert result["token_accuracy"] == 1.0
    assert result["coordinate_rmse"] == 0


def test_residual_bridge_recovers_original_tokens_without_claiming_compression():
    cfg = ProbeConfig(word_count=48, repetitions=1)
    codec, text = _corpus(cfg)
    result = _run_once(codec, text, cfg, "residual_phi", PhiMConfig())
    assert result["exact_text"]
    assert result["token_accuracy"] == 1.0
    assert result["coordinate_rmse"] < cfg.residual_quantum
    assert result["coordinate_payload_bytes"] > result["ideal_uint16_token_id_bytes"]


def test_direct_warp_is_measured_and_never_assumed_invertible():
    cfg = ProbeConfig(word_count=48, repetitions=1)
    codec, text = _corpus(cfg)
    result = _run_once(codec, text, cfg, "direct_phi", PhiMConfig())
    assert 0.0 <= result["token_accuracy"] <= 1.0
    assert result["coordinate_rmse"] >= 0


def test_paired_pipeline_is_shadow_only():
    report = run_probe(ProbeConfig(word_count=32, repetitions=2))
    assert report["status"] == "shadow_only_no_runtime_mutation"
    assert set(report["methods"]) == {"baseline", "direct_phi", "residual_phi"}
    assert report["methods"]["baseline"]["exact_text"]
    assert report["methods"]["residual_phi"]["exact_text"]
    assert report["selected"] == "baseline"  # 18 vs 12 bytes/coordinate
    assert not report["promotion_eligible"]
    assert all(t > 0 for t in report["median_end_to_end_ms"].values())


@pytest.mark.parametrize("kwargs", [
    {"word_count": 0}, {"word_count": 1001}, {"repetitions": 0},
    {"repetitions": 22}, {"seed": -1}, {"time": -1},
    {"residual_quantum": float("nan")},
])
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        ProbeConfig(**kwargs)


def test_unsupported_mode_fails_closed():
    cfg = ProbeConfig(word_count=8, repetitions=1)
    codec, text = _corpus(cfg)
    with pytest.raises(ValueError, match="mode"):
        _run_once(codec, text, cfg, "arbitrary", PhiMConfig())
