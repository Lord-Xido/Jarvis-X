from types import SimpleNamespace

import pytest

from jarvisx.dr_moagi_codex import DrMoagiCodexResult, FixedPointResult
from jarvisx.dr_moagi_epistemic import (
    EpistemicDrMoagiCodex,
    EpistemicGateConfig,
    EvidenceKind,
    EvidencePacket,
    ObservationPacket,
    scene_digest,
    scene_nrmse,
)

POINT = (0.0, 0.0, 0.0)


class FakeCodex:
    def __init__(self, decoded_value=1.0, eta_theta=0.1):
        self.decoded_value = decoded_value
        self.config = SimpleNamespace(eta_theta=eta_theta)
        self.calls = 0

    def execute(self, scene, **kwargs):
        self.calls += 1
        return DrMoagiCodexResult(
            encoded_latent=(1.0,),
            inward_latent=(1.0,),
            raw_latent=(1.0,),
            smoothed_latent=(1.0,),
            projected_latent=(self.decoded_value,),
            decoded_scene={POINT: self.decoded_value},
            theta_before=None,
            theta_after=None,
            source_charge={POINT: 2.0},
            permeation_field={POINT: 3.0 + 0.0j},
            fixed_point=FixedPointResult((1.0,), 4, True, 0.0, None),
            virtual_depth_label="1000000^1000000",
        )


def obs(value=1.0, **kwargs):
    return ObservationPacket(
        scene={POINT: value},
        source_id=kwargs.pop("source_id", "camera-1"),
        kind=kwargs.pop("kind", EvidenceKind.SENSOR),
        **kwargs,
    )


def evidence(value=1.0, **kwargs):
    return EvidencePacket(
        scene={POINT: value},
        source_id=kwargs.pop("source_id", "depth-1"),
        kind=kwargs.pop("kind", EvidenceKind.INSTRUMENT),
        **kwargs,
    )


def test_model_generated_observation_is_rejected_before_candidate_execution():
    codex = FakeCodex()
    guard = EpistemicDrMoagiCodex(codex)
    with pytest.raises(ValueError, match="external input"):
        guard.execute(obs(kind=EvidenceKind.MODEL), evidence=[evidence()])
    assert codex.calls == 0


def test_missing_independent_evidence_quarantines_hypothesis_learning_and_permeation():
    codex = FakeCodex()
    guard = EpistemicDrMoagiCodex(codex)
    result = guard.execute(obs(), evidence=[], theta=(1.0,), theta_gradient=(0.5,))
    assert not result.verdict.admitted
    assert result.committed_scene is None
    assert not result.learning_committed
    assert result.theta_after == (1.0,)
    assert result.released_source_charge == {}
    assert result.released_permeation_field == {}
    assert result.hypothesis_scene == {POINT: 1.0}


def test_candidate_conflicting_with_observation_and_evidence_is_rejected():
    codex = FakeCodex(decoded_value=2.0)
    guard = EpistemicDrMoagiCodex(codex)
    result = guard.execute(obs(1.0), evidence=[evidence(1.0)])
    assert not result.verdict.admitted
    assert "hypothesis exceeds observation error threshold" in result.verdict.reasons
    assert any("conflicts with evidence" in reason for reason in result.verdict.reasons)


def test_verified_candidate_commits_learning_and_releases_permeation():
    codex = FakeCodex(decoded_value=1.0, eta_theta=0.2)
    guard = EpistemicDrMoagiCodex(codex)
    result = guard.execute(
        obs(1.0),
        evidence=[evidence(1.0)],
        theta=(1.0, -1.0),
        theta_gradient=(0.5, 0.25),
    )
    assert result.verdict.admitted
    assert result.committed_scene == {POINT: 1.0}
    assert result.learning_committed
    assert result.theta_after == pytest.approx((0.9, -1.05))
    assert result.released_source_charge == {POINT: 2.0}
    assert result.released_permeation_field == {POINT: 3.0 + 0.0j}


def test_same_dependency_group_does_not_fake_independent_evidence_count():
    codex = FakeCodex()
    guard = EpistemicDrMoagiCodex(
        codex,
        EpistemicGateConfig(min_independent_evidence=2),
    )
    result = guard.execute(
        obs(),
        evidence=[
            evidence(source_id="depth-a", independence_key="sensor-rig-1"),
            evidence(source_id="depth-b", independence_key="sensor-rig-1"),
        ],
    )
    assert not result.verdict.admitted
    assert result.verdict.independent_evidence_count == 1


def test_evidence_derived_from_hypothesis_is_inadmissible():
    codex = FakeCodex()
    guard = EpistemicDrMoagiCodex(codex)
    result = guard.execute(obs(), evidence=[evidence(derived_from_hypothesis=True)])
    assert not result.verdict.admitted
    assert any("inadmissible evidence source" in reason for reason in result.verdict.reasons)


def test_run_anchor_is_immutable_until_explicit_reset():
    codex = FakeCodex()
    guard = EpistemicDrMoagiCodex(codex)
    first = guard.begin_run(obs(1.0))
    with pytest.raises(RuntimeError, match="immutable"):
        guard.begin_run(obs(2.0, source_id="camera-2"))
    assert guard.anchor == first
    guard.reset_run()
    second = guard.begin_run(obs(2.0, source_id="camera-2"))
    assert second.digest != first.digest


def test_scene_metrics_are_deterministic_and_fail_closed_on_support_mismatch():
    a = {(0.0, 0.0, 0.0): 1.0, (1.0, 0.0, 0.0): 2.0}
    b = {(1.0, 0.0, 0.0): 2.0, (0.0, 0.0, 0.0): 1.0}
    assert scene_digest(a) == scene_digest(b)
    assert scene_nrmse(a, b) == pytest.approx(0.0)
    assert scene_nrmse({POINT: 1.0}, a) == float("inf")
