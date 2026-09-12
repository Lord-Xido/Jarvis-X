import json
import math

from jarvisx.dr_moagi_multimodal_loop import Modality
from jarvisx.dr_moagi_operational_runtime import (
    CandidateConfig,
    RecursiveOperationalController,
    candidate_grid,
)


def test_operational_candidate_is_verified_and_localises_error():
    controller = RecursiveOperationalController(
        {
            Modality.TEXT: "Jarvis X recursive operational verification",
            Modality.AUDIO: bytes(range(96)),
        }
    )
    _, report = controller.evaluate(
        CandidateConfig(
            edge=8,
            temporal_depth=3,
            mix=0.35,
            cycles=2,
            deterministic_stride=10**20,
            seed=17,
        )
    )

    verification = report.verification
    assert verification.valid
    assert verification.finite_metrics
    assert verification.fusion_normalized
    assert verification.generated_all_modalities
    assert verification.input_lengths_preserved
    assert math.isfinite(verification.aggregate_reconstruction_mse)
    assert math.isfinite(verification.aggregate_cycle_mse)
    assert math.isfinite(verification.fixed_point_delta)
    assert math.isfinite(verification.score)
    assert verification.active_regions
    assert all(region.absolute_error >= 0.0 for region in verification.active_regions)


def test_candidate_grid_is_cartesian_product():
    configs = candidate_grid(
        edge=8,
        depths=(2, 4),
        mixes=(0.2, 0.5, 0.8),
        cycles=2,
        seed=19,
        deterministic_stride=123,
    )
    assert len(configs) == 6
    assert {(c.temporal_depth, c.mix) for c in configs} == {
        (2, 0.2),
        (2, 0.5),
        (2, 0.8),
        (4, 0.2),
        (4, 0.5),
        (4, 0.8),
    }


def test_optimizer_exports_selected_candidate_and_memory(tmp_path):
    out = tmp_path / "out"
    memory = tmp_path / "telemetry.jsonl"
    controller = RecursiveOperationalController(
        {Modality.TEXT: "verified inward loop"},
        memory_path=memory,
        active_region_limit=5,
    )
    configs = (
        CandidateConfig(edge=8, temporal_depth=2, mix=0.20, cycles=2, seed=23),
        CandidateConfig(edge=8, temporal_depth=3, mix=0.45, cycles=2, seed=23),
    )

    result = controller.optimize(configs, output_dir=out)

    assert result.selected.verification.valid
    assert result.selected in result.candidates
    valid_scores = [c.verification.score for c in result.candidates if c.verification.valid]
    assert result.selected.verification.score == min(valid_scores)

    assert (out / "operational-report.json").is_file()
    assert (out / "metrics.json").is_file()
    assert (out / "fused-latent.obj").is_file()
    assert (out / "generated-text.raw").is_file()

    payload = json.loads((out / "operational-report.json").read_text("utf-8"))
    assert payload["selected"]["verification"]["valid"] is True
    assert len(payload["candidates"]) == 2

    lines = memory.read_text("utf-8").splitlines()
    assert len(lines) == 1
    ledger_record = json.loads(lines[0])
    assert ledger_record["selected"]["verification"]["valid"] is True


def test_invalid_candidate_configuration_is_rejected():
    controller = RecursiveOperationalController({Modality.TEXT: "x"})
    bad = CandidateConfig(edge=7, temporal_depth=2, mix=0.2, cycles=2)
    try:
        controller.evaluate(bad)
    except ValueError as exc:
        assert "edge" in str(exc)
    else:
        raise AssertionError("invalid odd edge must fail")
