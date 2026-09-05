from jarvisx.empirical_validation_v2 import run_validation


def test_v2_empirical_validation_aggregates_core_and_adaptive_evidence():
    report = run_validation(repetitions=2, octree_max_depth=1)
    names = {check.name for check in report.checks}

    assert report.schema_version == "jarvisx.empirical-validation.v2"
    assert len(report.checks) == 10
    assert report.passed
    assert {
        "vm_deterministic_replay",
        "shared_candidate_admission_contract",
        "field_runtime_candidate_transaction",
        "deep_distiller_atomic_adaptation",
        "virtual_3d_optimizer_admission",
        "orthogonal_quantization_precision_gate",
    } <= names
