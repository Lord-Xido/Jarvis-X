from jarvisx.candidate_adapters import virtual_3d_tuning_receipt
from jarvisx.candidate_contract import CandidateDecision
from jarvisx.dr_moagi_virtual_3d_ae import Config, DrMoagiVirtual3DAE


def _config():
    return Config(
        tile=3,
        bits=24,
        latent=6,
        passes=4,
        alpha=0.65,
        beta=0.65,
        alpha_candidates=(0.55, 0.65, 0.80),
        beta_candidates=(0.35, 0.50, 0.65),
    )


def test_virtual_3d_optimizer_agrees_with_global_candidate_contract():
    config = _config()
    result = DrMoagiVirtual3DAE(config).optimize()
    receipt = virtual_3d_tuning_receipt(config, result)

    expected = CandidateDecision.COMMIT if result.improved else CandidateDecision.ROLLBACK
    assert receipt.decision is expected
    assert receipt.verify()
    assert receipt.proposal.objective_after <= receipt.proposal.objective_before
    assert receipt.proposal.resource_usage.work_units == result.candidates_evaluated


def test_virtual_3d_receipt_is_deterministic():
    config = _config()
    first_result = DrMoagiVirtual3DAE(config).optimize()
    second_result = DrMoagiVirtual3DAE(config).optimize()

    first = virtual_3d_tuning_receipt(config, first_result)
    second = virtual_3d_tuning_receipt(config, second_result)

    assert first.receipt_hash == second.receipt_hash
    assert first.to_dict() == second.to_dict()
