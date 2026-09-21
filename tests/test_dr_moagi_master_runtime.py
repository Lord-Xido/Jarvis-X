from __future__ import annotations

from jarvisx.dr_moagi_epistemic import (
    EpistemicExecutionResult,
    EpistemicVerdict,
    EvidenceKind,
    ObservationPacket,
)
from jarvisx.dr_moagi_master_runtime import (
    DrMoagiMasterRuntime,
    MasterCommitStatus,
)
from jarvisx.system_runtime import (
    ExecutionReceipt,
    ExecutionStatus,
    PlanCandidate,
)


POINT = (0.0, 0.0, 0.0)


def observation():
    return ObservationPacket(
        scene={POINT: 1.0},
        source_id="sensor-1",
        kind=EvidenceKind.SENSOR,
    )


def epistemic_result(*, admitted: bool) -> EpistemicExecutionResult:
    verdict = EpistemicVerdict(
        admitted=admitted,
        reasons=() if admitted else ("rejected fixture",),
        observation_nrmse=0.0 if admitted else 1.0,
        anchor_nrmse=0.0 if admitted else 1.0,
        max_evidence_nrmse=0.0 if admitted else None,
        independent_evidence_count=1 if admitted else 0,
        observation_digest="observation",
        anchor_digest="anchor",
        evidence_digests=("evidence",) if admitted else (),
    )
    return EpistemicExecutionResult(
        verdict=verdict,
        hypothesis_scene={POINT: 1.0},
        hypothesis_latent=(1.0,),
        committed_scene={POINT: 1.0} if admitted else None,
        theta_before=(1.0,),
        theta_after=(0.9,) if admitted else (1.0,),
        learning_committed=admitted,
        released_source_charge={POINT: 2.0} if admitted else {},
        released_permeation_field={POINT: 3.0 + 0.0j} if admitted else {},
        fixed_point_iterations=4,
        fixed_point_converged=True,
        virtual_depth_label="1000000^1000000",
    )


def execution_receipt(*, committed: bool) -> ExecutionReceipt:
    return ExecutionReceipt(
        request_id="req-1",
        request_fingerprint="fingerprint",
        status=ExecutionStatus.COMMITTED if committed else ExecutionStatus.REJECTED,
        committed=committed,
        cycles=1 if committed else 0,
        state=(("R1", 1),) if committed else (),
        state_hash="state" if committed else None,
        vm_ledger_head="vm" if committed else None,
        audit_head="audit",
        plan_id="plan-1",
        error_type=None if committed else "PolicyRejection",
        error_message=None if committed else "fixture rejection",
    )


class FakeEpistemic:
    def __init__(self, result: EpistemicExecutionResult):
        self.result = result
        self.calls = 0

    def execute(self, observation, **kwargs):
        self.calls += 1
        return self.result


class FakeAuthority:
    def __init__(self, receipt: ExecutionReceipt):
        self.receipt = receipt
        self.calls = 0
        self.last_candidates = ()

    def execute_plans(self, **kwargs):
        self.calls += 1
        self.last_candidates = tuple(kwargs["candidates"])
        return self.receipt


def plans(_result):
    return (
        PlanCandidate(
            plan_id="plan-1",
            program=(0,),
            quality=1.0,
            cost=0.0,
            latency=0.0,
            risk=0.0,
        ),
    )


def test_epistemic_rejection_never_reaches_authority_plane():
    epistemic = FakeEpistemic(epistemic_result(admitted=False))
    authority = FakeAuthority(execution_receipt(committed=True))
    builder_calls = []

    def builder(result):
        builder_calls.append(result)
        return plans(result)

    runtime = DrMoagiMasterRuntime(
        epistemic=epistemic,
        authority=authority,
        plan_builder=builder,
    )
    result = runtime.execute(
        observation(),
        request_id="req-1",
        evidence=(),
        granted_capabilities=("vm.execute",),
    )

    assert result.status is MasterCommitStatus.EPISTEMIC_REJECTED
    assert not result.committed
    assert authority.calls == 0
    assert builder_calls == []
    assert result.authoritative_scene is None
    assert result.released_source_charge == {}
    assert result.released_permeation_field == {}
    assert result.theta_after == result.theta_before == (1.0,)


def test_authority_rejection_rolls_back_epistemic_learning_and_release():
    epistemic = FakeEpistemic(epistemic_result(admitted=True))
    authority = FakeAuthority(execution_receipt(committed=False))
    runtime = DrMoagiMasterRuntime(
        epistemic=epistemic,
        authority=authority,
        plan_builder=plans,
    )

    result = runtime.execute(
        observation(),
        request_id="req-1",
        evidence=(),
        granted_capabilities=("vm.execute",),
    )

    assert result.status is MasterCommitStatus.AUTHORITY_REJECTED
    assert not result.committed
    assert authority.calls == 1
    assert result.epistemic.learning_committed
    assert result.epistemic.released_source_charge == {POINT: 2.0}
    assert result.theta_after == (1.0,)
    assert result.authoritative_scene is None
    assert result.released_source_charge == {}
    assert result.released_permeation_field == {}


def test_dual_admission_exposes_authoritative_scene_learning_and_permeation():
    epistemic = FakeEpistemic(epistemic_result(admitted=True))
    authority = FakeAuthority(execution_receipt(committed=True))
    runtime = DrMoagiMasterRuntime(
        epistemic=epistemic,
        authority=authority,
        plan_builder=plans,
    )

    result = runtime.execute(
        observation(),
        request_id="req-1",
        evidence=(),
        granted_capabilities=("vm.execute",),
    )

    assert result.status is MasterCommitStatus.COMMITTED
    assert result.committed
    assert authority.calls == 1
    assert result.authoritative_scene == {POINT: 1.0}
    assert result.theta_before == (1.0,)
    assert result.theta_after == (0.9,)
    assert result.released_source_charge == {POINT: 2.0}
    assert result.released_permeation_field == {POINT: 3.0 + 0.0j}


def test_empty_plan_set_fails_closed_without_authority_execution():
    epistemic = FakeEpistemic(epistemic_result(admitted=True))
    authority = FakeAuthority(execution_receipt(committed=True))
    runtime = DrMoagiMasterRuntime(
        epistemic=epistemic,
        authority=authority,
        plan_builder=lambda _result: (),
    )

    result = runtime.execute(
        observation(),
        request_id="req-1",
        evidence=(),
        granted_capabilities=("vm.execute",),
    )

    assert result.status is MasterCommitStatus.AUTHORITY_REJECTED
    assert authority.calls == 0
    assert result.theta_after == (1.0,)
    assert result.released_permeation_field == {}
    assert "no bounded authority candidates" in (result.reason or "")


def test_plan_builder_exception_fails_closed():
    epistemic = FakeEpistemic(epistemic_result(admitted=True))
    authority = FakeAuthority(execution_receipt(committed=True))

    def broken_builder(_result):
        raise RuntimeError("compiler fixture failed")

    runtime = DrMoagiMasterRuntime(
        epistemic=epistemic,
        authority=authority,
        plan_builder=broken_builder,
    )
    result = runtime.execute(
        observation(),
        request_id="req-1",
        evidence=(),
        granted_capabilities=("vm.execute",),
    )

    assert result.status is MasterCommitStatus.AUTHORITY_REJECTED
    assert authority.calls == 0
    assert result.theta_after == (1.0,)
    assert "compiler fixture failed" in (result.reason or "")
