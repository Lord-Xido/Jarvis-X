"""System-wide transaction membrane for the Dr Moagi research stack.

The master runtime composes two already separate trust decisions:

1. epistemic admission decides whether a generated hypothesis is sufficiently
   supported to become a verified research candidate;
2. the bounded system control plane decides whether a compiled execution plan
   is permitted, verifiable and committed as authoritative VM/task state.

The central invariant is therefore stronger than either inner layer alone:

    hypothesis -> epistemic admission -> authority execution -> audit -> commit

Epistemic admission does not itself grant execution authority. Likewise a VM
receipt cannot retroactively make an unverified hypothesis factual. Parameter
updates and permeation fields are exposed as *authoritative* by this wrapper
only when both gates succeed.

A caller must supply an explicit ``PlanBuilder`` adapter from the admitted
research result to bounded ``PlanCandidate`` objects. This module deliberately
does not invent a compiler from 3D scenes to canonical 64-bit bytecode.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Iterable, Protocol, Sequence

from .dr_moagi_codex import Latent, ScalarField
from .dr_moagi_epistemic import (
    EpistemicExecutionResult,
    EvidencePacket,
    ObservationPacket,
)
from .system_runtime import (
    DeterministicPlanner,
    ExecutionReceipt,
    PlanCandidate,
    ResourceBudget,
)


class EpistemicExecutor(Protocol):
    def execute(
        self,
        observation: ObservationPacket,
        *,
        evidence: Sequence[EvidencePacket],
        theta: Sequence[float] | None = None,
        theta_gradient: Sequence[float] | None = None,
        **kwargs: Any,
    ) -> EpistemicExecutionResult: ...


class AuthorityExecutor(Protocol):
    def execute_plans(
        self,
        *,
        request_id: str,
        candidates: Iterable[PlanCandidate],
        granted_capabilities: Iterable[str],
        budget: ResourceBudget | None = None,
        planner: DeterministicPlanner | None = None,
    ) -> ExecutionReceipt: ...


PlanBuilder = Callable[[EpistemicExecutionResult], Iterable[PlanCandidate]]


class MasterCommitStatus(str, Enum):
    """Outer system status after both trust boundaries are evaluated."""

    EPISTEMIC_REJECTED = "epistemic_rejected"
    AUTHORITY_REJECTED = "authority_rejected"
    COMMITTED = "committed"


@dataclass(frozen=True)
class MasterSystemResult:
    """One auditable outer transition of the Dr Moagi master runtime.

    The nested epistemic result is retained for diagnostic inspection, but the
    fields prefixed with ``authoritative`` / ``released`` below are the only
    outputs this layer treats as committed system state.
    """

    status: MasterCommitStatus
    epistemic: EpistemicExecutionResult
    execution: ExecutionReceipt | None
    authoritative_scene: ScalarField | None
    theta_before: Latent | None
    theta_after: Latent | None
    released_source_charge: ScalarField
    released_permeation_field: dict[tuple[float, float, float], complex]
    reason: str | None = None

    @property
    def committed(self) -> bool:
        return self.status is MasterCommitStatus.COMMITTED


class DrMoagiMasterRuntime:
    """Compose hypothesis verification with authoritative bounded execution."""

    def __init__(
        self,
        *,
        epistemic: EpistemicExecutor,
        authority: AuthorityExecutor,
        plan_builder: PlanBuilder,
        planner: DeterministicPlanner | None = None,
    ) -> None:
        self.epistemic = epistemic
        self.authority = authority
        self.plan_builder = plan_builder
        self.planner = planner

    def execute(
        self,
        observation: ObservationPacket,
        *,
        request_id: str,
        evidence: Sequence[EvidencePacket],
        granted_capabilities: Iterable[str],
        theta: Sequence[float] | None = None,
        theta_gradient: Sequence[float] | None = None,
        budget: ResourceBudget | None = None,
        **codex_kwargs: Any,
    ) -> MasterSystemResult:
        """Run one end-to-end candidate -> evidence -> authority transaction.

        Failure is closed at both membranes:

        * epistemic rejection prevents plan construction/execution entirely;
        * authority rejection/failure rolls parameter and output authority back
          to the pre-step values even if the inner epistemic gate admitted the
          research candidate.
        """

        epistemic_result = self.epistemic.execute(
            observation,
            evidence=evidence,
            theta=theta,
            theta_gradient=theta_gradient,
            **codex_kwargs,
        )

        if not epistemic_result.verdict.admitted:
            return self._noncommit(
                MasterCommitStatus.EPISTEMIC_REJECTED,
                epistemic_result,
                reason="epistemic admission rejected the generated hypothesis",
            )

        if epistemic_result.committed_scene is None:
            return self._noncommit(
                MasterCommitStatus.AUTHORITY_REJECTED,
                epistemic_result,
                reason="epistemic result was admitted without a committed research scene",
            )

        try:
            candidates = tuple(self.plan_builder(epistemic_result))
            if not candidates:
                return self._noncommit(
                    MasterCommitStatus.AUTHORITY_REJECTED,
                    epistemic_result,
                    reason="plan builder produced no bounded authority candidates",
                )
            receipt = self.authority.execute_plans(
                request_id=request_id,
                candidates=candidates,
                granted_capabilities=granted_capabilities,
                budget=budget or ResourceBudget(),
                planner=self.planner,
            )
        except Exception as exc:
            return self._noncommit(
                MasterCommitStatus.AUTHORITY_REJECTED,
                epistemic_result,
                reason=f"authority path failed closed: {type(exc).__name__}: {exc}",
            )

        if not receipt.committed:
            detail = receipt.error_message or receipt.error_type or receipt.status.value
            return self._noncommit(
                MasterCommitStatus.AUTHORITY_REJECTED,
                epistemic_result,
                execution=receipt,
                reason=f"authority execution did not commit: {detail}",
            )

        return MasterSystemResult(
            status=MasterCommitStatus.COMMITTED,
            epistemic=epistemic_result,
            execution=receipt,
            authoritative_scene=dict(epistemic_result.committed_scene),
            theta_before=epistemic_result.theta_before,
            theta_after=epistemic_result.theta_after,
            released_source_charge=dict(epistemic_result.released_source_charge),
            released_permeation_field=dict(epistemic_result.released_permeation_field),
            reason=None,
        )

    @staticmethod
    def _noncommit(
        status: MasterCommitStatus,
        epistemic: EpistemicExecutionResult,
        *,
        execution: ExecutionReceipt | None = None,
        reason: str,
    ) -> MasterSystemResult:
        return MasterSystemResult(
            status=status,
            epistemic=epistemic,
            execution=execution,
            authoritative_scene=None,
            theta_before=epistemic.theta_before,
            theta_after=epistemic.theta_before,
            released_source_charge={},
            released_permeation_field={},
            reason=reason,
        )
