"""Risk-constrained autonomic enterprise controller."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from typing import Iterable, Optional, Tuple

from .events import CausalEventLedger
from .policy import CommitPolicyEngine, CommitRequest
from .twin import DigitalTwin, EnterpriseState, Scenario, SimulationResult


def _proposal_id(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return "proposal-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class ActionProposal:
    proposal_id: str
    tenant_id: str
    scenario: Scenario
    simulation: SimulationResult
    utility: float
    commit_request: CommitRequest


class AutonomicEnterpriseController:
    """Observe -> simulate -> rank -> authorize -> commit."""

    def __init__(
        self,
        ledger: Optional[CausalEventLedger] = None,
        policy: Optional[CommitPolicyEngine] = None,
        twin: Optional[DigitalTwin] = None,
    ) -> None:
        self.ledger = ledger or CausalEventLedger()
        self.policy = policy or CommitPolicyEngine()
        self.twin = twin or DigitalTwin()

    def propose(
        self,
        *,
        tenant_id: str,
        subject: str,
        state: EnterpriseState,
        scenarios: Iterable[Scenario],
        resource: str = "enterprise",
        approval_epoch: int = 0,
        risk_aversion: float = 1.0,
    ) -> Tuple[ActionProposal, ...]:
        scenario_set = tuple(scenarios)
        if not scenario_set:
            raise ValueError("at least one scenario is required")
        ranked = self.twin.rank(state, scenario_set, risk_aversion=risk_aversion)
        proposals = []
        version = self.ledger.version(tenant_id)
        scenario_lookup = {scenario.name: scenario for scenario in scenario_set}
        if len(scenario_lookup) != len(scenario_set):
            raise ValueError("scenario names must be unique")
        for simulation in ranked:
            scenario = scenario_lookup[simulation.scenario]
            risk = max(
                0.0,
                min(
                    1.0,
                    (1.0 - simulation.survival_probability)
                    + simulation.expected_shortfall_minor
                    / max(abs(state.cash_minor) + 1, 1),
                ),
            )
            utility = (
                simulation.cumulative_profit_minor
                - risk_aversion * simulation.expected_shortfall_minor
                + simulation.health * 100_000
            )
            bindings = {
                "scenario": asdict(scenario),
                "state": asdict(state),
                "simulation": {
                    "profit": simulation.cumulative_profit_minor,
                    "shortfall": simulation.expected_shortfall_minor,
                    "survival": simulation.survival_probability,
                },
            }
            request = CommitRequest(
                tenant_id=tenant_id,
                subject=subject,
                action="enterprise.apply_scenario",
                resource=resource,
                state_version=version,
                approval_epoch=approval_epoch,
                bindings=bindings,
                estimated_cost_minor=max(0, scenario.one_off_cost_minor),
                risk=risk,
                required_roles=frozenset({"tenant_owner"}),
                required_approvals=1 if risk > 0.2 else 0,
            )
            proposals.append(
                ActionProposal(
                    proposal_id=_proposal_id(
                        {
                            "tenant": tenant_id,
                            "version": version,
                            "scenario": bindings,
                        }
                    ),
                    tenant_id=tenant_id,
                    scenario=scenario,
                    simulation=simulation,
                    utility=utility,
                    commit_request=request,
                )
            )
        return tuple(sorted(proposals, key=lambda item: item.utility, reverse=True))

    def commit(
        self,
        proposal: ActionProposal,
        witness_token: str,
        *,
        approvals: Iterable[str] = (),
    ) -> str:
        request = replace(
            proposal.commit_request,
            approval_tokens=tuple(approvals),
        )
        if self.ledger.version(proposal.tenant_id) != request.state_version:
            raise PermissionError("failed:state_version")
        decision = self.policy.decide(request, witness_token)
        if not decision.allowed:
            raise PermissionError(decision.reason)
        event = self.ledger.append(
            tenant_id=proposal.tenant_id,
            stream="autonomy",
            event_type="autonomy.action_committed",
            payload={
                "proposal_id": proposal.proposal_id,
                "scenario": proposal.scenario.name,
                "utility": proposal.utility,
                "proof_hash": decision.proof_hash,
                "approval_ids": list(decision.approval_ids),
                "expected_shortfall_minor": proposal.simulation.expected_shortfall_minor,
                "survival_probability": proposal.simulation.survival_probability,
            },
            actor=request.subject,
            causation_id=decision.witness_id,
            expected_version=request.state_version,
        )
        return event.event_id
