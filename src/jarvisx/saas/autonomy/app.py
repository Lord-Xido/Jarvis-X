"""Standalone FastAPI service for the autonomic enterprise control plane."""

from __future__ import annotations

import os
import secrets
from dataclasses import asdict
from typing import Dict, List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from .control import ActionProposal, AutonomicEnterpriseController
from .policy import CommitPolicyEngine
from .twin import EnterpriseState, Scenario


def _controller() -> AutonomicEnterpriseController:
    configured = os.getenv("DM_AUTONOMY_SIGNING_KEY")
    signing_key = configured if configured else secrets.token_bytes(32)
    return AutonomicEnterpriseController(
        policy=CommitPolicyEngine(signing_key=signing_key)
    )


app = FastAPI(
    title="Dr Moagi Autonomic Enterprise OS",
    version="2.1.0-experimental",
    description=(
        "Signed-authority digital-twin and commit-time-authorized enterprise "
        "control plane."
    ),
)
controller = _controller()
proposals: Dict[str, ActionProposal] = {}


def authorize_service(
    x_autonomy_token: Optional[str] = Header(default=None),
) -> None:
    expected = os.getenv("DM_AUTONOMY_TOKEN", "")
    insecure = os.getenv("DM_AUTONOMY_ALLOW_INSECURE", "0") == "1"
    if insecure:
        return
    if not expected:
        raise HTTPException(status_code=503, detail="autonomy token is not configured")
    if not secrets.compare_digest(x_autonomy_token or "", expected):
        raise HTTPException(status_code=401, detail="invalid autonomy token")


def authorize_issuer(
    x_autonomy_issuer_token: Optional[str] = Header(default=None),
) -> None:
    insecure = os.getenv("DM_AUTONOMY_ALLOW_INSECURE", "0") == "1"
    if insecure:
        return
    expected = os.getenv("DM_AUTONOMY_ISSUER_TOKEN", "")
    service_token = os.getenv("DM_AUTONOMY_TOKEN", "")
    signing_key = os.getenv("DM_AUTONOMY_SIGNING_KEY", "")
    if not expected or not signing_key:
        raise HTTPException(
            status_code=503,
            detail="authority issuer and signing key are not configured",
        )
    if expected == service_token:
        raise HTTPException(
            status_code=503,
            detail="issuer token must be distinct from the service token",
        )
    if not secrets.compare_digest(x_autonomy_issuer_token or "", expected):
        raise HTTPException(status_code=401, detail="invalid authority issuer token")


class StateRequest(BaseModel):
    tenant_id: str
    subject: str
    cash_minor: int
    monthly_revenue_minor: int
    monthly_cost_minor: int
    receivables_minor: int = 0
    pipeline_minor: int = 0
    delivery_health: float = Field(ge=0, le=1)
    finance_health: float = Field(ge=0, le=1)
    governance_health: float = Field(ge=0, le=1)
    churn_rate: float = Field(ge=0, le=1)
    collection_rate: float = Field(ge=0, le=1)
    scenarios: List[dict]
    approval_epoch: int = 0


class WitnessIssueRequest(BaseModel):
    proposal_id: str
    roles: List[str] = Field(default_factory=list)
    ttl_seconds: int = Field(default=300, ge=1, le=3600)


class ApprovalIssueRequest(BaseModel):
    proposal_id: str
    approver: str
    ttl_seconds: int = Field(default=300, ge=1, le=3600)


class CommitRequestModel(BaseModel):
    proposal_id: str
    witness_token: str
    approval_tokens: List[str] = Field(default_factory=list)


@app.get("/health")
def health():
    return {"status": "ok", "service": "dr-moagi-autonomic-enterprise-os"}


@app.post("/v2/autonomy/proposals", dependencies=[Depends(authorize_service)])
def create_proposals(request: StateRequest):
    state = EnterpriseState(
        cash_minor=request.cash_minor,
        monthly_revenue_minor=request.monthly_revenue_minor,
        monthly_cost_minor=request.monthly_cost_minor,
        receivables_minor=request.receivables_minor,
        pipeline_minor=request.pipeline_minor,
        delivery_health=request.delivery_health,
        finance_health=request.finance_health,
        governance_health=request.governance_health,
        churn_rate=request.churn_rate,
        collection_rate=request.collection_rate,
    )
    ranked = controller.propose(
        tenant_id=request.tenant_id,
        subject=request.subject,
        state=state,
        scenarios=[Scenario(**scenario) for scenario in request.scenarios],
        approval_epoch=request.approval_epoch,
    )
    for proposal in ranked:
        proposals[proposal.proposal_id] = proposal
    return [
        {
            "proposal_id": item.proposal_id,
            "scenario": asdict(item.scenario),
            "simulation": asdict(item.simulation),
            "utility": item.utility,
            "commit_request": {
                **item.commit_request.__dict__,
                "required_roles": sorted(item.commit_request.required_roles),
                "approval_tokens": [],
            },
        }
        for item in ranked
    ]


@app.post(
    "/v2/autonomy/authority/witnesses",
    dependencies=[Depends(authorize_issuer)],
)
def issue_witness(request: WitnessIssueRequest):
    proposal = proposals.get(request.proposal_id)
    if proposal is None:
        raise HTTPException(status_code=404, detail="proposal not found")
    return {
        "witness_token": controller.policy.issue_witness(
            proposal.commit_request,
            roles=request.roles,
            ttl_seconds=request.ttl_seconds,
        )
    }


@app.post(
    "/v2/autonomy/authority/approvals",
    dependencies=[Depends(authorize_issuer)],
)
def issue_approval(request: ApprovalIssueRequest):
    proposal = proposals.get(request.proposal_id)
    if proposal is None:
        raise HTTPException(status_code=404, detail="proposal not found")
    return {
        "approval_token": controller.policy.issue_approval(
            proposal.commit_request,
            approver=request.approver,
            ttl_seconds=request.ttl_seconds,
        )
    }


@app.post("/v2/autonomy/commit", dependencies=[Depends(authorize_service)])
def commit(request: CommitRequestModel):
    proposal = proposals.get(request.proposal_id)
    if proposal is None:
        raise HTTPException(status_code=404, detail="proposal not found")
    try:
        event_id = controller.commit(
            proposal,
            request.witness_token,
            approvals=request.approval_tokens,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    return {
        "event_id": event_id,
        "ledger_version": controller.ledger.version(proposal.tenant_id),
        "merkle_root": controller.ledger.merkle_root(proposal.tenant_id),
    }
