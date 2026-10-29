"""Fail-closed ADR-016 release-evidence gate.

The reference system is intentionally NOT production ready. A green unit test
or package build cannot substitute for live authority adapters, isolation,
independent security assessment and real failure-mode qualification.
"""
from __future__ import annotations

import argparse
import json

SCHEMA = "JarvisX/ADR016/ProductionGate1"

RESEARCH_EVIDENCE = (
    "typed shadow frames and deterministic reconstruction oracles",
    "SQLite WAL backed one-store commit with process kill fault injection",
    "four separately persisted local SQLite participant states",
    "recoverable coordinator commit decision with idempotent local applies",
    "authenticated AF_UNIX HMAC RPC with bounded signed messages",
    "participant durable decision and abort intent verification",
    "coordinator-gated read that refuses incomplete commits",
)

REQUIRED_PRODUCTION_EVIDENCE = {
    "actual_live_backend_authority": (
        "Each independent VM, OS, ANN and geometry service implements a "
        "durable authoritative prepare/apply/recover contract"
    ),
    "global_visibility_or_declared_saga": (
        "Atomic read isolation across live services is demonstrated, or each "
        "externally visible operation is explicitly a compensatable saga"
    ),
    "independent_operator_authentication": (
        "Identity-bound operator authorization separate from transport HMAC "
        "and in-process approve=True with revocation and audit"
    ),
    "coordinator_leadership": (
        "Multi-host leader fencing, partition recovery, election, timeout and "
        "split-brain safety are independently qualified"
    ),
    "physical_storage_recovery": (
        "Disk-full, torn-write and power-loss recovery are tested on target "
        "filesystems and deployment storage, not only os._exit simulations"
    ),
    "independent_security_review": (
        "Threat model, penetration testing and third-party security sign-off "
        "completed for deployment configuration and code"
    ),
    "operations_and_rollback": (
        "Signed release provenance, vulnerability review, monitoring, "
        "incident response and deployment/rollback runbook exercised"
    ),
    "external_evidence": (
        "Evidence outside the simulated backends is obtained for any live "
        "operation; missing evidence is never marked PASS"
    ),
}


def report(commit_sha: str = "") -> dict:
    return {
        "schema": SCHEMA,
        "commit_sha": commit_sha,
        "scope": "local isolated shadow research reference",
        "research_reference_evidence": list(RESEARCH_EVIDENCE),
        "production_readiness": "BLOCKED",
        "production_deployment_allowed": False,
        "required_unverified": [
            {"gate": key, "status": "BLOCKED", "required_evidence": value}
            for key, value in REQUIRED_PRODUCTION_EVIDENCE.items()
        ],
        "safety_invariant": (
            "No live backend deployment or issue #425 completion can be "
            "inferred from shadow tests or green CI alone."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-production", action="store_true")
    parser.add_argument("--commit", default="")
    args = parser.parse_args()
    output = report(args.commit)
    print(json.dumps(output, indent=2, sort_keys=True))
    if args.require_production:
        return 2  # Always fail closed until required evidence has real verification.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
