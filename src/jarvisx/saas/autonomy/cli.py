"""Command-line entry point for the autonomic enterprise control plane."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from .control import AutonomicEnterpriseController
from .twin import EnterpriseState, Scenario


def _load_json(value: str):
    if value.startswith("@"):
        with open(value[1:], "r", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(value)


def main() -> None:
    parser = argparse.ArgumentParser(prog="drmoagi-autonomy")
    subparsers = parser.add_subparsers(dest="command", required=True)

    simulate = subparsers.add_parser("simulate", help="rank enterprise scenarios")
    simulate.add_argument("--tenant", required=True)
    simulate.add_argument("--subject", required=True)
    simulate.add_argument("--state", required=True, help="JSON or @file")
    simulate.add_argument("--scenarios", required=True, help="JSON array or @file")
    simulate.add_argument("--risk-aversion", type=float, default=1.0)

    serve = subparsers.add_parser("serve", help="run the FastAPI control plane")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8090)

    args = parser.parse_args()
    if args.command == "serve":
        import uvicorn

        uvicorn.run("jarvisx.saas.autonomy.app:app", host=args.host, port=args.port)
        return

    state = EnterpriseState(**_load_json(args.state))
    scenarios = [Scenario(**item) for item in _load_json(args.scenarios)]
    controller = AutonomicEnterpriseController()
    ranked = controller.propose(
        tenant_id=args.tenant,
        subject=args.subject,
        state=state,
        scenarios=scenarios,
        risk_aversion=args.risk_aversion,
    )
    print(
        json.dumps(
            [
                {
                    "proposal_id": proposal.proposal_id,
                    "scenario": asdict(proposal.scenario),
                    "simulation": asdict(proposal.simulation),
                    "utility": proposal.utility,
                }
                for proposal in ranked
            ],
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
