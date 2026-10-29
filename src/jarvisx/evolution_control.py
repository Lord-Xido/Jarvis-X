"""JARVIS-X deterministic architecture evolution *planner*, not an autonomous deployer.

ADR-016/017/031 style: typed inventory -> candidate -> admissibility ->
shadow evaluation -> immutable receipt. No mutation of source, financial, medical,
vehicle, model weights, network services, deployed infrastructure or git refs.
Python 3.10+; standard library only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

INVENTORY_PATH = Path(__file__).with_name("evolution_inventory.json")
SCHEMA = "jarvisx.evolution.v1"
GATES = frozenset({
    "typed_state", "geometry_profile", "resource_bounds", "deterministic_replay",
    "roundtrip", "rollback", "integrity", "policy", "independent_tests",
    "human_approval", "security_review", "domain_review", "benchmark_receipt",
})
MANDATORY = frozenset({"typed_state", "resource_bounds", "integrity", "policy", "rollback", "human_approval"})
RISK_GATES = {
    "normal": frozenset(),
    "elevated": frozenset({"security_review", "independent_tests"}),
    "restricted": frozenset({"domain_review", "security_review", "independent_tests"}),
}
PROFILES = ("minimal", "balanced", "throughput")
# All rates and scores here are heuristic planning scores, never measured performance.
BRICK_OPTIONS = (1, 2, 4, 8, 16)
REPETITION_OPTIONS = (1, 2, 4, 8)
CYCLE_OPTIONS = (4, 8, 16, 32, 64)


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def load_inventory(path: Path = INVENTORY_PATH) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    validate_inventory(obj)
    return obj


def validate_inventory(obj: dict[str, Any]) -> None:
    if obj.get("schema") != SCHEMA:
        raise ValueError("unsupported inventory schema")
    nodes = obj.get("components")
    tasks = obj.get("tasks")
    if not isinstance(nodes, list) or not nodes or not isinstance(tasks, list) or not tasks:
        raise ValueError("nonempty components and tasks required")
    nids = [c["id"] for c in nodes]
    tids = [t["id"] for t in tasks]
    if len(set(nids)) != len(nids) or len(set(tids)) != len(tids):
        raise ValueError("duplicate ids")
    valid_maturity = {"stable-reference", "alpha", "reference-laboratory", "integration-candidate", "demonstration", "specification", "future-work", "restricted"}
    for c in nodes:
        if c["maturity"] not in valid_maturity or c["authority"] not in ("read-only", "review-required", "no-auto-access"):
            raise ValueError("invalid maturity or authority")
        if c["authority"] == "no-auto-access" and c["maturity"] != "restricted":
            raise ValueError("restricted component mismatch")
    edges = {}
    for t in tasks:
        if t["risk"] not in RISK_GATES or not (set(t["required_gates"]) <= GATES):
            raise ValueError("invalid task policy")
        if t["component"] not in nids or t["mode"] not in ("shadow", "manual-only"):
            raise ValueError("invalid task component/mode")
        if not set(t["required_gates"]) >= MANDATORY | RISK_GATES[t["risk"]]:
            raise ValueError("task missing mandatory gates")
        edges[t["id"]] = tuple(t["requires"])
        if t["id"] in t["requires"]:
            raise ValueError("task self-dependency")
    if any(dep not in edges for parents in edges.values() for dep in parents):
        raise ValueError("unknown task dependency")
    seen: set[str] = set()
    visiting: set[str] = set()

    def dfs(n: str) -> None:
        if n in visiting:
            raise ValueError("dependency cycle")
        if n in seen:
            return
        visiting.add(n)
        for parent in edges[n]:
            dfs(parent)
        visiting.remove(n)
        seen.add(n)
    for n in edges:
        dfs(n)


@dataclass(frozen=True)
class Proposal:
    schema: str
    inventory_sha256: str
    profile: str
    task_ids: tuple[str, ...]
    resident_brick_ceiling: int
    verification_repetitions: int
    cycle_ceiling: int
    gates: tuple[str, ...]
    mode: str = "shadow-only"


def check_proposal(p: Proposal, inventory: dict[str, Any]) -> tuple[bool, tuple[str, ...]]:
    errors: list[str] = []
    lookup = {t["id"]: t for t in inventory["tasks"]}
    if p.schema != SCHEMA or p.inventory_sha256 != digest(inventory):
        errors.append("inventory identity/schema mismatch")
    if p.mode != "shadow-only" or p.profile not in PROFILES:
        errors.append("unauthorized execution mode/profile")
    if p.resident_brick_ceiling not in BRICK_OPTIONS or p.verification_repetitions not in REPETITION_OPTIONS or p.cycle_ceiling not in CYCLE_OPTIONS:
        errors.append("resource or verification ceiling outside policy")
    if len(p.task_ids) != len(set(p.task_ids)) or not p.task_ids:
        errors.append("tasks empty or repeated")
    unknown = set(p.task_ids) - set(lookup)
    if unknown:
        errors.append("unknown task ids")
    selected = set(p.task_ids)
    for name in p.task_ids:
        t = lookup.get(name)
        if not t:
            continue
        if t["mode"] != "shadow":
            errors.append(f"manual-only task selected: {name}")
        if not set(t["requires"]) <= selected:
            errors.append(f"missing task prerequisites for: {name}")
    expected = set(MANDATORY)
    for name in p.task_ids:
        t = lookup.get(name)
        if t:
            expected.update(t["required_gates"])
    if not expected <= set(p.gates) or not set(p.gates) <= GATES or len(set(p.gates)) != len(p.gates):
        errors.append("mandatory verification gates absent/unknown/repeated")
    return not errors, tuple(sorted(set(errors)))


def topological_shadow_tasks(inventory: dict[str, Any]) -> tuple[str, ...]:
    tasks = {t["id"]: t for t in inventory["tasks"] if t["mode"] == "shadow"}
    result: list[str] = []
    while len(result) < len(tasks):
        ready = sorted((t for t in tasks.values() if t["id"] not in result and set(t["requires"]) <= set(result)), key=lambda t: (-t["priority"], t["id"]))
        if not ready:
            raise ValueError("non-executable shadow task dependency")
        result.append(ready[0]["id"])
    return tuple(result)


def make_proposal(inventory: dict[str, Any], *, profile: str, count: int, bricks: int, repeats: int, cycles: int) -> Proposal:
    ordered = topological_shadow_tasks(inventory)
    ids = ordered[:count]
    lookup = {t["id"]: t for t in inventory["tasks"]}
    gates = set(MANDATORY)
    for item in ids:
        gates.update(lookup[item]["required_gates"])
    return Proposal(SCHEMA, digest(inventory), profile, ids, bricks, repeats, cycles, tuple(sorted(gates)))


def shadow_score(p: Proposal, inventory: dict[str, Any]) -> int:
    """Deterministic internal *planning* utility only, not a throughput estimate."""
    ok, reasons = check_proposal(p, inventory)
    if not ok:
        raise ValueError("inadmissible candidate: " + ", ".join(reasons))
    lookup = {t["id"]: t for t in inventory["tasks"]}
    # Evidence-first: high-priority prerequisite closure dominates larger ceilings.
    coverage = sum(lookup[t]["priority"] for t in p.task_ids)
    verification = p.verification_repetitions * 3
    resource_cost = p.resident_brick_ceiling * 2 + p.cycle_ceiling // 4
    plan_complexity = len(p.task_ids) * len(p.task_ids)
    return 12 * coverage + verification - resource_cost - plan_complexity


def evolve(inventory: dict[str, Any], iterations: int = 1000, seed: int = 20261027) -> dict[str, Any]:
    """Bounded, deterministic policy search. Never edits code or applies a candidate."""
    if not 1 <= iterations <= 10000:
        raise ValueError("iterations must be 1..10000")
    ordered = topological_shadow_tasks(inventory)
    rng = random.Random(seed)
    best: Proposal | None = None
    best_score: int | None = None
    accepted = 0
    for k in range(iterations):
        proposal = make_proposal(inventory,
            profile=PROFILES[k % len(PROFILES)], count=rng.randint(1, len(ordered)),
            bricks=rng.choice(BRICK_OPTIONS), repeats=rng.choice(REPETITION_OPTIONS),
            cycles=rng.choice(CYCLE_OPTIONS))
        ok, _ = check_proposal(proposal, inventory)
        if not ok:
            continue
        accepted += 1
        score = shadow_score(proposal, inventory)
        if best_score is None or score > best_score or (score == best_score and canonical(asdict(proposal)) < canonical(asdict(best))):
            best_score, best = score, proposal
    assert best is not None
    tasks = {t["id"]: t for t in inventory["tasks"]}
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "finding": "shadow-plan-only: does not modify any runtime, service, code, GitHub branch, financial/medical/vehicle system, or protected asset",
        "inventory_sha256": digest(inventory),
        "search": {"iterations": iterations, "seed": seed, "admissible_candidates": accepted, "objective": "heuristic prerequisite/evidence coverage minus resource and plan complexity, NOT measured acceleration"},
        "candidate": json.loads(canonical(asdict(best))),
        "candidate_sha256": digest(asdict(best)),
        "shadow_score": best_score,
        "recommended_sequence": [{"id": n, "purpose": tasks[n]["purpose"], "required_gates": tasks[n]["required_gates"], "evidence_required": tasks[n]["evidence_required"]} for n in best.task_ids],
        "blocked_manual_tasks": [t["id"] for t in inventory["tasks"] if t["mode"] == "manual-only"],
        "verification": {"policy_admissible": True, "tests_executed": False, "changes_applied": False, "production_authorized": False},
    }
    report["receipt_sha256"] = digest(report)
    return report


def verify_receipt(report: dict[str, Any], inventory: dict[str, Any]) -> tuple[bool, tuple[str, ...]]:
    issues = []
    item = dict(report)
    claimed = item.pop("receipt_sha256", None)
    if claimed != digest(item):
        issues.append("receipt hash mismatch")
    try:
        p = Proposal(**item["candidate"])
        good, problems = check_proposal(p, inventory)
        if not good:
            issues.extend(problems)
        if item.get("candidate_sha256") != digest(asdict(p)):
            issues.append("candidate hash mismatch")
        if item.get("shadow_score") != shadow_score(p, inventory):
            issues.append("planning score mismatch")
        if item.get("verification") != {"policy_admissible": True, "tests_executed": False, "changes_applied": False, "production_authorized": False}:
            issues.append("incorrect claimed execution status")
        # Independently recompute search result, preventing a re-hashed forged plan.
        search = item["search"]
        expected = evolve(inventory, search["iterations"], search["seed"])
        if item != {k: v for k, v in expected.items() if k != "receipt_sha256"}:
            issues.append("not a reproducible deterministic search result")
    except (KeyError, TypeError, ValueError, AssertionError) as exc:
        issues.append("invalid proposal: " + str(exc))
    return not issues, tuple(sorted(set(issues)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="JARVIS-X review-only architecture evolution planner")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("inventory", help="print the curated architecture inventory")
    plan = sub.add_parser("plan", help="search candidate plans with hard policy gates")
    plan.add_argument("--iterations", type=int, default=1000)
    plan.add_argument("--seed", type=int, default=20261027)
    plan.add_argument("--output", type=Path, default=None)
    verify = sub.add_parser("verify", help="independently verify a plan receipt and replay its search")
    verify.add_argument("receipt", type=Path)
    args = parser.parse_args(argv)
    inv = load_inventory()
    if args.command == "inventory":
        print(json.dumps(inv, indent=2))
        return 0
    if args.command == "plan":
        result = evolve(inv, args.iterations, args.seed)
        rendered = json.dumps(result, sort_keys=True, indent=2) + "\n"
        if args.output:
            args.output.write_text(rendered, encoding="utf-8")
        else:
            print(rendered, end="")
        return 0
    receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
    ok, issues = verify_receipt(receipt, inv)
    print(json.dumps({"verified": ok, "issues": issues}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
