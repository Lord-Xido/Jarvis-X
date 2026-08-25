#!/usr/bin/env python3
"""Read-only GitHub portfolio audit for the Lord-Xido/Jarvis-X control plane.

The script intentionally proposes actions but does not mutate GitHub state. Writes remain
behind an explicit human/connector review boundary.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any, Iterable

API = "https://api.github.com"


class GitHubError(RuntimeError):
    pass


def request_json(path: str, token: str | None) -> Any:
    url = path if path.startswith("http") else f"{API}{path}"
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "jarvis-x-account-control-plane/1",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise GitHubError(f"GitHub API {exc.code} for {url}: {body}") from exc
    except urllib.error.URLError as exc:
        raise GitHubError(f"GitHub API request failed for {url}: {exc}") from exc


def paginate(path: str, token: str | None, per_page: int = 100) -> Iterable[dict[str, Any]]:
    sep = "&" if "?" in path else "?"
    page = 1
    while True:
        batch = request_json(f"{path}{sep}per_page={per_page}&page={page}", token)
        if not isinstance(batch, list):
            raise GitHubError(f"Expected list response for {path}")
        for item in batch:
            yield item
        if len(batch) < per_page:
            return
        page += 1


@dataclass(frozen=True)
class PullAudit:
    number: int
    title: str
    draft: bool
    base: str
    head: str
    updated_at: str
    ahead_by: int | None
    behind_by: int | None
    comparison_status: str | None
    disposition: str


@dataclass(frozen=True)
class RepoAudit:
    name: str
    full_name: str
    private: bool
    archived: bool
    fork: bool
    default_branch: str
    description: str | None
    open_issues_count: int
    stars: int
    forks: int
    pushed_at: str | None
    open_prs: tuple[PullAudit, ...]
    recommendations: tuple[str, ...]


def compare(owner: str, repo: str, base: str, head: str, token: str | None) -> dict[str, Any] | None:
    base_q = urllib.parse.quote(base, safe="")
    head_q = urllib.parse.quote(head, safe="")
    try:
        data = request_json(f"/repos/{owner}/{repo}/compare/{base_q}...{head_q}", token)
    except GitHubError:
        return None
    return data if isinstance(data, dict) else None


def classify_pr(pr: dict[str, Any], comparison: dict[str, Any] | None) -> str:
    if pr.get("draft"):
        if comparison and int(comparison.get("behind_by", 0)) > 0:
            return "draft-needs-sync"
        return "draft-review"
    if comparison is None:
        return "review"
    behind = int(comparison.get("behind_by", 0))
    status = comparison.get("status")
    if behind > 0 or status == "diverged":
        return "sync-before-merge"
    return "integration-candidate"


def audit_repo(owner: str, repo: dict[str, Any], token: str | None) -> RepoAudit:
    name = repo["name"]
    pulls_raw = list(paginate(f"/repos/{owner}/{name}/pulls?state=open", token, per_page=100))
    pulls: list[PullAudit] = []
    for pr in pulls_raw:
        base = pr["base"]["ref"]
        head = pr["head"]["ref"]
        cmp = compare(owner, name, base, head, token)
        pulls.append(
            PullAudit(
                number=int(pr["number"]),
                title=str(pr["title"]),
                draft=bool(pr.get("draft")),
                base=base,
                head=head,
                updated_at=str(pr.get("updated_at") or ""),
                ahead_by=int(cmp["ahead_by"]) if cmp and cmp.get("ahead_by") is not None else None,
                behind_by=int(cmp["behind_by"]) if cmp and cmp.get("behind_by") is not None else None,
                comparison_status=str(cmp.get("status")) if cmp and cmp.get("status") else None,
                disposition=classify_pr(pr, cmp),
            )
        )

    recs: list[str] = []
    if repo.get("archived"):
        recs.append("Keep archived unless there is a documented reactivation reason.")
    if not repo.get("description"):
        recs.append("Add a concise repository description.")
    if repo.get("fork"):
        recs.append("Document why this fork exists or detach/archive it if obsolete.")
    if pulls:
        blocked = sum(p.disposition in {"draft-needs-sync", "sync-before-merge"} for p in pulls)
        if blocked:
            recs.append(f"Synchronize {blocked} open PR(s) with their base before merge consideration.")
        if len(pulls) >= 10:
            recs.append("Run a PR-stack/supersession review; the open integration queue is large.")
    if name == "vigilant-winner":
        recs.append("Placeholder candidate: archive in repository settings when the namespace is no longer needed.")
    if name == "Jarvis-X":
        recs.append("Treat as canonical integration/control-plane repository; keep experimental work gated by PR review and CI.")
    if name == "stable-agent":
        recs.append("Keep private until promotion criteria, tests, threat model and narrow public API are satisfied.")

    return RepoAudit(
        name=name,
        full_name=str(repo["full_name"]),
        private=bool(repo.get("private")),
        archived=bool(repo.get("archived")),
        fork=bool(repo.get("fork")),
        default_branch=str(repo.get("default_branch") or "main"),
        description=repo.get("description"),
        open_issues_count=int(repo.get("open_issues_count") or 0),
        stars=int(repo.get("stargazers_count") or 0),
        forks=int(repo.get("forks_count") or 0),
        pushed_at=repo.get("pushed_at"),
        open_prs=tuple(pulls),
        recommendations=tuple(recs),
    )


def build_report(owner: str, token: str | None) -> dict[str, Any]:
    repos = [
        r
        for r in paginate(f"/user/repos?affiliation=owner&sort=updated&direction=desc", token)
        if r.get("owner", {}).get("login", "").lower() == owner.lower()
    ]
    audits = [audit_repo(owner, repo, token) for repo in repos]
    open_prs = sum(len(repo.open_prs) for repo in audits)
    sync_needed = sum(
        pr.disposition in {"draft-needs-sync", "sync-before-merge"}
        for repo in audits
        for pr in repo.open_prs
    )
    return {
        "schema": "jarvisx.github-account-audit.v1",
        "owner": owner,
        "authority": "read-only-candidate-analysis",
        "invariants": [
            "No automatic merge, close, delete, archive or force-push.",
            "CI evidence is necessary but not sufficient for integration.",
            "Virtual/experimental capability claims remain separate from shipped capability.",
            "Repository mutations require an explicit reviewed commit boundary.",
        ],
        "summary": {
            "owned_repositories": len(audits),
            "public_repositories": sum(not r.private for r in audits),
            "private_repositories": sum(r.private for r in audits),
            "open_pull_requests": open_prs,
            "pull_requests_needing_sync": sync_needed,
        },
        "repositories": [asdict(repo) for repo in audits],
    }


def print_human(report: dict[str, Any]) -> None:
    summary = report["summary"]
    print(f"GitHub control-plane audit for {report['owner']}")
    print(
        f"repos={summary['owned_repositories']} public={summary['public_repositories']} "
        f"private={summary['private_repositories']} open_prs={summary['open_pull_requests']} "
        f"needs_sync={summary['pull_requests_needing_sync']}"
    )
    for repo in report["repositories"]:
        privacy = "private" if repo["private"] else "public"
        print(f"\n[{repo['full_name']}] {privacy} open_prs={len(repo['open_prs'])}")
        for rec in repo["recommendations"]:
            print(f"  - {rec}")
        for pr in repo["open_prs"]:
            lag = "?" if pr["behind_by"] is None else pr["behind_by"]
            print(f"  PR #{pr['number']}: {pr['disposition']} behind={lag} — {pr['title']}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--owner", default=os.getenv("GITHUB_OWNER", "Lord-Xido"))
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON only")
    parser.add_argument("--output", help="Write JSON report to this path")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    token = os.getenv("GITHUB_TOKEN")
    try:
        report = build_report(args.owner, token)
    except GitHubError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    if args.json:
        print(text)
    else:
        print_human(report)
        if not token:
            print("\nNote: unauthenticated API limits apply; set GITHUB_TOKEN for private repositories and higher rate limits.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
