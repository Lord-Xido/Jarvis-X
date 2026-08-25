# GitHub Account Control Plane

Jarvis-X treats GitHub portfolio maintenance as a governed operational workflow rather than an unrestricted autonomous mutation loop.

## Control loop

```text
OBSERVE
  repositories, branches, pull requests, CI state, public/private boundaries
    ↓
CLASSIFY
  canonical | experimental | incubation | placeholder
    ↓
PROPOSE
  sync | review | merge candidate | superseded candidate | archive candidate
    ↓
VERIFY
  current base/head relationship, CI at current head, stack dependencies,
  capability claims, security/reproducibility gates
    ↓
COMMIT
  explicit reviewed GitHub mutation
    ↓
RECORD
  issue/PR history, commit provenance, release notes
```

## Authority boundary

The account manager may inventory, classify, compare and recommend automatically. It must not automatically:

- merge or close pull requests;
- delete branches or files;
- force-push refs;
- archive or delete repositories;
- publish releases;
- alter security, governance or provenance rules.

Those remain explicit reviewed mutations.

## Reference auditor

Run:

```bash
python scripts/github_account_control_plane.py
```

For private repositories and higher GitHub API limits:

```bash
export GITHUB_TOKEN=...
python scripts/github_account_control_plane.py --json --output account-audit.json
```

The token is read from the environment and is never written to the report.

## Portfolio roles

- **Jarvis-X** — canonical systems/integration repository and account control plane.
- **3D-Virtual-AI-Interactive-Interface** — bounded browser/native visualization and interactive demonstration layer.
- **stable-agent** — private incubation boundary for focused agent work before public promotion.
- **vigilant-winner** — inactive placeholder; archive candidate when the namespace is no longer needed.

## PR disposition vocabulary

- `integration-candidate` — branch is not behind its base; still requires CI/review.
- `sync-before-merge` — non-draft branch is behind/diverged from its base.
- `draft-review` — draft branch currently aligned enough for review.
- `draft-needs-sync` — draft branch is behind its base and should be synchronized before integration review.
- `review` — comparison could not be established; manual inspection required.

CI success is necessary but not sufficient. A green head can still be stale, stacked on an unmerged dependency, conflicting with `main`, or superseded by newer work.

## Current management priority

The highest-value account-level action is PR backlog normalization: determine which open Jarvis-X PRs are still relevant, synchronize viable candidates with current `main`, explicitly mark stacked dependencies, and close only after a supersession/relevance review.

The account should present one coherent architecture: executable capabilities on `main`, experimental capabilities in reviewable branches/PRs, and public claims bounded by current evidence.
