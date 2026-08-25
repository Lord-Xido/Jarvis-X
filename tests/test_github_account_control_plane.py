"""Focused tests for the read-only GitHub account control-plane auditor."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _load_control_plane() -> ModuleType:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "github_account_control_plane.py"
    )
    spec = importlib.util.spec_from_file_location(
        "jarvisx_github_account_control_plane", module_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


control_plane = _load_control_plane()


def test_draft_pr_needing_sync_is_not_integration_candidate() -> None:
    pr = {"draft": True}
    comparison = {"behind_by": 4, "status": "diverged"}

    assert control_plane.classify_pr(pr, comparison) == "draft-needs-sync"


def test_aligned_draft_remains_review_only() -> None:
    pr = {"draft": True}
    comparison = {"behind_by": 0, "status": "ahead"}

    assert control_plane.classify_pr(pr, comparison) == "draft-review"


def test_non_draft_behind_base_requires_sync() -> None:
    pr = {"draft": False}
    comparison = {"behind_by": 1, "status": "diverged"}

    assert control_plane.classify_pr(pr, comparison) == "sync-before-merge"


def test_non_draft_diverged_base_requires_sync_even_without_behind_count() -> None:
    pr = {"draft": False}
    comparison = {"behind_by": 0, "status": "diverged"}

    assert control_plane.classify_pr(pr, comparison) == "sync-before-merge"


def test_non_draft_current_branch_is_only_an_integration_candidate() -> None:
    pr = {"draft": False}
    comparison = {"behind_by": 0, "status": "ahead"}

    assert control_plane.classify_pr(pr, comparison) == "integration-candidate"


def test_missing_comparison_fails_to_manual_review() -> None:
    pr = {"draft": False}

    assert control_plane.classify_pr(pr, None) == "review"
