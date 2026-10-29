"""Reproducible end-to-end 3D shadow transaction geometry evidence.

Observe four isolated real backend engines; prepare four separately persisted
shadow SQLite participants; simulate interruption after applying ANN; recover;
export before/partial/complete canonical envelope + JSON + SVG artifacts.
No live backend or production authority is modified.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

from reference.canonical_shadow_tx.bridge import (
    DOMAINS, observe_ann, observe_geometry, observe_os, observe_vm,
)
from reference.canonical_shadow_tx.geometry_witness import (
    decode_lattice, svg_view, witness,
)
from reference.canonical_shadow_tx.multiprocess import Coordinator


def export_run(out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    import torch
    torch.set_num_threads(2)
    frames = tuple(sorted((observe_ann(), observe_geometry(),
                           observe_os(), observe_vm()), key=lambda f: f.domain))
    summary = {"schema": "CanonicalShadowTx/GeometricEndToEnd3D1",
               "domains": list(DOMAINS),
               "authority": "four independently persisted local SHADOW participant snapshots",
               "production_deployment": "BLOCKED",
               "results": {}}
    with tempfile.TemporaryDirectory() as folder:
        with Coordinator(Path(folder)) as coordinator:
            coordinator.prepare("geometry-e2e", frames)

            def capture(stage: str) -> None:
                result = witness(coordinator, "geometry-e2e", iterations=1000)
                if decode_lattice(result.packet) != result.participant_statuses:
                    raise RuntimeError("3D lossless reconstruction failed")
                (out / (stage + ".env")).write_bytes(result.packet)
                (out / (stage + ".json")).write_text(
                    json.dumps(result.as_dict(), sort_keys=True, indent=2) + "\n"
                )
                (out / (stage + ".svg")).write_text(svg_view(result))
                summary["results"][stage] = {
                    "coordinator_stage": result.stage,
                    "geometrically_verified": result.verifiable,
                    "commit_complete": result.complete,
                    "participant_statuses": result.participant_statuses,
                    "envelope_sha256": result.as_dict()["lattice"]["envelope_sha256"],
                }

            capture("prepared")
            def interrupt(point: str) -> None:
                if point == "after_apply_ann":
                    raise OSError("deliberate post-decision commit interruption")
            try:
                coordinator.approve("geometry-e2e", approve=True, fault=interrupt)
                raise AssertionError("expected fault injection did not occur")
            except OSError:
                pass
            capture("partial")
            if coordinator.recover() != {"geometry-e2e": "COMPLETE"}:
                raise AssertionError("incomplete recovery")
            capture("recovered")
            if coordinator.coordinated_read("geometry-e2e") != frames:
                raise AssertionError("coordinated read failed")
    expected = (
        ("prepared", True, False),
        ("partial", True, False),
        ("recovered", True, True),
    )
    for name, verified, complete in expected:
        row = summary["results"][name]
        if (row["geometrically_verified"], row["commit_complete"]) != (
                verified, complete):
            raise AssertionError("3D transition failed: " + name)
    if (summary["results"]["partial"]["participant_statuses"]["ann"] != "COMMITTED"
            or summary["results"]["partial"]["participant_statuses"]["os"] != "PREPARED"):
        raise AssertionError("missing partial-visibility failure witness")
    (out / "summary.json").write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export_run(args.out), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
