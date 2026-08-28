"""Run a bounded 10x10x10 inward-4D self-description experiment."""

from __future__ import annotations

import argparse
import json
import math

from jarvisx.inward4d_ann import Inward4DANN, index_to_coordinate


def target_field(side: int = 10) -> list[float]:
    """Create a deterministic bounded volumetric interference fixture."""

    field: list[float] = []
    center = (side - 1) / 2.0
    for index in range(side**3):
        x, y, z = index_to_coordinate(index, side)
        dx = (x - center) / center
        dy = (y - center) / center
        dz = (z - center) / center
        radius = math.sqrt(dx * dx + dy * dy + dz * dz)
        envelope = math.exp(-1.8 * radius * radius)
        interference = math.sin(2.0 * math.pi * dx) * math.cos(math.pi * dy * dz)
        field.append(0.65 * envelope * interference)
    return field


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=25)
    args = parser.parse_args()

    engine = Inward4DANN()
    target = target_field()
    report = engine.optimize(target, max_epochs=args.epochs)
    last_step = report.history[-1] if report.history else None
    output = {
        "engine": "Jarvis-X 10x10x10 Inward 4D ANN reference",
        "arithmetic": engine.arithmetic_summary(),
        "optimization": {
            "attempted_epochs": report.attempted_epochs,
            "committed_epochs": report.committed_epochs,
            "initial_loss": report.initial.loss.total,
            "final_loss": report.final.loss.total,
            "initial_description_residual_rms": report.initial.description_residual_rms,
            "final_description_residual_rms": report.final.description_residual_rms,
            "converged": report.converged,
            "last_learning_rate": last_step.learning_rate_used if last_step else 0.0,
            "active_synapses": engine.active_synapse_count,
        },
        "claim_boundary": "bounded deterministic Python reference; no compression or performance claim",
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
