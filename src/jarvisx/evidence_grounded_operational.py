"""Evidence-grounded operational bridge for Jarvis-X.

This module permeates active multimodal evidence acquisition into the bounded
Dr Moagi operational controller. Retrieval is no longer merely advisory when a
caller declares required evidence modalities: authoritative candidate
optimization is blocked until the evidence gate passes.

The bridge deliberately keeps evidence acquisition and payload execution typed
and separate. An audio EvidenceSegment does not magically become audio bytes;
native adapters must supply those bytes to the operational controller when the
downstream computation actually requires them.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

from .dr_moagi_multimodal_loop import Modality as RuntimeModality
from .dr_moagi_operational_runtime import (
    CandidateConfig,
    OptimizationReport,
    RecursiveOperationalController,
    candidate_grid,
)
from .multimodal_evidence_runtime import (
    ActiveMultimodalEvidenceEngine,
    RetrievalQuery,
    RetrievalResult,
    demo_store,
    result_to_json,
    MultimodalRetriever,
)


class EvidenceGateRejected(RuntimeError):
    """Raised when required external evidence has not passed CTR verification."""


@dataclass(frozen=True)
class EvidenceGateReceipt:
    accepted: bool
    uncertainty: float
    evidence_count: int
    rounds: int
    unresolved_modalities: tuple[str, ...]
    fixed_point_residual: float | None


@dataclass(frozen=True)
class EvidenceGroundedOptimizationReport:
    evidence: EvidenceGateReceipt
    operational: OptimizationReport


def evidence_gate_receipt(result: RetrievalResult) -> EvidenceGateReceipt:
    residual = result.rounds[-1].fixed_point_residual if result.rounds else None
    return EvidenceGateReceipt(
        accepted=result.accepted,
        uncertainty=result.uncertainty,
        evidence_count=len(result.evidence),
        rounds=len(result.rounds),
        unresolved_modalities=result.unresolved_modalities,
        fixed_point_residual=residual,
    )


class EvidenceGroundedOperationalController:
    """Require verified multimodal evidence before operational promotion."""

    def __init__(
        self,
        evidence_engine: ActiveMultimodalEvidenceEngine,
        operational_controller: RecursiveOperationalController,
    ) -> None:
        self.evidence_engine = evidence_engine
        self.operational_controller = operational_controller

    def optimize(
        self,
        query: RetrievalQuery,
        configs: Iterable[CandidateConfig],
        *,
        output_dir: Optional[str | Path] = None,
    ) -> EvidenceGroundedOptimizationReport:
        evidence_result = self.evidence_engine.run(query)
        gate = evidence_gate_receipt(evidence_result)

        if not gate.accepted:
            unresolved = ", ".join(gate.unresolved_modalities) or "none"
            raise EvidenceGateRejected(
                "multimodal evidence gate rejected authoritative promotion: "
                f"uncertainty={gate.uncertainty:.6f}, "
                f"unresolved_modalities={unresolved}"
            )

        operational = self.operational_controller.optimize(
            configs,
            output_dir=output_dir,
        )

        if output_dir is not None:
            destination = Path(output_dir)
            destination.mkdir(parents=True, exist_ok=True)
            (destination / "evidence-gate.json").write_text(
                json.dumps(result_to_json(evidence_result), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            (destination / "evidence-grounded-summary.json").write_text(
                json.dumps(
                    {
                        "schema_version": "jarvisx.evidence-grounded-operational.v1",
                        "evidence_gate": asdict(gate),
                        "selected_config": asdict(operational.selected.config),
                        "selected_verification": asdict(
                            operational.selected.verification
                        ),
                        "claim_boundary": {
                            "implemented": (
                                "required-modality evidence gate precedes operational "
                                "candidate promotion"
                            ),
                            "not_implied": (
                                "evidence metadata is not substituted for unavailable "
                                "native media bytes"
                            ),
                        },
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )

        return EvidenceGroundedOptimizationReport(
            evidence=gate,
            operational=operational,
        )


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run the evidence-gated Jarvis-X operational reference pipeline."
    )
    p.add_argument(
        "--query",
        default="song where his partner asks him to come home",
    )
    p.add_argument("--out", default="./evidence-grounded-out")
    p.add_argument("--edge", type=int, default=4)
    p.add_argument("--depths", default="2")
    p.add_argument("--mixes", default="0.35")
    p.add_argument("--cycles", type=int, default=2)
    p.add_argument("--seed", type=int, default=0x4A415256495358)
    p.add_argument("--deterministic-stride", type=int, default=10**20)
    return p


def _csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def _csv_floats(value: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parser().parse_args(argv)

    evidence_engine = ActiveMultimodalEvidenceEngine(
        MultimodalRetriever(demo_store())
    )
    operational = RecursiveOperationalController(
        payloads={RuntimeModality.TEXT: args.query}
    )
    controller = EvidenceGroundedOperationalController(
        evidence_engine,
        operational,
    )
    query = RetrievalQuery(
        text=args.query,
        top_k=3,
        required_modalities=(),
        preferred_modalities=(),
        uncertainty_threshold=0.45,
        max_rounds=3,
    )
    configs = candidate_grid(
        edge=args.edge,
        depths=_csv_ints(args.depths),
        mixes=_csv_floats(args.mixes),
        cycles=args.cycles,
        seed=args.seed,
        deterministic_stride=args.deterministic_stride,
    )

    try:
        report = controller.optimize(query, configs, output_dir=args.out)
    except EvidenceGateRejected as exc:
        print(str(exc))
        return 2

    print("Jarvis-X evidence-grounded operational bridge")
    print(f"evidence_accepted={report.evidence.accepted}")
    print(f"uncertainty={report.evidence.uncertainty:.6f}")
    print(f"selected_score={report.operational.selected.verification.score:.6f}")
    print(f"output={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
