"""Operational control layer for the bounded Jarvis-X 3D multimodal loop.

This module turns :mod:`jarvisx.dr_moagi_multimodal_loop` into an auditable
end-to-end runtime with four explicit responsibilities:

1. ingest user payloads without inventing semantic codecs;
2. execute deterministic candidate configurations of the 3D loop;
3. verify reconstruction/cycle/fixed-point behaviour and localise error;
4. select the lowest-cost valid candidate, export it and append telemetry.

The controller deliberately keeps physical execution claims bounded.  The
``10**24`` address space in the underlying loop remains virtual accounting; the
metrics below are measurements of the finite Python computation actually run.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

from .dr_moagi_multimodal_loop import (
    MODALITIES,
    DrMoagiMultimodal3DLoop,
    Modality,
    Volume3D,
)


@dataclass(frozen=True)
class CandidateConfig:
    """One bounded runtime configuration considered by the optimiser."""

    edge: int = 8
    temporal_depth: int = 4
    mix: float = 0.35
    cycles: int = 3
    seed: int = 0x4A415256495358
    deterministic_stride: int = 10**20

    def validate(self) -> None:
        if self.edge < 4 or self.edge % 2:
            raise ValueError("edge must be even and >= 4")
        if not 1 <= self.temporal_depth <= 64:
            raise ValueError("temporal_depth must be in [1, 64]")
        if not 0.0 <= self.mix <= 1.0:
            raise ValueError("mix must be in [0, 1]")
        if self.cycles < 1:
            raise ValueError("cycles must be >= 1")
        if self.deterministic_stride < 0:
            raise ValueError("deterministic_stride must be non-negative")


@dataclass(frozen=True)
class ActiveErrorRegion:
    modality: str
    index: int
    x: int
    y: int
    z: int
    absolute_error: float


@dataclass(frozen=True)
class VerificationReport:
    valid: bool
    finite_metrics: bool
    fusion_normalized: bool
    generated_all_modalities: bool
    input_lengths_preserved: bool
    aggregate_reconstruction_mse: float
    aggregate_cycle_mse: float
    fixed_point_delta: float
    fixed_point_penalty: float
    payload_l1_error: float
    score: float
    active_regions: tuple[ActiveErrorRegion, ...]
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class CandidateReport:
    config: CandidateConfig
    verification: VerificationReport
    elapsed_ms: float


@dataclass(frozen=True)
class OptimizationReport:
    selected: CandidateReport
    candidates: tuple[CandidateReport, ...]
    generated_at_unix: float


def _payload_bytes(value: bytes | bytearray | memoryview | str) -> bytes:
    if isinstance(value, str):
        return value.encode("utf-8")
    return bytes(value)


def _mean_absolute_byte_error(expected: bytes, actual: bytes) -> float:
    """Return normalized byte-domain L1 error in [0, 1]."""

    if len(expected) != len(actual):
        return 1.0
    if not expected:
        return 0.0
    return sum(abs(a - b) for a, b in zip(expected, actual)) / (255.0 * len(expected))


def _finite(value: float) -> bool:
    return math.isfinite(value)


def _voxel_regions(
    modality: Modality,
    reference: Volume3D,
    generated: Volume3D,
    *,
    limit: int = 12,
    minimum_error: float = 0.05,
) -> tuple[ActiveErrorRegion, ...]:
    """Return the largest local reconstruction errors as sparse 3D regions."""

    reference._same(generated)
    edge = reference.edge
    ranked: list[tuple[float, int]] = []
    for index, (a, b) in enumerate(zip(reference.values, generated.values)):
        error = abs(a - b)
        if error >= minimum_error:
            ranked.append((error, index))
    ranked.sort(reverse=True)
    out: list[ActiveErrorRegion] = []
    for error, index in ranked[: max(0, limit)]:
        x = index % edge
        y = (index // edge) % edge
        z = index // (edge * edge)
        out.append(
            ActiveErrorRegion(
                modality=modality.value,
                index=index,
                x=x,
                y=y,
                z=z,
                absolute_error=error,
            )
        )
    return tuple(out)


class TelemetryLedger:
    """Append-only JSONL memory for accepted operational runs."""

    def __init__(self, path: Optional[str | Path]) -> None:
        self.path = Path(path).expanduser() if path else None

    def append(self, report: OptimizationReport) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "generated_at_unix": report.generated_at_unix,
            "selected": _candidate_to_dict(report.selected),
        }
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")


class RecursiveOperationalController:
    """Generate -> contrast -> verify -> select -> export -> remember."""

    def __init__(
        self,
        payloads: Optional[Mapping[Modality | str, bytes | str]] = None,
        *,
        memory_path: Optional[str | Path] = None,
        active_region_limit: int = 12,
    ) -> None:
        self.payloads: dict[Modality, bytes] = {}
        self.ledger = TelemetryLedger(memory_path)
        self.active_region_limit = max(0, int(active_region_limit))
        if payloads:
            for modality, payload in payloads.items():
                self.set_payload(modality, payload)

    def set_payload(self, modality: Modality | str, payload: bytes | str) -> None:
        key = modality if isinstance(modality, Modality) else Modality(str(modality).lower())
        self.payloads[key] = _payload_bytes(payload)

    def _build_engine(self, config: CandidateConfig) -> DrMoagiMultimodal3DLoop:
        config.validate()
        engine = DrMoagiMultimodal3DLoop(
            edge=config.edge,
            temporal_depth=config.temporal_depth,
            cross_modal_mix=config.mix,
            seed=config.seed,
        )
        for modality, payload in self.payloads.items():
            engine.set_payload(modality, payload)
        return engine

    def _verify(
        self,
        engine: DrMoagiMultimodal3DLoop,
        config: CandidateConfig,
    ) -> VerificationReport:
        metrics = engine.last_metrics
        if metrics is None:
            raise RuntimeError("engine must be run before verification")

        reasons: list[str] = []
        finite_metrics = all(
            _finite(v)
            for v in (
                metrics.aggregate_reconstruction_mse,
                metrics.aggregate_cycle_mse,
            )
        )
        if not finite_metrics:
            reasons.append("non-finite aggregate metric")

        weights = [item.fusion_weight for item in metrics.modalities.values()]
        fusion_normalized = bool(weights) and abs(sum(weights) - 1.0) <= 1.0e-6
        if not fusion_normalized:
            reasons.append("fusion weights are not normalized")

        generated_all = all(modality in engine.generated for modality in MODALITIES)
        if not generated_all:
            reasons.append("one or more modalities were not generated")

        input_lengths_preserved = True
        payload_errors: list[float] = []
        for modality, expected in self.payloads.items():
            try:
                actual = engine.generated_payload(modality)
            except RuntimeError:
                input_lengths_preserved = False
                payload_errors.append(1.0)
                continue
            if len(actual) != max(1, len(expected)):
                input_lengths_preserved = False
            payload_errors.append(_mean_absolute_byte_error(expected or b"\0", actual))
        if not input_lengths_preserved:
            reasons.append("generated payload length changed")

        payload_l1_error = sum(payload_errors) / len(payload_errors) if payload_errors else 0.0

        if _finite(metrics.fixed_point_delta):
            fixed_point_penalty = min(1.0, max(0.0, metrics.fixed_point_delta))
        else:
            # A first-cycle infinity is expected because there is no previous signature.
            # It is not expected once a candidate has executed two or more cycles.
            fixed_point_penalty = 1.0
            if config.cycles >= 2:
                reasons.append("fixed-point delta did not become finite")

        regions: list[ActiveErrorRegion] = []
        if generated_all:
            for modality in MODALITIES:
                if modality not in engine.inputs:
                    continue
                regions.extend(
                    _voxel_regions(
                        modality,
                        engine.inputs[modality],
                        engine.generated[modality],
                        limit=self.active_region_limit,
                    )
                )
        regions.sort(key=lambda item: item.absolute_error, reverse=True)
        active_regions = tuple(regions[: self.active_region_limit])

        # Correctness/validity is a hard gate.  Quality then chooses among valid candidates.
        valid = (
            finite_metrics
            and fusion_normalized
            and generated_all
            and input_lengths_preserved
            and (config.cycles < 2 or _finite(metrics.fixed_point_delta))
        )

        score = (
            metrics.aggregate_reconstruction_mse
            + metrics.aggregate_cycle_mse
            + 0.25 * fixed_point_penalty
            + 0.50 * payload_l1_error
        )
        if not valid:
            score = math.inf

        return VerificationReport(
            valid=valid,
            finite_metrics=finite_metrics,
            fusion_normalized=fusion_normalized,
            generated_all_modalities=generated_all,
            input_lengths_preserved=input_lengths_preserved,
            aggregate_reconstruction_mse=metrics.aggregate_reconstruction_mse,
            aggregate_cycle_mse=metrics.aggregate_cycle_mse,
            fixed_point_delta=metrics.fixed_point_delta,
            fixed_point_penalty=fixed_point_penalty,
            payload_l1_error=payload_l1_error,
            score=score,
            active_regions=active_regions,
            reasons=tuple(reasons),
        )

    def evaluate(self, config: CandidateConfig) -> tuple[DrMoagiMultimodal3DLoop, CandidateReport]:
        started = time.perf_counter_ns()
        engine = self._build_engine(config)
        engine.run(config.cycles, config.deterministic_stride)
        verification = self._verify(engine, config)
        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0
        return engine, CandidateReport(config=config, verification=verification, elapsed_ms=elapsed_ms)

    def optimize(
        self,
        configs: Iterable[CandidateConfig],
        *,
        output_dir: Optional[str | Path] = None,
    ) -> OptimizationReport:
        reports: list[CandidateReport] = []
        engines: list[DrMoagiMultimodal3DLoop] = []
        for config in configs:
            engine, report = self.evaluate(config)
            engines.append(engine)
            reports.append(report)

        if not reports:
            raise ValueError("at least one candidate configuration is required")

        valid_indices = [i for i, report in enumerate(reports) if report.verification.valid]
        if not valid_indices:
            reason_text = "; ".join(
                f"candidate {i}: {', '.join(r.verification.reasons) or 'invalid'}"
                for i, r in enumerate(reports)
            )
            raise RuntimeError(f"no candidate passed verification: {reason_text}")

        selected_index = min(valid_indices, key=lambda i: reports[i].verification.score)
        selected = reports[selected_index]
        selected_engine = engines[selected_index]
        result = OptimizationReport(
            selected=selected,
            candidates=tuple(reports),
            generated_at_unix=time.time(),
        )

        if output_dir is not None:
            destination = Path(output_dir)
            destination.mkdir(parents=True, exist_ok=True)
            selected_engine.export(destination)
            (destination / "operational-report.json").write_text(
                json.dumps(_optimization_to_dict(result), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

        self.ledger.append(result)
        return result


def _candidate_to_dict(report: CandidateReport) -> dict[str, object]:
    data = asdict(report)
    # JSON has no representation for infinity; make it explicit and portable.
    delta = report.verification.fixed_point_delta
    data["verification"]["fixed_point_delta"] = delta if math.isfinite(delta) else None
    score = report.verification.score
    data["verification"]["score"] = score if math.isfinite(score) else None
    return data


def _optimization_to_dict(report: OptimizationReport) -> dict[str, object]:
    return {
        "generated_at_unix": report.generated_at_unix,
        "selected": _candidate_to_dict(report.selected),
        "candidates": [_candidate_to_dict(item) for item in report.candidates],
    }


def candidate_grid(
    *,
    edge: int,
    depths: Sequence[int],
    mixes: Sequence[float],
    cycles: int,
    seed: int,
    deterministic_stride: int,
) -> tuple[CandidateConfig, ...]:
    configs = []
    for depth in depths:
        for mix in mixes:
            configs.append(
                CandidateConfig(
                    edge=edge,
                    temporal_depth=int(depth),
                    mix=float(mix),
                    cycles=cycles,
                    seed=seed,
                    deterministic_stride=deterministic_stride,
                )
            )
    return tuple(configs)


def _csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def _csv_floats(value: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Verified, self-selecting operational controller for the Jarvis-X 3D multimodal loop"
    )
    p.add_argument("--input", action="append", default=[], metavar="MODALITY=PATH")
    p.add_argument("--text", help="literal UTF-8 text payload")
    p.add_argument("--edge", type=int, default=8)
    p.add_argument("--depths", type=_csv_ints, default=(2, 4), help="candidate temporal depths")
    p.add_argument("--mixes", type=_csv_floats, default=(0.20, 0.35, 0.50), help="candidate fusion mixes")
    p.add_argument("--cycles", type=int, default=3)
    p.add_argument("--seed", type=int, default=0x4A415256495358)
    p.add_argument("--deterministic-stride", type=int, default=10**20)
    p.add_argument("--memory", default=None, help="append-only JSONL telemetry ledger")
    p.add_argument("--out", default="./dm-operational-out")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parser().parse_args(argv)
    controller = RecursiveOperationalController(memory_path=args.memory)

    try:
        for spec in args.input:
            name, path = spec.split("=", 1)
            controller.set_payload(Modality(name.lower()), Path(path).read_bytes())
        if args.text is not None:
            controller.set_payload(Modality.TEXT, args.text)

        configs = candidate_grid(
            edge=args.edge,
            depths=args.depths,
            mixes=args.mixes,
            cycles=args.cycles,
            seed=args.seed,
            deterministic_stride=args.deterministic_stride,
        )
        result = controller.optimize(configs, output_dir=args.out)
    except (OSError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"error: {exc}") from exc

    if not args.quiet:
        print(json.dumps(_optimization_to_dict(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
