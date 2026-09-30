"""Operational 3D EM-Sig software runtime.

This module extends jarvisx.em_sig_emulation into a complete, bounded
3D software execution path:

bits -> QPSK -> spatial source program -> 3D interference proxy
     -> voxel readout -> ternary logic volume -> sparse sensors
     -> six-bit spatial latent -> fixed-point refinement
     -> residual correction -> recovered bits

The runtime is deterministic software research infrastructure. It does not solve
Maxwell's equations, drive RF hardware, measure electromagnetic fields, or
establish physical throughput/BER claims.
"""

from __future__ import annotations

import argparse
import cmath
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence, cast

from jarvisx.em_sig_emulation import (
    EMSigConfig,
    LatentTrace,
    create_world,
    modeled_rate_arithmetic,
    qpsk_map,
    refine_latent_fixed_point,
    residual_correction_loop,
)


@dataclass(frozen=True)
class SourceProgram3D:
    """One software source instruction in the spatial excitation program."""

    index: int
    x: float
    y: float
    z: float
    amplitude: float
    phase_radians: float


@dataclass(frozen=True)
class SensorSample3D:
    """Sparse readout from the materialized ternary logic volume."""

    x: int
    y: int
    z: int
    readout: float
    logic: int


@dataclass(frozen=True)
class EMSig3DConfig:
    """Configuration for the operational 3D software runtime."""

    em: EMSigConfig = EMSigConfig()
    source_radius: float = 1.8
    source_z_offset: float = 0.7
    logic_deadband: float = 0.20
    sensor_stride: int = 2

    def validate(self) -> None:
        self.em.validate()
        if self.source_radius <= 0.0:
            raise ValueError("source_radius must be positive")
        if self.source_z_offset < 0.0:
            raise ValueError("source_z_offset must be non-negative")
        if not 0.0 <= self.logic_deadband < 1.0:
            raise ValueError("logic_deadband must be in [0, 1)")
        if self.sensor_stride < 1:
            raise ValueError("sensor_stride must be >= 1")


def _flat_index(x: int, y: int, z: int, extent: int) -> int:
    return x + extent * (y + extent * z)


def build_source_program(
    symbols: Sequence[complex],
    *,
    source_count: int,
    radius: float,
    z_offset: float,
) -> tuple[SourceProgram3D, ...]:
    """Map QPSK symbols into a deterministic ring of 3D source instructions."""

    if source_count < 1:
        raise ValueError("source_count must be positive")
    if len(symbols) < source_count:
        raise ValueError("not enough symbols to program all sources")

    program: list[SourceProgram3D] = []
    for index in range(source_count):
        angle = 2.0 * math.pi * index / source_count
        symbol = symbols[index]
        program.append(
            SourceProgram3D(
                index=index,
                x=radius * math.cos(angle),
                y=radius * math.sin(angle),
                z=z_offset if index % 2 == 0 else -z_offset,
                amplitude=abs(symbol),
                phase_radians=cmath.phase(symbol),
            )
        )
    return tuple(program)


def programmed_interference_field(
    config: EMSig3DConfig,
    program: Sequence[SourceProgram3D],
) -> tuple[tuple[float, ...], float]:
    """Materialize a data-dependent normalized 3D scalar field proxy."""

    extent = config.em.materialized_extent
    k = 2.0 * math.pi / config.em.normalized_wavelength
    values: list[float] = []
    peak = 0.0

    for iz in range(extent):
        z = -1.0 + 2.0 * iz / (extent - 1)
        for iy in range(extent):
            y = -1.0 + 2.0 * iy / (extent - 1)
            for ix in range(extent):
                x = -1.0 + 2.0 * ix / (extent - 1)
                total = 0j
                for source in program:
                    dx = x - source.x
                    dy = y - source.y
                    dz = z - source.z
                    distance = max(math.sqrt(dx * dx + dy * dy + dz * dz), 1.0e-6)
                    phase = k * distance + source.phase_radians
                    total += source.amplitude * cmath.exp(1j * phase) / distance
                intensity = total.real * total.real + total.imag * total.imag
                values.append(intensity)
                peak = max(peak, intensity)

    if peak <= 0.0:
        return tuple(values), 0.0
    return tuple(value / peak for value in values), peak


def threshold_logic_volume(
    field: Sequence[float],
    *,
    extent: int,
    deadband: float,
) -> tuple[tuple[float, ...], tuple[int, ...]]:
    """Decode normalized field values into a centered ternary 3D logic volume."""

    if len(field) != extent**3:
        raise ValueError("field size does not match extent^3")
    if not 0.0 <= deadband < 1.0:
        raise ValueError("deadband must be in [0, 1)")

    readout: list[float] = []
    logic: list[int] = []
    for value in field:
        centered = 2.0 * value - 1.0
        readout.append(centered)
        if centered > deadband:
            logic.append(1)
        elif centered < -deadband:
            logic.append(-1)
        else:
            logic.append(0)
    return tuple(readout), tuple(logic)


def logic_volume_stats(logic: Sequence[int]) -> dict[str, int | float]:
    """Summarize the ternary lattice."""

    total = len(logic)
    negative = sum(value == -1 for value in logic)
    deadband = sum(value == 0 for value in logic)
    positive = sum(value == 1 for value in logic)
    return {
        "voxels": total,
        "negative": negative,
        "deadband": deadband,
        "positive": positive,
        "active": negative + positive,
        "active_fraction": (negative + positive) / max(total, 1),
    }


def sparse_sensor_readout(
    readout: Sequence[float],
    logic: Sequence[int],
    *,
    extent: int,
    stride: int,
) -> tuple[SensorSample3D, ...]:
    """Sample a regular sparse sensor lattice from the 3D logical volume."""

    if len(readout) != extent**3 or len(logic) != extent**3:
        raise ValueError("readout/logic size does not match extent^3")
    if stride < 1:
        raise ValueError("stride must be >= 1")

    samples: list[SensorSample3D] = []
    for z in range(0, extent, stride):
        for y in range(0, extent, stride):
            for x in range(0, extent, stride):
                index = _flat_index(x, y, z, extent)
                samples.append(
                    SensorSample3D(
                        x=x,
                        y=y,
                        z=z,
                        readout=readout[index],
                        logic=logic[index],
                    )
                )
    return tuple(samples)


def spatial_latent_6bit(
    logic: Sequence[int],
    samples: Sequence[SensorSample3D],
) -> int:
    """Fold the 3D logic geometry into a deterministic six-bit signature."""

    accumulator = 0x15
    for index, value in enumerate(logic):
        encoded = value + 1
        accumulator = ((accumulator * 33) ^ (encoded + index * 17)) & 0xFFFFFFFF
    for sample in samples:
        packed = (
            (sample.x * 3)
            ^ (sample.y * 5)
            ^ (sample.z * 7)
            ^ ((sample.logic + 1) * 11)
        )
        accumulator = ((accumulator * 65) ^ packed) & 0xFFFFFFFF
    return accumulator & 0x3F


def _latent_trace_json(trace: Sequence[LatentTrace]) -> list[dict[str, int]]:
    return [asdict(row) for row in trace]


def run_3d_runtime(config: EMSig3DConfig = EMSig3DConfig()) -> dict[str, object]:
    """Execute the complete deterministic 3D software signalling pipeline."""

    config.validate()
    em = config.em

    source_bits = create_world(em.bit_count, em.seed)
    symbols = qpsk_map(source_bits)
    program = build_source_program(
        symbols,
        source_count=em.field_sources,
        radius=config.source_radius,
        z_offset=config.source_z_offset,
    )
    field, raw_peak = programmed_interference_field(config, program)
    readout, logic = threshold_logic_volume(
        field,
        extent=em.materialized_extent,
        deadband=config.logic_deadband,
    )
    sensors = sparse_sensor_readout(
        readout,
        logic,
        extent=em.materialized_extent,
        stride=config.sensor_stride,
    )

    latent_target = spatial_latent_6bit(logic, sensors)
    latent_state, latent_trace = refine_latent_fixed_point(
        latent_target,
        em.max_latent_iterations,
    )

    reconstructed, correction_trace, decoder_fixed_point = residual_correction_loop(
        source_bits,
        initial_flip_probability=em.initial_flip_probability,
        correction_fraction=em.correction_fraction,
        max_cycles=em.max_correction_cycles,
        seed=em.seed + 1,
    )
    final_errors = sum(
        expected != actual for expected, actual in zip(source_bits, reconstructed)
    )

    stats = logic_volume_stats(logic)
    center = em.materialized_extent // 2
    center_index = _flat_index(center, center, center, em.materialized_extent)

    return {
        "schema_version": "jarvisx.em-sig-3d-runtime.v1",
        "provenance": "simulated",
        "execution_contract": (
            "deterministic software 3D signalling runtime; no physical RF actuation "
            "or electromagnetic measurement"
        ),
        "config": {
            "em": asdict(em),
            "source_radius": config.source_radius,
            "source_z_offset": config.source_z_offset,
            "logic_deadband": config.logic_deadband,
            "sensor_stride": config.sensor_stride,
        },
        "pipeline": [
            "bit_source",
            "qpsk_encode",
            "source_program_3d",
            "interference_volume_3d",
            "ternary_logic_volume",
            "sparse_sensor_readout",
            "spatial_latent_6bit",
            "latent_fixed_point",
            "residual_decode",
            "verify",
        ],
        "layers": {
            "L0_source": {
                "bits": len(source_bits),
                "symbols": len(symbols),
                "rate_arithmetic": modeled_rate_arithmetic(em),
            },
            "L1_source_program_3d": {
                "source_count": len(program),
                "program": [asdict(row) for row in program],
                "encoding": (
                    "QPSK symbol magnitude/phase mapped to deterministic 3D source positions"
                ),
            },
            "L2_field_volume_3d": {
                "extent": em.materialized_extent,
                "voxels": em.materialized_extent**3,
                "raw_peak_before_normalization": raw_peak,
                "center_normalized_intensity": field[center_index],
                "boundary": (
                    "data-dependent scalar interference proxy; not Poynting flux "
                    "and not a Maxwell/FDTD/FEM solve"
                ),
            },
            "L3_logic_volume_3d": {
                "deadband": config.logic_deadband,
                "stats": stats,
                "center_readout": readout[center_index],
                "center_logic": logic[center_index],
                "logic_values": [-1, 0, 1],
            },
            "L4_sparse_sensors": {
                "stride": config.sensor_stride,
                "count": len(sensors),
                "samples": [asdict(row) for row in sensors],
            },
            "L5_spatial_latent": {
                "target_state_6bit": latent_target,
                "final_state_6bit": latent_state,
                "trace": _latent_trace_json(latent_trace),
                "fixed_point": latent_state == latent_target,
                "invertible": False,
                "note": (
                    "six-bit spatial signature is deliberately lossy; exact source "
                    "recovery requires residual information"
                ),
            },
            "L6_residual_decoder": {
                "trace": [asdict(row) for row in correction_trace],
                "final_errors": final_errors,
                "final_observed_ber": final_errors / len(source_bits),
                "fixed_point": decoder_fixed_point,
                "source_recovered": tuple(reconstructed) == tuple(source_bits),
            },
        },
        "verification": {
            "latent_fixed_point": latent_state == latent_target,
            "decoder_fixed_point": decoder_fixed_point,
            "source_recovered": tuple(reconstructed) == tuple(source_bits),
            "logic_volume_accounted": (
                stats["negative"] + stats["deadband"] + stats["positive"]
                == em.materialized_extent**3
            ),
        },
        "claim_boundary": {
            "verified": (
                "software source encoding, data-dependent 3D interference proxy, "
                "voxel quantization, sparse sensing, latent convergence, residual recovery"
            ),
            "not_verified": (
                "physical RF transmission, Maxwell-field accuracy, antenna hardware, "
                "measured electromagnetic logic, hardware throughput, or physical BER"
            ),
        },
    }


def write_report(report: dict[str, object], output: Path) -> None:
    """Write a stable JSON execution receipt."""

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bit-count", type=int, default=65_536)
    parser.add_argument("--materialized-extent", type=int, default=12)
    parser.add_argument("--field-sources", type=int, default=16)
    parser.add_argument("--logic-deadband", type=float, default=0.20)
    parser.add_argument("--sensor-stride", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/em-sig-3d-runtime.json"),
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    em = EMSigConfig(
        bit_count=args.bit_count,
        materialized_extent=args.materialized_extent,
        field_sources=args.field_sources,
        seed=args.seed,
    )
    config = EMSig3DConfig(
        em=em,
        logic_deadband=args.logic_deadband,
        sensor_stride=args.sensor_stride,
    )
    report = run_3d_runtime(config)
    write_report(report, args.output)
    verification = cast(dict[str, bool], report["verification"])
    print(
        "EM-Sig 3D runtime:",
        f"latent={verification['latent_fixed_point']}",
        f"decoder={verification['decoder_fixed_point']}",
        f"recovered={verification['source_recovered']}",
        f"output={args.output}",
    )


if __name__ == "__main__":
    main()
