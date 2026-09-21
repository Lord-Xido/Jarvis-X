"""Bounded EM-Sig 1000^3 research emulation.

This module is deliberately a *surrogate* software model. It maps bits -> QPSK
symbols -> a reduced scalar interference field -> a six-bit latent state -> a
residual-correction loop. It does not claim to solve Maxwell's equations on a
1000^3 mesh, model a fabricated RF/photonic device, or measure physical
throughput/BER. All timing values derived from configured rates are design
arithmetic, not hardware measurements.
"""

from __future__ import annotations

import argparse
import cmath
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class EMSigConfig:
    """Configuration for a bounded, reproducible EM-Sig proxy run."""

    logical_extent: int = 1000
    materialized_extent: int = 12
    bit_count: int = 65_536
    sample_rate_hz: float = 120.0e9
    bits_per_symbol: int = 2
    carrier_hz: float = 60.0e9
    array_side: int = 32
    field_sources: int = 16
    normalized_wavelength: float = 1.0
    hop_delay_seconds: float = 0.14e-12
    latent_bits: int = 6
    max_latent_iterations: int = 64
    max_correction_cycles: int = 8
    initial_flip_probability: float = 0.026
    correction_fraction: float = 0.78
    raw_channels: int = 3
    bytes_per_channel: int = 1
    sparse_residual_nodes_design: int = 10_000
    seed: int = 42

    def validate(self) -> None:
        if self.logical_extent <= 0 or self.materialized_extent < 3:
            raise ValueError("grid extents must be positive and materialized_extent >= 3")
        if self.bit_count < 2 or self.bit_count % self.bits_per_symbol:
            raise ValueError("bit_count must be divisible by bits_per_symbol")
        if self.sample_rate_hz <= 0.0 or self.bits_per_symbol <= 0:
            raise ValueError("sample_rate_hz and bits_per_symbol must be positive")
        if self.array_side <= 0 or self.field_sources <= 0:
            raise ValueError("array and source counts must be positive")
        if self.latent_bits != 6:
            raise ValueError("this reference emulation fixes the latent word at 6 bits")
        if not 0.0 <= self.initial_flip_probability <= 1.0:
            raise ValueError("initial_flip_probability must be in [0, 1]")
        if not 0.0 < self.correction_fraction <= 1.0:
            raise ValueError("correction_fraction must be in (0, 1]")
        if self.max_correction_cycles < 1 or self.max_latent_iterations < 1:
            raise ValueError("iteration limits must be positive")


@dataclass(frozen=True)
class CorrectionTrace:
    cycle: int
    errors: int
    observed_ber: float
    active_count: int
    active_fraction: float


@dataclass(frozen=True)
class LatentTrace:
    iteration: int
    state: int
    residual_bits: int


def create_world(bit_count: int, seed: int) -> tuple[int, ...]:
    """Create a deterministic binary world/source stream."""

    rng = random.Random(seed)
    return tuple(rng.getrandbits(1) for _ in range(bit_count))


def qpsk_map(bits: Sequence[int]) -> tuple[complex, ...]:
    """Map pairs of bits to unit-energy Gray-style QPSK symbols."""

    if len(bits) % 2:
        raise ValueError("QPSK input length must be even")
    inv_sqrt_2 = 1.0 / math.sqrt(2.0)
    table = {
        (0, 0): complex(+inv_sqrt_2, +inv_sqrt_2),
        (0, 1): complex(-inv_sqrt_2, +inv_sqrt_2),
        (1, 1): complex(-inv_sqrt_2, -inv_sqrt_2),
        (1, 0): complex(+inv_sqrt_2, -inv_sqrt_2),
    }
    return tuple(table[(bits[i], bits[i + 1])] for i in range(0, len(bits), 2))


def modeled_rate_arithmetic(config: EMSigConfig) -> dict[str, float]:
    """Return transparent timing arithmetic from configured rates.

    A sample is treated as one QPSK symbol, hence raw payload rate is
    sample_rate_hz * bits_per_symbol. No parallel-lane multiplier is assumed.
    """

    payload_rate_bps = config.sample_rate_hz * config.bits_per_symbol
    return {
        "configured_sample_rate_hz": config.sample_rate_hz,
        "configured_bits_per_symbol": float(config.bits_per_symbol),
        "modeled_payload_rate_bps": payload_rate_bps,
        "configured_stream_seconds": config.bit_count / payload_rate_bps,
        "one_billion_bits_seconds": 1.0e9 / payload_rate_bps,
        "one_trillion_bits_seconds": 1.0e12 / payload_rate_bps,
        "modeled_1000_hop_propagation_seconds": (
            config.logical_extent * config.hop_delay_seconds
        ),
    }


def array_factor(
    *,
    side: int,
    angles_deg: Iterable[float],
    steer_deg: float = 0.0,
) -> tuple[tuple[float, float], ...]:
    """Compute a normalized square-array factor on an angular cut.

    This is a far-field array-factor abstraction, not a full antenna solver.
    Element spacing is fixed at lambda/2.
    """

    spacing = 0.5
    k = 2.0 * math.pi
    steer = math.sin(math.radians(steer_deg))
    out: list[tuple[float, float]] = []
    normalizer = float(side * side)

    for angle_deg in angles_deg:
        s = math.sin(math.radians(angle_deg))
        total = 0j
        for y in range(side):
            for x in range(side):
                centered_x = x - (side - 1) / 2.0
                centered_y = y - (side - 1) / 2.0
                phase = k * spacing * (centered_x + centered_y) * (s - steer)
                total += cmath.exp(1j * phase)
        out.append((float(angle_deg), abs(total) / normalizer))
    return tuple(out)


def _source_positions(count: int, radius: float = 1.8) -> tuple[tuple[float, float, float], ...]:
    """Place deterministic sources on two staggered rings around the volume."""

    positions: list[tuple[float, float, float]] = []
    for i in range(count):
        angle = 2.0 * math.pi * i / count
        z = 0.7 if i % 2 == 0 else -0.7
        positions.append((radius * math.cos(angle), radius * math.sin(angle), z))
    return tuple(positions)


def scalar_interference_field(config: EMSigConfig) -> tuple[tuple[float, ...], float]:
    """Materialize a reduced scalar interference-intensity proxy.

    The returned scalar is |sum exp(j k r)/r|^2. It is useful for exercising
    inward focusing and thresholding, but it is *not* Poynting flux E x H and
    it is not a Maxwell/FDTD/FEM solution.
    """

    n = config.materialized_extent
    sources = _source_positions(config.field_sources)
    k = 2.0 * math.pi / config.normalized_wavelength
    values: list[float] = []
    maximum = 0.0

    for iz in range(n):
        z = -1.0 + 2.0 * iz / (n - 1)
        for iy in range(n):
            y = -1.0 + 2.0 * iy / (n - 1)
            for ix in range(n):
                x = -1.0 + 2.0 * ix / (n - 1)
                field = 0j
                for sx, sy, sz in sources:
                    dx, dy, dz = x - sx, y - sy, z - sz
                    r = max(math.sqrt(dx * dx + dy * dy + dz * dz), 1.0e-6)
                    field += cmath.exp(1j * k * r) / r
                intensity = field.real * field.real + field.imag * field.imag
                values.append(intensity)
                maximum = max(maximum, intensity)

    if maximum <= 0.0:
        return tuple(values), 0.0
    return tuple(value / maximum for value in values), maximum


def core_energy(field: Sequence[float], extent: int) -> float:
    """Average a central 3x3x3 region of a flattened cubic field."""

    if len(field) != extent**3:
        raise ValueError("field size does not match extent^3")
    center = extent // 2
    offsets = (-1, 0, 1)
    samples: list[float] = []
    for dz in offsets:
        z = min(max(center + dz, 0), extent - 1)
        for dy in offsets:
            y = min(max(center + dy, 0), extent - 1)
            for dx in offsets:
                x = min(max(center + dx, 0), extent - 1)
                samples.append(field[x + extent * (y + extent * z)])
    return sum(samples) / len(samples)


def quantize_latent_6bit(normalized_energy: float) -> int:
    """Map normalized core energy to an unsigned six-bit state."""

    clipped = max(0.0, min(1.0, normalized_energy))
    return max(0, min(63, int(round(clipped * 63.0))))


def refine_latent_fixed_point(target_state: int, max_iterations: int) -> tuple[int, tuple[LatentTrace, ...]]:
    """Move one bit at a time toward the six-bit target and record convergence."""

    target_state &= 0x3F
    state = 0
    trace: list[LatentTrace] = []

    for iteration in range(max_iterations):
        residual = state ^ target_state
        trace.append(LatentTrace(iteration, state, residual.bit_count()))
        if residual == 0:
            return state, tuple(trace)
        lowest_different_bit = residual & -residual
        state ^= lowest_different_bit
        state &= 0x3F

    residual = state ^ target_state
    trace.append(LatentTrace(max_iterations, state, residual.bit_count()))
    return state, tuple(trace)


def residual_correction_loop(
    source: Sequence[int],
    *,
    initial_flip_probability: float,
    correction_fraction: float,
    max_cycles: int,
    seed: int,
) -> tuple[tuple[int, ...], tuple[CorrectionTrace, ...], bool]:
    """Run a deterministic error-driven active-set correction loop.

    This is a software decoder surrogate. BER values are empirical only for the
    materialized bitstream passed here; a zero observed error count does not
    establish a physical BER of 1e-12 or any other smaller probability.
    """

    rng = random.Random(seed)
    reconstructed = list(source)
    for i in range(len(reconstructed)):
        if rng.random() < initial_flip_probability:
            reconstructed[i] ^= 1

    trace: list[CorrectionTrace] = []
    previous: tuple[int, ...] | None = None
    fixed_point = False

    for cycle in range(max_cycles + 1):
        residual = [i for i, (a, b) in enumerate(zip(source, reconstructed)) if a != b]
        errors = len(residual)
        active_count = len(source) if cycle == 0 else errors
        trace.append(
            CorrectionTrace(
                cycle=cycle,
                errors=errors,
                observed_ber=errors / len(source),
                active_count=active_count,
                active_fraction=active_count / len(source),
            )
        )

        snapshot = tuple(reconstructed)
        if errors == 0 and previous == snapshot:
            fixed_point = True
            break
        if errors == 0:
            previous = snapshot
            continue

        fix_count = max(1, int(math.ceil(errors * correction_fraction)))
        for index in residual[:fix_count]:
            reconstructed[index] = source[index]
        previous = snapshot

    return tuple(reconstructed), tuple(trace), fixed_point


def compression_design_estimate(config: EMSigConfig) -> dict[str, float | int]:
    """Compare optimistic and coordinate-explicit sparse payload estimates."""

    raw_bytes = config.logical_extent**3 * config.raw_channels * config.bytes_per_channel
    coord_bits_per_axis = math.ceil(math.log2(config.logical_extent))
    coordinate_bits = 3 * coord_bits_per_axis
    value_bits = config.latent_bits
    nodes = config.sparse_residual_nodes_design

    indexed_payload_bits = nodes * (coordinate_bits + value_bits) + config.latent_bits
    indexed_payload_bytes = math.ceil(indexed_payload_bits / 8)
    optimistic_payload_bytes = nodes * config.bytes_per_channel + math.ceil(config.latent_bits / 8)

    return {
        "raw_bytes": raw_bytes,
        "design_sparse_nodes": nodes,
        "coordinate_bits_per_axis": coord_bits_per_axis,
        "indexed_bits_per_node": coordinate_bits + value_bits,
        "indexed_payload_bytes_minimum": indexed_payload_bytes,
        "indexed_compression_ratio": raw_bytes / indexed_payload_bytes,
        "implicit_coordinate_payload_bytes_optimistic": optimistic_payload_bytes,
        "implicit_coordinate_compression_ratio_optimistic": raw_bytes / optimistic_payload_bytes,
    }


def run_emulation(config: EMSigConfig = EMSigConfig()) -> dict[str, object]:
    """Execute all five bounded emulation layers and return a JSON-native trace."""

    config.validate()
    bits = create_world(config.bit_count, config.seed)
    symbols = qpsk_map(bits)

    angular_cut = array_factor(
        side=config.array_side,
        angles_deg=range(-60, 61, 10),
        steer_deg=0.0,
    )
    field, unnormalized_peak = scalar_interference_field(config)
    energy = core_energy(field, config.materialized_extent)
    latent_target = quantize_latent_6bit(energy)
    latent_state, latent_trace = refine_latent_fixed_point(
        latent_target,
        config.max_latent_iterations,
    )

    reconstructed, correction_trace, fixed_point = residual_correction_loop(
        bits,
        initial_flip_probability=config.initial_flip_probability,
        correction_fraction=config.correction_fraction,
        max_cycles=config.max_correction_cycles,
        seed=config.seed + 1,
    )

    final_errors = sum(a != b for a, b in zip(bits, reconstructed))
    empirical_detection_floor = 1.0 / config.bit_count

    return {
        "schema_version": "jarvisx.em-sig-emulation.v1",
        "provenance": "simulated",
        "config": asdict(config),
        "layers": {
            "L1_serdes": {
                "bits": len(bits),
                "symbols": len(symbols),
                "rate_arithmetic": modeled_rate_arithmetic(config),
            },
            "L2_qpsk": {
                "constellation_points": 4,
                "mean_symbol_energy": sum(abs(s) ** 2 for s in symbols) / len(symbols),
                "carrier_hz_parameter": config.carrier_hz,
                "boundary": "QPSK symbol mapping only; no DAC or measured RF chain is modeled.",
            },
            "L3_spatial_field": {
                "configured_array_elements": config.array_side**2,
                "materialized_voxels": config.materialized_extent**3,
                "logical_voxels": config.logical_extent**3,
                "field_sources": config.field_sources,
                "array_factor_cut": [[a, magnitude] for a, magnitude in angular_cut],
                "scalar_field_peak_before_normalization": unnormalized_peak,
                "boundary": (
                    "Reduced scalar interference proxy |sum exp(jkr)/r|^2; not Poynting flux "
                    "and not a full 1000^3 Maxwell solve."
                ),
            },
            "L4_latent": {
                "core_energy_normalized": energy,
                "target_state_6bit": latent_target,
                "final_state_6bit": latent_state,
                "iterations": [asdict(row) for row in latent_trace],
                "latent_fixed_point": latent_state == latent_target,
            },
            "L5_residual_correction": {
                "trace": [asdict(row) for row in correction_trace],
                "final_errors": final_errors,
                "final_observed_ber": final_errors / config.bit_count,
                "empirical_single_error_resolution": empirical_detection_floor,
                "fixed_point": fixed_point,
                "boundary": (
                    "Observed BER is finite-stream software error fraction. Zero observed errors "
                    "does not verify a physical BER below 1/bit_count."
                ),
            },
        },
        "compression_design_estimate": compression_design_estimate(config),
        "claim_boundary": {
            "verified_by_this_run": (
                "deterministic software transforms, bounded scalar-field proxy, six-bit quantization, "
                "and numerical residual convergence"
            ),
            "not_verified_by_this_run": (
                "120 GSa/s hardware, femtosecond phase lock, physical 1000^3 Poynting transport, "
                "114 Peta-nodes/s, 18.5 GPix/s, 1e-12 physical BER, or fabricated-device compression"
            ),
        },
    }


def write_report(report: dict[str, object], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/em-sig-emulation.json"))
    parser.add_argument("--materialized-extent", type=int, default=12)
    parser.add_argument("--bit-count", type=int, default=65_536)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config = EMSigConfig(
        materialized_extent=args.materialized_extent,
        bit_count=args.bit_count,
        seed=args.seed,
    )
    report = run_emulation(config)
    write_report(report, args.output)
    fixed = bool(report["layers"]["L5_residual_correction"]["fixed_point"])  # type: ignore[index]
    return 0 if fixed else 1


if __name__ == "__main__":
    raise SystemExit(main())
