"""Opt-in Phi_M -> Jarvis-X GridTokenCodec A/B experiment; never deploys or commits.

Tests real finite lexical 3D coordinate encode/decode with direct Phi_M and a
quantized residual side-channel. Reported payload sizes exclude dictionaries,
metadata and framing; no compression or trained semantic-model claims.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .dr_moagi_phi_m_3d import PhiMConfig, phi_m
from .moagi_3d_geometric_intelligence import GridTokenCodec


@dataclass(frozen=True)
class ProbeConfig:
    word_count: int = 128
    repetitions: int = 7
    seed: int = 20261029
    time: float = 1.5
    residual_quantum: float = 0.0001

    def __post_init__(self) -> None:
        for name, upper in (("word_count", 1000), ("repetitions", 21)):
            val = getattr(self, name)
            if type(val) is not int or not 1 <= val <= upper:
                raise ValueError(f"{name} must be an integer between 1 and {upper}")
        if type(self.seed) is not int or not 0 <= self.seed <= 2**32-1:
            raise ValueError("seed must be a uint32 integer")
        if not math.isfinite(self.time) or not 0.0 <= self.time <= 10.0:
            raise ValueError("time must be finite and in [0,10]")
        if not math.isfinite(self.residual_quantum) or not 1e-6 <= self.residual_quantum <= 0.001:
            raise ValueError("residual_quantum must be in [1e-6,0.001]")


def _corpus(config: ProbeConfig) -> tuple[GridTokenCodec, str]:
    # Fully populated finite vocabulary ensures all in-range 3D positions decode.
    vocab = [" "] + [f"w{i:04d}" for i in range(4095)]
    codec = GridTokenCodec(max_tokens=4096, vocabulary=vocab)
    rng = np.random.default_rng(config.seed)
    words = [vocab[i] for i in rng.integers(1, len(vocab), size=config.word_count)]
    return codec, " ".join(words)


def _run_once(codec: GridTokenCodec, text: str, config: ProbeConfig, mode: str,
              phi_config: PhiMConfig) -> dict[str, float | int | bool]:
    if mode not in ("baseline", "direct_phi", "residual_phi"):
        raise ValueError("invalid experiment mode")
    encoded = codec.encode(text)
    x = np.asarray(encoded.points, dtype=np.float64)
    n = len(x)
    if n == 0 or n > phi_config.max_points:
        raise ValueError("token count exceeds operator budget")
    if mode == "baseline":
        packed = np.asarray(x, dtype="<f4").tobytes()
        recovered = np.frombuffer(packed, dtype="<f4").reshape(-1, 3).astype(np.float64)
        payload_bytes = len(packed)  # Actual f32 buffer; framing excluded
    else:
        warped = phi_m(x, config.time, phi_config)
        if mode == "direct_phi":
            recovered = np.asarray(warped, dtype="<f4").astype(np.float64)
            payload_bytes = n * 12
        else:
            warped32 = np.asarray(warped, dtype="<f4")
            raw_codes = np.rint((x - warped32.astype(np.float64)) / config.residual_quantum)
            if (not np.isfinite(raw_codes).all() or
                    np.any(raw_codes < -32768) or np.any(raw_codes > 32767)):
                raise RuntimeError("residual exceeds signed int16 capacity; candidate rejected")
            codes = raw_codes.astype("<i2")
            # Decode only from an actual f32 warp packet + i16 side-channel.
            recovered = (np.frombuffer(warped32.tobytes(), dtype="<f4")
                         .reshape(-1, 3).astype(np.float64)
                         + np.frombuffer(codes.tobytes(), dtype="<i2")
                         .reshape(-1, 3).astype(np.float64) * config.residual_quantum)
            payload_bytes = n * 18
            # Boundary values at +/-1 can exceed the cube by half a quantum.
            recovered = np.clip(recovered, -1.0, 1.0)
    decoded = codec.decode(recovered.tolist(), method="nearest")
    matches = sum(a == b for a, b in zip(decoded.tokens, encoded.tokens))
    return {
        "tokens": n,
        "exact_text": decoded.text == text,
        "token_accuracy": matches / n,
        "coordinate_rmse": float(np.sqrt(np.mean(np.square(x - recovered)))),
        "coordinate_payload_bytes": payload_bytes,
        "ideal_uint16_token_id_bytes": n * 2,
    }


def run_probe(config: ProbeConfig = ProbeConfig(),
              phi_config: PhiMConfig = PhiMConfig()) -> dict[str, object]:
    """Run paired timed experiments with same corpus and bounded work budget."""
    codec, text = _corpus(config)
    modes = ("baseline", "direct_phi", "residual_phi")
    rng = np.random.default_rng(config.seed + 1)
    measurements: dict[str, list[float]] = {mode: [] for mode in modes}
    outcomes: dict[str, dict[str, float | int | bool]] = {}
    for mode in modes:  # SciPy lazy imports/cache setup excluded from timer
        outcomes[mode] = _run_once(codec, text, config, mode, phi_config)
    for _ in range(config.repetitions):
        for index in rng.permutation(3):
            mode = modes[int(index)]
            start = time.perf_counter_ns()
            result = _run_once(codec, text, config, mode, phi_config)
            elapsed_ms = (time.perf_counter_ns() - start) / 1e6
            assert result == outcomes[mode], "non-deterministic codec roundtrip"
            measurements[mode].append(elapsed_ms)
    medians = {mode: float(np.median(measurements[mode])) for mode in modes}
    speed_ratios = {mode: medians[mode] / medians["baseline"] for mode in modes}
    baseline = outcomes["baseline"]
    residual = outcomes["residual_phi"]
    # A guarded shadow recommendation, not an operational system transition.
    eligible = bool(residual["exact_text"] and
                    residual["coordinate_payload_bytes"] <= baseline["coordinate_payload_bytes"] and
                    medians["residual_phi"] <= medians["baseline"])
    return {
        "status": "shadow_only_no_runtime_mutation",
        "parameters": asdict(config),
        "phi_parameters": asdict(phi_config),
        "corpus": "synthetic_fixed_grid_4096_vocabulary",
        "methods": outcomes,
        "median_end_to_end_ms": medians,
        "latency_ratio_to_baseline": speed_ratios,
        "payload_notes": "Coordinate payload excludes vocabulary, framing and metadata; uint16 IDs are an idealized lower bound, not an existing packet format.",
        "promotion_eligible": eligible,
        "selected": "residual_phi" if eligible else "baseline",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--word-count", type=int, default=128)
    ap.add_argument("--repetitions", type=int, default=7)
    ap.add_argument("--seed", type=int, default=20261029)
    ap.add_argument("--time", type=float, default=1.5)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    cfg = ProbeConfig(args.word_count, args.repetitions, args.seed, args.time)
    result = run_probe(cfg)
    data = json.dumps(result, sort_keys=True, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(data + "\n", encoding="utf-8")
    print(data)


if __name__ == "__main__":
    main()
