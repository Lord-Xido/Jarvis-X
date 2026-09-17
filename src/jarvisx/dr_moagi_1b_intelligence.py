"""Dr Moagi exact-1B sparse multimodal 3D intelligence reference engine.

This module composes the merged DM3D byte-exact spatial codec with an exact
1,000,000,000-address INT8 weight space. The 1B weight vector is virtual by
default: deterministic 4 KiB pages are generated on demand and only trained
overlays are persisted.

The system is operational software, not a pretrained foundation model.
Capability depends on learned overlays, data, objectives, validation and
backend kernels. "Intelligence" here denotes observable adaptive software
behaviour: multimodal ingest, shared representation, residual refinement,
memory, generation, validation and rollback.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

from .dm3d_rom import (
    LATENT_BYTES as TILE_LATENT_BYTES,
    TILE_BYTES,
    correct_with_residual,
    decode_tile,
    deterministic_tile,
    encode_tile,
    mse_from_residual,
    refine_latent,
    residual_field,
)

ONE_BILLION = 1_000_000_000
BLOCK_WEIGHTS = 4096

WEIGHT_PARTITIONS: Dict[str, int] = {
    "text_adapter": 80_000_000,
    "image_adapter": 100_000_000,
    "audio_adapter": 70_000_000,
    "video_adapter": 110_000_000,
    "volume3d_adapter": 90_000_000,
    "mesh_adapter": 50_000_000,
    "code_adapter": 80_000_000,
    "shared_fusion_core": 260_000_000,
    "omega_refiner": 80_000_000,
    "multimodal_heads": 80_000_000,
}
if sum(WEIGHT_PARTITIONS.values()) != ONE_BILLION:
    raise RuntimeError("1B weight layout is not exact")

MODALITIES = ("text", "image", "audio", "video", "volume3d", "mesh", "code")
ADAPTER_BANK = {
    "text": "text_adapter",
    "image": "image_adapter",
    "audio": "audio_adapter",
    "video": "video_adapter",
    "volume3d": "volume3d_adapter",
    "mesh": "mesh_adapter",
    "code": "code_adapter",
}


@dataclass(frozen=True)
class BankLayout:
    name: str
    offset: int
    count: int

    @property
    def end(self) -> int:
        return self.offset + self.count

    @property
    def blocks(self) -> int:
        return (self.count + BLOCK_WEIGHTS - 1) // BLOCK_WEIGHTS


def _build_layout() -> Dict[str, BankLayout]:
    result: Dict[str, BankLayout] = {}
    offset = 0
    for name, count in WEIGHT_PARTITIONS.items():
        result[name] = BankLayout(name, offset, count)
        offset += count
    if offset != ONE_BILLION:
        raise AssertionError(offset)
    return result


LAYOUT = _build_layout()


class VirtualInt8Weights:
    """Sparse page store for the exact one-billion-address INT8 weight vector."""

    def __init__(self, seed: int = 7, overlay_dir: Optional[str | Path] = None):
        self.seed = int(seed)
        self.overlay_dir = Path(overlay_dir) if overlay_dir else None
        self._cache: Dict[Tuple[str, int], Tuple[int, ...]] = {}
        if self.overlay_dir:
            self.overlay_dir.mkdir(parents=True, exist_ok=True)

    def _layout(self, bank: str) -> BankLayout:
        try:
            return LAYOUT[bank]
        except KeyError as exc:
            raise KeyError(f"unknown weight bank {bank!r}") from exc

    def block_len(self, bank: str, block_id: int) -> int:
        layout = self._layout(bank)
        if not 0 <= block_id < layout.blocks:
            raise IndexError(f"{bank} block {block_id} outside [0,{layout.blocks})")
        start = block_id * BLOCK_WEIGHTS
        return min(BLOCK_WEIGHTS, layout.count - start)

    def _path(self, bank: str, block_id: int) -> Path:
        if self.overlay_dir is None:
            raise RuntimeError("overlay directory is disabled")
        return self.overlay_dir / f"{bank}.{block_id:08d}.i8"

    def _generate(self, bank: str, block_id: int, n: int) -> Tuple[int, ...]:
        layout = self._layout(bank)
        base = layout.offset + block_id * BLOCK_WEIGHTS
        seed = f"{self.seed}:{bank}:{base}".encode()
        out = []
        counter = 0
        while len(out) < n:
            digest = hashlib.blake2b(seed + counter.to_bytes(8, "little"), digest_size=64).digest()
            out.extend((byte & 31) - 16 for byte in digest)
            counter += 1
        return tuple(out[:n])

    def read_block(self, bank: str, block_id: int) -> Tuple[int, ...]:
        key = (bank, int(block_id))
        if key in self._cache:
            return self._cache[key]
        n = self.block_len(bank, block_id)
        if self.overlay_dir:
            path = self._path(bank, block_id)
            if path.exists():
                values = array("b")
                with path.open("rb") as handle:
                    values.fromfile(handle, n)
                if len(values) != n:
                    raise ValueError(f"corrupt weight overlay: {path}")
                result = tuple(int(v) for v in values)
                self._cache[key] = result
                return result
        result = self._generate(bank, block_id, n)
        self._cache[key] = result
        return result

    def write_block(self, bank: str, block_id: int, values: Iterable[int]) -> None:
        vals = tuple(max(-128, min(127, int(v))) for v in values)
        expected = self.block_len(bank, block_id)
        if len(vals) != expected:
            raise ValueError(f"expected {expected} values, got {len(vals)}")
        self._cache[(bank, block_id)] = vals
        if self.overlay_dir:
            payload = array("b", vals)
            with self._path(bank, block_id).open("wb") as handle:
                payload.tofile(handle)

    def manifest(self) -> dict:
        return {
            "logical_weights": ONE_BILLION,
            "logical_raw_int8_bytes": ONE_BILLION,
            "block_weights": BLOCK_WEIGHTS,
            "banks": {
                name: {
                    "offset": layout.offset,
                    "count": layout.count,
                    "blocks": layout.blocks,
                }
                for name, layout in LAYOUT.items()
            },
        }


@dataclass(frozen=True)
class EngineConfig:
    latent_edge: int = 4
    active_experts: int = 4
    refine_steps: int = 3
    residual_gain: float = 0.25
    omega_decay: float = 0.9
    seed: int = 7

    @property
    def latent_dim(self) -> int:
        return self.latent_edge**3

    def validate(self) -> None:
        if not 2 <= self.latent_edge <= 16:
            raise ValueError("latent_edge must be in [2,16]")
        if not 1 <= self.active_experts <= 16:
            raise ValueError("active_experts must be in [1,16]")
        if not 0 <= self.refine_steps <= 32:
            raise ValueError("refine_steps must be in [0,32]")
        if not 0.0 < self.residual_gain <= 1.0:
            raise ValueError("residual_gain must be in (0,1]")
        if not 0.0 <= self.omega_decay < 1.0:
            raise ValueError("omega_decay must be in [0,1)")


@dataclass
class IntelligenceState:
    modality_latents: Dict[str, list[float]]
    fused: list[float]
    refined: list[float]
    omega: list[float]
    residual_norm: float
    active_weight_blocks: int


@dataclass(frozen=True)
class TileCodecResult:
    latent_bytes: int
    pre_correction_mse: float
    exact: bool
    source_sha256: str
    corrected_sha256: str


@dataclass(frozen=True)
class EvolutionResult:
    bank: str
    block_id: int
    before_loss: float
    after_loss: float
    accepted: bool


def _raw_bytes(value: object) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, str):
        return value.encode("utf-8")
    if isinstance(value, Path):
        return value.read_bytes()
    if hasattr(value, "tobytes"):
        return value.tobytes()
    return repr(value).encode("utf-8")


def bytes_to_latent(data: object, dim: int) -> list[float]:
    raw = _raw_bytes(data) or b"\0"
    sums = [0.0] * dim
    counts = [0] * dim
    for i, byte in enumerate(raw):
        j = i % dim
        sums[j] += byte - 127.5
        counts[j] += 1
    return [sums[i] / (127.5 * max(1, counts[i])) for i in range(dim)]


def _mse(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError("shape mismatch")
    return sum((x - y) ** 2 for x, y in zip(a, b)) / max(1, len(a))


def _norm(a: Sequence[float]) -> float:
    return math.sqrt(sum(x * x for x in a))


def _route_ids(payload: bytes, bank: str, count: int) -> list[int]:
    blocks = LAYOUT[bank].blocks
    digest = hashlib.blake2b(payload + bank.encode("ascii"), digest_size=64).digest()
    ids = []
    counter = 0
    while len(ids) < count:
        if counter * 4 + 4 > len(digest):
            digest = hashlib.blake2b(digest, digest_size=64).digest()
            counter = 0
        offset = counter * 4
        ids.append(int.from_bytes(digest[offset : offset + 4], "little") % blocks)
        counter += 1
    return ids


class SparseExpertOperator:
    def __init__(self, weights: VirtualInt8Weights, config: EngineConfig):
        self.weights = weights
        self.config = config

    def apply(self, bank: str, vector: Sequence[float], route_payload: bytes) -> list[float]:
        d = self.config.latent_dim
        if len(vector) != d:
            raise ValueError(f"expected latent size {d}, got {len(vector)}")
        accum = [0.0] * d
        ids = _route_ids(route_payload, bank, self.config.active_experts)
        for block_id in ids:
            page = self.weights.read_block(bank, block_id)
            gate_sum = 0.0
            for i, value in enumerate(vector):
                gate_sum += value * (page[(2 * i) % len(page)] / 16.0)
            gate = math.tanh(gate_sum / d)
            for i in range(d):
                accum[i] += gate * (page[(2 * i + 1) % len(page)] / 16.0)
        scale = 1.0 / len(ids)
        return [float(vector[i]) + accum[i] * scale for i in range(d)]


class DrMoagi1B3DIntelligenceEngine:
    """Sparse multimodal -> shared 3D latent -> residual refinement -> decode runtime."""

    def __init__(
        self,
        config: Optional[EngineConfig] = None,
        overlay_dir: Optional[str | Path] = None,
    ):
        self.config = config or EngineConfig()
        self.config.validate()
        self.weights = VirtualInt8Weights(self.config.seed, overlay_dir)
        self.operator = SparseExpertOperator(self.weights, self.config)
        self.omega = [0.0] * self.config.latent_dim

    def encode_modality(self, modality: str, payload: object) -> list[float]:
        if modality not in ADAPTER_BANK:
            raise KeyError(f"unsupported modality {modality!r}")
        raw = _raw_bytes(payload)
        vector = bytes_to_latent(raw, self.config.latent_dim)
        return self.operator.apply(ADAPTER_BANK[modality], vector, raw)

    def fuse(self, latents: Mapping[str, Sequence[float]]) -> list[float]:
        if not latents:
            return [0.0] * self.config.latent_dim
        names = sorted(latents)
        d = self.config.latent_dim
        fused = [sum(float(latents[name][i]) for name in names) / len(names) for i in range(d)]
        route = b"|".join(name.encode("ascii") for name in names)
        fused = self.operator.apply("shared_fusion_core", fused, route + b":0")
        return self.operator.apply("shared_fusion_core", fused, route + b":1")

    def refine(self, latent: Sequence[float]) -> Tuple[list[float], float]:
        z = [float(v) for v in latent]
        initial = z.copy()
        for step in range(self.config.refine_steps):
            route = f"refine:{step}:".encode() + _float_signature(z)
            proposal = self.operator.apply("omega_refiner", z, route)
            residual = [proposal[i] - z[i] for i in range(len(z))]
            z = [z[i] + self.config.residual_gain * residual[i] for i in range(len(z))]
        return z, _norm([z[i] - initial[i] for i in range(len(z))])

    def forward(self, inputs: Mapping[str, object]) -> IntelligenceState:
        unknown = set(inputs) - set(MODALITIES)
        if unknown:
            raise KeyError(f"unsupported modalities: {sorted(unknown)}")
        if not inputs:
            raise ValueError("at least one modality is required")
        latents = {name: self.encode_modality(name, payload) for name, payload in inputs.items()}
        fused = self.fuse(latents)
        refined, residual_norm = self.refine(fused)
        beta = self.config.omega_decay
        self.omega = [beta * old + (1.0 - beta) * new for old, new in zip(self.omega, refined)]
        refined = [0.8 * new + 0.2 * mem for new, mem in zip(refined, self.omega)]
        calls = len(inputs) + 2 + self.config.refine_steps
        return IntelligenceState(
            modality_latents=latents,
            fused=fused,
            refined=refined,
            omega=self.omega.copy(),
            residual_norm=residual_norm,
            active_weight_blocks=calls * self.config.active_experts,
        )

    def decode_bytes(
        self,
        state: IntelligenceState,
        modality: str,
        output_bytes: int = 1024,
    ) -> bytes:
        if modality not in MODALITIES:
            raise KeyError(f"unsupported modality {modality!r}")
        n = max(0, min(int(output_bytes), 1_000_000))
        if n == 0:
            return b""
        route = modality.encode("ascii") + _float_signature(state.refined)
        head = self.operator.apply("multimodal_heads", state.refined, route)
        seed = hashlib.sha256(modality.encode("ascii") + _float_signature(head, full=True)).digest()
        output = bytearray()
        counter = 0
        while len(output) < n:
            output.extend(hashlib.sha256(seed + counter.to_bytes(8, "little")).digest())
            counter += 1
        return bytes(output[:n])

    def process_tile(self, source: bytes | bytearray, quant_step: int = 8) -> TileCodecResult:
        if len(source) != TILE_BYTES:
            raise ValueError(f"3D tile must contain exactly {TILE_BYTES} bytes")
        tile = bytearray(source)
        latent = encode_tile(tile, quant_step=quant_step)
        latent = refine_latent(tile, latent, iterations=4, alpha_num=1, alpha_den=1)
        reconstruction = decode_tile(latent)
        residual = residual_field(tile, reconstruction)
        mse = mse_from_residual(residual)
        corrected = correct_with_residual(reconstruction, residual)
        source_hash = hashlib.sha256(tile).hexdigest()
        corrected_hash = hashlib.sha256(corrected).hexdigest()
        return TileCodecResult(
            latent_bytes=len(latent),
            pre_correction_mse=mse,
            exact=corrected == tile and source_hash == corrected_hash,
            source_sha256=source_hash,
            corrected_sha256=corrected_hash,
        )

    def evolve_adapter(
        self,
        modality: str,
        payload: object,
        target: object,
        max_trials: int = 16,
    ) -> EvolutionResult:
        """Bounded local hill-climb with validation and automatic rollback."""
        if modality not in ADAPTER_BANK:
            raise KeyError(modality)
        raw = _raw_bytes(payload)
        target_latent = bytes_to_latent(target, self.config.latent_dim)
        bank = ADAPTER_BANK[modality]
        block_id = _route_ids(raw, bank, self.config.active_experts)[0]
        original = self.weights.read_block(bank, block_id)

        def loss() -> float:
            return _mse(self.encode_modality(modality, raw), target_latent)

        baseline = loss()
        best_loss = baseline
        best = original
        trials = max(0, min(int(max_trials), min(len(original), 64)))
        for position in range(trials):
            for delta in (-1, 1):
                candidate = list(original)
                candidate[position] = max(-128, min(127, candidate[position] + delta))
                self.weights.write_block(bank, block_id, candidate)
                candidate_loss = loss()
                if math.isfinite(candidate_loss) and candidate_loss < best_loss:
                    best_loss = candidate_loss
                    best = tuple(candidate)
        accepted = best_loss < baseline
        self.weights.write_block(bank, block_id, best if accepted else original)
        return EvolutionResult(bank, block_id, baseline, best_loss if accepted else baseline, accepted)

    def manifest(self) -> dict:
        result = self.weights.manifest()
        result.update(
            {
                "modalities": list(MODALITIES),
                "latent_geometry": [
                    self.config.latent_edge,
                    self.config.latent_edge,
                    self.config.latent_edge,
                ],
                "latent_dim": self.config.latent_dim,
                "active_experts_per_call": self.config.active_experts,
                "refine_steps": self.config.refine_steps,
                "virtual_byte_space": "10^9 x 10^9 x 10^9 = 10^27 byte positions",
                "tile_codec": "64^3 bytes -> 8^3 latent -> residual correction -> exact verify",
                "authority": "candidate outputs require enclosing Jarvis-X validation/commit",
            }
        )
        return result


def _float_signature(values: Sequence[float], full: bool = False) -> bytes:
    chosen = values if full else values[: min(16, len(values))]
    text = ",".join(f"{float(v):.9g}" for v in chosen)
    return text.encode("ascii")


def allocate_logical_weight_files(directory: str | Path) -> dict:
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    files = []
    for name, count in WEIGHT_PARTITIONS.items():
        path = root / f"{name}.i8"
        with path.open("wb") as handle:
            handle.truncate(count)
        files.append({"path": str(path), "logical_bytes": count})
    return {
        "logical_weights": ONE_BILLION,
        "logical_bytes": sum(item["logical_bytes"] for item in files),
        "files": files,
        "note": "capacity allocation only; deterministic base initialization remains virtual",
    }


def smoke() -> dict:
    engine = DrMoagi1B3DIntelligenceEngine(
        EngineConfig(latent_edge=4, active_experts=2, refine_steps=2, seed=7)
    )
    state = engine.forward(
        {
            "text": "Dr Moagi 3D residual intelligence engine",
            "image": bytes(range(256)) * 4,
            "audio": bytes((i * 7) & 255 for i in range(2048)),
            "volume3d": bytes((i * 11) & 255 for i in range(4096)),
            "code": b"x_hat = decode(encode(x)); residual = x - x_hat",
        }
    )
    output = engine.decode_bytes(state, "text", 256)
    evolution = engine.evolve_adapter("text", b"abc", b"abd", max_trials=8)
    return {
        "logical_weights": ONE_BILLION,
        "modalities": sorted(state.modality_latents),
        "latent_dim": engine.config.latent_dim,
        "active_weight_blocks": state.active_weight_blocks,
        "active_weight_values_upper_bound": state.active_weight_blocks * BLOCK_WEIGHTS,
        "residual_refinement_norm": state.residual_norm,
        "output_sha256": hashlib.sha256(output).hexdigest(),
        "evolution": {
            "bank": evolution.bank,
            "block_id": evolution.block_id,
            "before_loss": evolution.before_loss,
            "after_loss": evolution.after_loss,
            "accepted": evolution.accepted,
        },
        "status": "PASS",
    }


def tile_smoke(pattern: int = 1) -> dict:
    engine = DrMoagi1B3DIntelligenceEngine(
        EngineConfig(latent_edge=4, active_experts=1, refine_steps=1, seed=7)
    )
    source = deterministic_tile(0, 0, 0, pattern)
    result = engine.process_tile(source)
    return {
        "tile_bytes": TILE_BYTES,
        "latent_bytes": result.latent_bytes,
        "expected_latent_bytes": TILE_LATENT_BYTES,
        "pre_correction_mse": result.pre_correction_mse,
        "exact": result.exact,
        "source_sha256": result.source_sha256,
        "corrected_sha256": result.corrected_sha256,
        "status": "PASS" if result.exact else "FAIL",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dr Moagi exact-1B multimodal 3D intelligence engine")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("manifest")
    sub.add_parser("smoke")

    tile = sub.add_parser("tile-smoke")
    tile.add_argument("--pattern", type=int, default=1)

    allocate = sub.add_parser("allocate")
    allocate.add_argument("directory")

    process = sub.add_parser("process")
    process.add_argument("--input", action="append", default=[], metavar="MODALITY=PATH")
    process.add_argument("--text")
    process.add_argument("--output-modality", choices=MODALITIES, default="text")
    process.add_argument("--output-bytes", type=int, default=1024)
    process.add_argument("--out", default="./dr-moagi-1b-output")

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "manifest":
        print(json.dumps(DrMoagi1B3DIntelligenceEngine().manifest(), indent=2))
        return 0
    if args.command == "smoke":
        print(json.dumps(smoke(), indent=2))
        return 0
    if args.command == "tile-smoke":
        report = tile_smoke(args.pattern)
        print(json.dumps(report, indent=2))
        return 0 if report["status"] == "PASS" else 1
    if args.command == "allocate":
        print(json.dumps(allocate_logical_weight_files(args.directory), indent=2))
        return 0
    if args.command == "process":
        inputs: Dict[str, object] = {}
        if args.text is not None:
            inputs["text"] = args.text
        for spec in args.input:
            if "=" not in spec:
                raise SystemExit("--input must be MODALITY=PATH")
            modality, path = spec.split("=", 1)
            if modality not in MODALITIES:
                raise SystemExit(f"unsupported modality: {modality}")
            inputs[modality] = Path(path).read_bytes()
        engine = DrMoagi1B3DIntelligenceEngine()
        state = engine.forward(inputs)
        generated = engine.decode_bytes(state, args.output_modality, args.output_bytes)
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        payload_path = out / f"generated-{args.output_modality}.bin"
        payload_path.write_bytes(generated)
        report = {
            "inputs": sorted(inputs),
            "output_modality": args.output_modality,
            "output_bytes": len(generated),
            "output_sha256": hashlib.sha256(generated).hexdigest(),
            "latent_dim": engine.config.latent_dim,
            "active_weight_blocks": state.active_weight_blocks,
            "residual_refinement_norm": state.residual_norm,
            "payload": str(payload_path),
        }
        (out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
