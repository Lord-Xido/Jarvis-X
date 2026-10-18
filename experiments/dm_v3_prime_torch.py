#!/usr/bin/env python3
"""Executable DM-V3-PRIME 3D auto-codec research baseline.

The experiment binds the mathematical DM-V3-PRIME specification to a concrete
PyTorch implementation while preserving the deterministic Jarvis-X core as a
separate dependency-free control plane.

Implemented mechanisms:

* N^3 spatial context through 3D convolutions;
* recursive latent refinement with a real FFT-domain spectral operator;
* straight-through scalar quantization during training;
* a learned factorized Laplace entropy model for rate optimisation;
* a real zlib-compressed int16 latent bitstream for evaluation;
* synchronized CUDA timing when applicable;
* transactional Pi_{H,Lambda} verification with rollback;
* an explicit, measured 1000x speed target that is never inferred from virtual
  recursion depth.

This remains a bounded research prototype.  Training weights does not by itself
make the architecture execute 1000x faster; that target must be established by
measurement against a declared baseline.
"""

from __future__ import annotations

import argparse
import copy
import math
import struct
import time
import zlib
from dataclasses import dataclass
from statistics import median
from typing import Callable, TypeVar

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from jarvisx.dm_v3_prime_control import DMV3Metrics, PiHLambdaGate, VerificationPolicy

T = TypeVar("T")

MAGIC = b"DMV3PRM\0"
BITSTREAM_VERSION = 1
HEADER = struct.Struct("<8sHIIfI")


@dataclass(frozen=True)
class Evaluation:
    metrics: DMV3Metrics
    psnr_db: float
    bits_per_voxel: float
    bitstream_bytes: int


class RecursiveSpectralRefiner(nn.Module):
    """Bounded inward latent recursion with FFT-domain spectral gating."""

    def __init__(self, latent_dim: int, depth: int = 2, step_size: float = 0.1) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be at least one")
        if step_size <= 0.0:
            raise ValueError("step_size must be positive")

        self.depth = depth
        self.step_size = step_size
        self.norm = nn.LayerNorm(latent_dim)
        self.spectral_gain = nn.Parameter(torch.zeros(latent_dim // 2 + 1))
        self.residual = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, latent_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        for _ in range(self.depth):
            normalized = self.norm(z)
            spectrum = torch.fft.rfft(normalized, dim=-1)
            gain = torch.sigmoid(self.spectral_gain)
            upwelled = torch.fft.irfft(spectrum * gain, n=z.shape[-1], dim=-1)
            z = z + self.step_size * self.residual(upwelled)
        return z


class FactorizedLaplaceEntropyModel(nn.Module):
    """Learned factorized latent model used to estimate differentiable rate."""

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.log_scale = nn.Parameter(torch.zeros(latent_dim))

    @staticmethod
    def _cdf(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        negative = 0.5 * torch.exp(x / scale)
        positive = 1.0 - 0.5 * torch.exp(-x / scale)
        return torch.where(x < 0.0, negative, positive)

    def bits(self, centers: torch.Tensor, delta: float) -> torch.Tensor:
        scale = F.softplus(self.log_scale) + 1e-4
        half_step = 0.5 * delta
        upper = self._cdf(centers + half_step, scale)
        lower = self._cdf(centers - half_step, scale)
        probability = (upper - lower).clamp_min(1e-9)
        return -torch.log2(probability)


class DMV3PrimeEngine(nn.Module):
    """32^3 convolutional auto-codec with recursive spectral latent refinement."""

    def __init__(self, latent_dim: int = 64, recursion_depth: int = 2) -> None:
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(16),
            nn.ELU(),
            nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ELU(),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8 * 8, latent_dim),
        )

        self.refiner = RecursiveSpectralRefiner(latent_dim, depth=recursion_depth)
        self.entropy_model = FactorizedLaplaceEntropyModel(latent_dim)

        self.decoder_linear = nn.Linear(latent_dim, 32 * 8 * 8 * 8)
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(16),
            nn.ELU(),
            nn.ConvTranspose3d(16, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode_continuous(self, x: torch.Tensor) -> torch.Tensor:
        return self.refiner(self.encoder(x))

    def quantize_ste(self, z: torch.Tensor, delta: float) -> tuple[torch.Tensor, torch.Tensor]:
        scaled = z / delta
        q = torch.round(scaled)
        dequantized = q * delta
        z_ste = z + (dequantized - z).detach()
        return z_ste, q

    def decode_latent(self, z: torch.Tensor) -> torch.Tensor:
        flat = self.decoder_linear(z)
        return self.decoder(flat.view(-1, 32, 8, 8, 8))

    def forward(
        self,
        x: torch.Tensor,
        delta: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encode_continuous(x)
        z_q, _ = self.quantize_ste(z, delta)
        x_hat = self.decode_latent(z_q)
        rate_bits = self.entropy_model.bits(z_q, delta)
        return x_hat, z_q, rate_bits

    @torch.no_grad()
    def encode_bitstream(self, x: torch.Tensor, delta: float) -> bytes:
        self.eval()
        z = self.encode_continuous(x)
        q = torch.round(z / delta).clamp(-32768, 32767).to(torch.int16)
        q_np = q.detach().cpu().numpy().astype("<i2", copy=False)
        raw = q_np.tobytes(order="C")
        checksum = zlib.crc32(raw) & 0xFFFFFFFF
        compressed = zlib.compress(raw, level=9)
        header = HEADER.pack(
            MAGIC,
            BITSTREAM_VERSION,
            q_np.shape[0],
            q_np.shape[1],
            float(delta),
            checksum,
        )
        return header + compressed

    @torch.no_grad()
    def decode_bitstream(self, bitstream: bytes, device: torch.device) -> torch.Tensor:
        if len(bitstream) < HEADER.size:
            raise ValueError("bitstream is shorter than the DM-V3-PRIME header")

        magic, version, batch, latent_dim, delta, checksum = HEADER.unpack(
            bitstream[: HEADER.size]
        )
        if magic != MAGIC:
            raise ValueError("invalid DM-V3-PRIME bitstream magic")
        if version != BITSTREAM_VERSION:
            raise ValueError(f"unsupported bitstream version: {version}")
        if latent_dim != self.latent_dim:
            raise ValueError("bitstream latent dimension does not match decoder")

        raw = zlib.decompress(bitstream[HEADER.size :])
        expected_bytes = batch * latent_dim * np.dtype("<i2").itemsize
        if len(raw) != expected_bytes:
            raise ValueError("bitstream payload length does not match its header")
        if (zlib.crc32(raw) & 0xFFFFFFFF) != checksum:
            raise ValueError("bitstream payload checksum mismatch")

        q_np = np.frombuffer(raw, dtype="<i2").reshape(batch, latent_dim).copy()
        q = torch.from_numpy(q_np).to(device=device, dtype=torch.float32)
        return self.decode_latent(q * float(delta))


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed_call(device: torch.device, fn: Callable[[], T]) -> tuple[T, float]:
    synchronize(device)
    started = time.perf_counter()
    result = fn()
    synchronize(device)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return result, elapsed_ms


def parameter_memory_bytes(model: nn.Module) -> int:
    total = 0
    for tensor in list(model.parameters()) + list(model.buffers()):
        total += tensor.numel() * tensor.element_size()
    return total


def make_field(
    grid_x: torch.Tensor,
    grid_y: torch.Tensor,
    grid_z: torch.Tensor,
    phase: float,
) -> torch.Tensor:
    del grid_y  # reserved for future anisotropic field terms
    field = 0.5 + 0.5 * torch.sin(3.0 * grid_x + phase) * torch.cos(
        4.0 * grid_z - 0.5 * phase
    )
    return field.unsqueeze(0).unsqueeze(0)


@torch.no_grad()
def evaluate_engine(
    engine: DMV3PrimeEngine,
    x: torch.Tensor,
    delta: float,
    rate_weight: float,
    timing_repeats: int,
) -> Evaluation:
    engine.eval()
    device = x.device

    for _ in range(2):
        engine(x, delta)
    synchronize(device)

    latencies: list[float] = []
    output: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
    for _ in range(timing_repeats):
        output, elapsed_ms = timed_call(device, lambda: engine(x, delta))
        latencies.append(elapsed_ms)

    assert output is not None
    x_hat, z_q, rate_bits = output
    distortion = F.mse_loss(x_hat, x).item()
    differentiable_rate = rate_bits.mean().item()
    objective = distortion + rate_weight * differentiable_rate

    bitstream = engine.encode_bitstream(x, delta)
    decoded = engine.decode_bitstream(bitstream, device)
    finite = bool(torch.isfinite(decoded).all().item() and torch.isfinite(z_q).all().item())
    decoded_distortion = F.mse_loss(decoded, x).item()

    psnr_db = float("inf") if distortion == 0.0 else 10.0 * math.log10(1.0 / distortion)
    bits_per_voxel = len(bitstream) * 8.0 / x.numel()
    memory_bytes = parameter_memory_bytes(engine) + x.numel() * x.element_size()

    metrics = DMV3Metrics(
        distortion=decoded_distortion,
        latency_ms=median(latencies),
        objective=objective,
        memory_bytes=memory_bytes,
        risk=0.0 if finite else 1.0,
        stable=finite,
        safe=finite,
    )
    return Evaluation(
        metrics=metrics,
        psnr_db=psnr_db,
        bits_per_voxel=bits_per_voxel,
        bitstream_bytes=len(bitstream),
    )


def execute_moagi_engine(args: argparse.Namespace) -> int:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    resolution = 32
    coords = torch.linspace(-1.0, 1.0, resolution, device=device)
    grid_x, grid_y, grid_z = torch.meshgrid(coords, coords, coords, indexing="ij")

    engine = DMV3PrimeEngine(
        latent_dim=args.latent_dim,
        recursion_depth=args.recursion_depth,
    ).to(device)
    optimizer = torch.optim.AdamW(engine.parameters(), lr=args.lr, weight_decay=1e-5)

    validation_x = make_field(grid_x, grid_y, grid_z, phase=0.37)
    incumbent_state = copy.deepcopy(engine.state_dict())
    incumbent = evaluate_engine(
        engine,
        validation_x,
        args.delta,
        args.rate_weight,
        args.timing_repeats,
    )

    print("BOOT: DM-V3-PRIME Engine operational")
    print(
        f"Lattice: {resolution}^3 | Device: {device} | "
        f"Latent: {args.latent_dim} | Recursion depth: {args.recursion_depth}"
    )
    print(
        "BASELINE: "
        f"J={incumbent.metrics.objective:.6f} | "
        f"D={incumbent.metrics.distortion:.6f} | "
        f"PSNR={incumbent.psnr_db:.2f}dB | "
        f"Rate={incumbent.bits_per_voxel:.4f} b/voxel | "
        f"Latency={incumbent.metrics.latency_ms:.3f}ms"
    )

    engine.train()
    for cycle in range(args.cycles):
        phase = cycle * 0.1
        x = make_field(grid_x, grid_y, grid_z, phase=phase)

        optimizer.zero_grad(set_to_none=True)
        x_hat, _, rate_bits = engine(x, args.delta)
        distortion = F.mse_loss(x_hat, x)
        rate = rate_bits.mean()
        objective = distortion + args.rate_weight * rate
        objective.backward()
        torch.nn.utils.clip_grad_norm_(engine.parameters(), max_norm=5.0)
        optimizer.step()

        if cycle % args.log_every == 0 or cycle == args.cycles - 1:
            distortion_value = distortion.detach().item()
            psnr = (
                float("inf")
                if distortion_value == 0.0
                else 10.0 * math.log10(1.0 / distortion_value)
            )
            print(
                f"[Cycle {cycle:04d}] J={objective.detach().item():.6f} | "
                f"D={distortion_value:.6f} | PSNR={psnr:.2f}dB | "
                f"RateProxy={rate.detach().item():.3f} bits/latent"
            )

    candidate_state = copy.deepcopy(engine.state_dict())
    candidate = evaluate_engine(
        engine,
        validation_x,
        args.delta,
        args.rate_weight,
        args.timing_repeats,
    )

    policy = VerificationPolicy(
        max_distortion=args.max_distortion,
        max_memory_bytes=args.max_memory_mb * 1024 * 1024,
        max_risk=0.1,
        min_speedup=args.min_speedup,
        min_objective_improvement=args.min_objective_improvement,
        target_speedup=1000.0,
    )
    gate: PiHLambdaGate[dict[str, torch.Tensor]] = PiHLambdaGate(policy)
    selected_state, decision = gate.deploy(
        incumbent_state,
        candidate_state,
        incumbent.metrics,
        candidate.metrics,
    )
    engine.load_state_dict(selected_state)

    print("\nCANDIDATE:")
    print(
        f"J={candidate.metrics.objective:.6f} | "
        f"D={candidate.metrics.distortion:.6f} | "
        f"PSNR={candidate.psnr_db:.2f}dB | "
        f"Rate={candidate.bits_per_voxel:.4f} b/voxel | "
        f"Bitstream={candidate.bitstream_bytes} bytes | "
        f"Latency={candidate.metrics.latency_ms:.3f}ms"
    )
    print(
        f"VERIFY Pi_(H,Lambda): {'COMMIT' if decision.accepted else 'ROLLBACK'} | "
        f"Measured speedup={decision.speedup:.3f}x | "
        f"1000x target={'MET' if decision.speed_target_met else 'NOT MET'}"
    )
    if decision.reasons:
        for reason in decision.reasons:
            print(f"  - {reason}")

    print(f"Total parameter count: {sum(p.numel() for p in engine.parameters())}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the DM-V3-PRIME 3D auto-codec baseline")
    parser.add_argument("--cycles", type=int, default=200)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--recursion-depth", type=int, default=2)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--rate-weight", type=float, default=1e-3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--timing-repeats", type=int, default=7)
    parser.add_argument("--max-distortion", type=float, default=0.05)
    parser.add_argument("--max-memory-mb", type=int, default=4096)
    parser.add_argument("--min-speedup", type=float, default=1.0)
    parser.add_argument("--min-objective-improvement", type=float, default=0.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.cycles < 1:
        raise SystemExit("--cycles must be at least one")
    if args.delta <= 0.0:
        raise SystemExit("--delta must be positive")
    if args.timing_repeats < 1:
        raise SystemExit("--timing-repeats must be at least one")
    return execute_moagi_engine(args)


if __name__ == "__main__":
    raise SystemExit(main())
