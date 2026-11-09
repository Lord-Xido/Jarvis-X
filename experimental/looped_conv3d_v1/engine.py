"""JARVIS-X Ω³: bounded recurrent Conv3D autoencoder, CPU research reference.

Trainable 5 x stride-2 3D convolution encoder, 512-float bottleneck,
5 x stride-2 transposed convolution decoder, and replayable loopback.
An original-referenced guard is *evaluation only*, not a deployable decoder rule.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import nn

LATENT_DIM = 512
ARCHIVE_MAGIC = b"JX3DL512"
ARCHIVE_HEAD = struct.Struct("<8sII32s")
ARCHIVE_SIZE = ARCHIVE_HEAD.size + LATENT_DIM * 4 + 32  # 2128 bytes
CHANNELS = (4, 8, 12, 16, 16)
MAX_CYCLES = 4


def reproducible(seed: int = 42) -> None:
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)


class LoopedConv3D(nn.Module):
    """64³ or 128³, 1 channel, Conv3D stride-2 x 5, linear 512D core."""

    def __init__(self, side: int = 64):
        super().__init__()
        if side not in (64, 128):
            raise ValueError("supported voxel sides are 64 and 128")
        self.side = side
        self.bottleneck_side = side // 32
        enc = []
        last = 1
        for width in CHANNELS:
            enc.extend([nn.Conv3d(last, width, 3, stride=2, padding=1), nn.SiLU()])
            last = width
        self.encoder = nn.Sequential(*enc)
        self.flat_features = CHANNELS[-1] * self.bottleneck_side ** 3
        self.to_latent = nn.Linear(self.flat_features, LATENT_DIM)
        self.from_latent = nn.Linear(LATENT_DIM, self.flat_features)
        dec = []
        last = CHANNELS[-1]
        for width in (16, 12, 8, 4, 1):
            dec.append(nn.ConvTranspose3d(last, width, 4, stride=2, padding=1))
            dec.append(nn.Sigmoid() if width == 1 else nn.SiLU())
            last = width
        self.decoder = nn.Sequential(*dec)

    def _check_x(self, x: torch.Tensor) -> None:
        if x.dtype != torch.float32 or x.ndim != 5 or x.shape[1:] != (
            1, self.side, self.side, self.side
        ) or not torch.isfinite(x).all().item():
            raise ValueError(f"input must be finite float32 [B,1,{self.side},{self.side},{self.side}]")
        if x.min().item() < 0 or x.max().item() > 1:
            raise ValueError("input must be normalized to [0,1]")

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        self._check_x(x)
        return self.to_latent(self.encoder(x).flatten(1))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if z.dtype != torch.float32 or z.ndim != 2 or z.shape[1] != LATENT_DIM:
            raise ValueError("latent must have shape [B,512] and dtype float32")
        if not torch.isfinite(z).all().item():
            raise ValueError("nonfinite latent")
        y = self.decoder(self.from_latent(z).reshape(
            z.shape[0], CHANNELS[-1], self.bottleneck_side,
            self.bottleneck_side, self.bottleneck_side
        ))
        if y.shape != (z.shape[0], 1, self.side, self.side, self.side):
            raise RuntimeError("decoder shape mismatch")
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


def training_objective(model: LoopedConv3D, original: torch.Tensor, cycles: int = 2) -> torch.Tensor:
    """No reference bypass: each subsequent decoder input comes from prior output."""
    if not 1 <= cycles <= MAX_CYCLES:
        raise ValueError("cycles out of bounded range")
    model._check_x(original)
    current = original
    losses = []
    for _ in range(cycles):
        current = model(current)
        losses.append((current - original).square().mean())
    return torch.stack(losses).mean()


@dataclass
class OptimiseResult:
    before: float
    after: float
    accepted: int
    rejected: int
    history: list[dict]


def optimise(
    model: LoopedConv3D, original: torch.Tensor, *,
    cycles: int = 2, updates: int = 2, lr: float = 0.1, attempts: int = 6,
) -> OptimiseResult:
    """Exact candidate rollback, with full original-referenced objective each step.

    Rejected candidates restore *all* model parameters. No optimizer state is used.
    """
    if not (0 <= updates <= 16) or not (1 <= attempts <= 12) or not math.isfinite(lr) or lr <= 0:
        raise ValueError("invalid optimization budget")
    model.train()
    params = list(model.parameters())
    with torch.no_grad():
        baseline = float(training_objective(model, original, cycles))
    best = baseline
    accepted = rejected = 0
    history: list[dict] = []
    for iteration in range(updates):
        model.zero_grad(set_to_none=True)
        candidate_base = training_objective(model, original, cycles)
        candidate_base.backward()
        gradients = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in params]
        if not all(torch.isfinite(g).all().item() for g in gradients):
            raise RuntimeError("nonfinite gradients")
        snapshot = [p.detach().clone() for p in params]
        step_accepted = False
        for backtrack in range(attempts):
            step_size = lr / (2 ** backtrack)
            with torch.no_grad():
                for p, base, grad in zip(params, snapshot, gradients):
                    p.copy_(base - step_size * grad)
                try:
                    measured = float(training_objective(model, original, cycles))
                except (ValueError, RuntimeError, OverflowError):
                    # An unstable candidate is rejected; the parameter snapshot
                    # is restored on final failure (or overwritten on retry).
                    measured = math.inf
            if math.isfinite(measured) and measured < best - 1e-12:
                best = measured
                accepted += 1
                step_accepted = True
                history.append({"update": iteration, "accepted": True, "loss": measured, "lr": step_size})
                break
            rejected += 1
        if not step_accepted:
            with torch.no_grad():
                for p, base in zip(params, snapshot):
                    p.copy_(base)
            history.append({"update": iteration, "accepted": False, "loss": best, "lr": 0.0})
            break
    model.eval()
    return OptimiseResult(baseline, best, accepted, rejected, history)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def source_volume(side: int, seed: int = 1) -> torch.Tensor:
    if side not in (64, 128):
        raise ValueError("unsupported side")
    coords = torch.linspace(-1, 1, side)
    x, y, z = torch.meshgrid(coords, coords, coords, indexing="ij")
    blob = torch.exp(-((x + 0.17) ** 2 + (y - 0.12) ** 2 + z ** 2) * 5)
    detail = torch.sin((4 + seed) * x) * torch.cos(4 * y) * torch.sin(3 * z)
    vol = (0.34 + 0.42 * blob + 0.08 * detail).clamp(0, 1)
    # Synthetic SOURCE is quantized to UINT8 before conversion into normalized FP32.
    vol = torch.round(vol * 255.0) / 255.0
    return vol.unsqueeze(0).unsqueeze(0).contiguous().float()


def original_bytes(x: torch.Tensor) -> bytes:
    return torch.round(x.detach().cpu().clamp(0, 1) * 255).to(torch.uint8).numpy().tobytes()


def score(original: torch.Tensor, reconstructed: torch.Tensor) -> dict:
    err = float((original.detach() - reconstructed.detach()).square().mean())
    return {"mse_norm": err, "psnr_db": None if err == 0 else -10 * math.log10(err)}


def run_from_latent(
    model: LoopedConv3D, initial_z: torch.Tensor, *,
    cycles: int = 2, reference: torch.Tensor | None = None,
    guard: bool = False,
) -> tuple[torch.Tensor, list[dict]]:
    """Cycle 0 decodes transmitted latent; later cycles re-encode decoded output.

    Reference-guided guard is explicitly disallowed without original voxel data.
    It MUST NOT be treated as standalone compressed-stream inference.
    """
    if not 1 <= cycles <= MAX_CYCLES:
        raise ValueError("unbounded cycle budget")
    if guard and reference is None:
        raise ValueError("guard requires ORIGINAL reference; not decoder-only inference")
    if reference is not None:
        model._check_x(reference)
    best = None
    best_loss = math.inf
    trace = []
    z = initial_z
    with torch.no_grad():
        for iteration in range(cycles):
            if iteration:
                z = model.encode(current)
            current = model.decode(z)
            measured = score(reference, current) if reference is not None else None
            error = measured["mse_norm"] if measured is not None else None
            improved = error is not None and error < best_loss - 1e-12
            status = "unverified" if reference is None else ("accepted" if improved else "not_improved")
            if improved:
                best, best_loss = current.detach().clone(), error
            if guard and not improved:
                status = "rollback"
            record = {
                "cycle": iteration,
                "opcode": "0x30" if iteration == 0 else "0x10>0x30",
                "latent_sha256": sha256(z.detach().cpu().numpy().astype("<f4").tobytes()),
                "output_sha256": sha256(original_bytes(current)),
                "mse_norm": error,
                "psnr_db": measured["psnr_db"] if measured is not None else None,
                "status": status,
            }
            trace.append(record)
            if guard and not improved:
                return best, trace
    return current, trace


def save_checkpoint(model: LoopedConv3D, path: Path) -> str:
    state = {key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}
    save_file(state, str(path), metadata={"format": "JX3D-looped-conv3d-v1", "side": str(model.side), "latent": "512"})
    return sha256(path.read_bytes())


def load_checkpoint(path: Path, side: int, *, expected_sha: str | None = None) -> LoopedConv3D:
    digest = sha256(path.read_bytes())
    if expected_sha is not None and digest != expected_sha:
        raise ValueError("checkpoint digest mismatch")
    with safe_open(str(path), framework="pt", device="cpu") as source:
        meta = source.metadata() or {}
        if meta.get("format") != "JX3D-looped-conv3d-v1" or meta.get("side") != str(side) or meta.get("latent") != "512":
            raise ValueError("checkpoint model configuration mismatch")
    model = LoopedConv3D(side)
    model.load_state_dict(load_file(str(path), device="cpu"), strict=True)
    model.eval()
    return model


def pack_latent(model: LoopedConv3D, x: torch.Tensor, checkpoint_sha: str) -> bytes:
    if len(checkpoint_sha) != 64:
        raise ValueError("expected full checkpoint SHA-256")
    with torch.no_grad():
        z = model.encode(x)
    if z.shape != (1, LATENT_DIM):
        raise ValueError("archive supports exactly one volume")
    payload = z.detach().cpu().contiguous().numpy().astype("<f4", copy=False).tobytes()
    data = ARCHIVE_HEAD.pack(ARCHIVE_MAGIC, model.side, LATENT_DIM, bytes.fromhex(checkpoint_sha)) + payload
    result = data + hashlib.sha256(data).digest()
    if len(result) != ARCHIVE_SIZE:
        raise RuntimeError("archive length mismatch")
    return result


def unpack_latent(data: bytes, model: LoopedConv3D, checkpoint_sha: str) -> torch.Tensor:
    if len(data) != ARCHIVE_SIZE:
        raise ValueError("truncated or extended archive")
    magic, side, length, model_digest = ARCHIVE_HEAD.unpack(data[:ARCHIVE_HEAD.size])
    if magic != ARCHIVE_MAGIC or side != model.side or length != LATENT_DIM:
        raise ValueError("invalid archive header or dimensions")
    if model_digest != bytes.fromhex(checkpoint_sha):
        raise ValueError("archive bound to different model weights")
    if hashlib.sha256(data[:-32]).digest() != data[-32:]:
        raise ValueError("archive checksum mismatch")
    array = np.frombuffer(data[ARCHIVE_HEAD.size:-32], dtype="<f4").copy()
    if not np.isfinite(array).all():
        raise ValueError("nonfinite latent payload")
    return torch.from_numpy(array.reshape(1, LATENT_DIM))


def demo(side: int = 64, updates: int = 2, cycles: int = 2, output: Path | None = None) -> dict:
    reproducible(42)
    model = LoopedConv3D(side)
    original = source_volume(side)
    with torch.no_grad():
        initial = score(original, model(original))["mse_norm"]
    result = optimise(model, original, cycles=cycles, updates=updates)
    with torch.no_grad():
        seed = model.encode(original)
        prediction, raw = run_from_latent(model, seed, cycles=cycles, reference=original)
        guarded, observed = run_from_latent(model, seed, cycles=cycles, reference=original, guard=True)
    evidence = {
        "schema": "JX3D-looped-Conv3D-v1", "side": side,
        "input_uint8_bytes": side ** 3, "latent_values": LATENT_DIM,
        "latent_float32_bytes": LATENT_DIM * 4, "archive_bytes": ARCHIVE_SIZE,
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "model_float32_parameter_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
        "initial_cycle0_mse_norm": initial,
        "training": vars(result),
        "raw_loopback": raw,
        "guarded_reference_evaluation": observed,
        "guarded_return_mse_norm": score(original, guarded)["mse_norm"],
        "note": "Synthetic in-sample training; reference guard unavailable during decoder-only inference; model weights omitted from latent archive ratio",
    }
    if output:
        output.mkdir(parents=True, exist_ok=True)
        checkpoint = output / f"jx3d-{side}.safetensors"
        digest = save_checkpoint(model, checkpoint)
        payload = pack_latent(model, original, digest)
        (output / f"sample-{side}.jxz512").write_bytes(payload)
        loaded = load_checkpoint(checkpoint, side, expected_sha=digest)
        re_z = unpack_latent(payload, loaded, digest)
        with torch.no_grad():
            replay, _ = run_from_latent(loaded, re_z, cycles=cycles)
        if not torch.equal(prediction, replay):
            raise RuntimeError("same-machine saved artifact replay failed")
        evidence["checkpoint_bytes"] = checkpoint.stat().st_size
        evidence["joint_archive_and_checkpoint_bytes"] = checkpoint.stat().st_size + len(payload)
        evidence["nominal_uint8_to_latent_archive_ratio_excluding_weights"] = side ** 3 / len(payload)
        evidence["nominal_uint8_to_archive_and_checkpoint_ratio_single_volume"] = side ** 3 / (checkpoint.stat().st_size + len(payload))
        evidence["checkpoint_sha256"] = digest
        evidence["archive_sha256"] = sha256(payload)
        evidence["replay_exact_same_machine"] = True
        (output / "trace.json").write_text(json.dumps(evidence, sort_keys=True, indent=2) + "\n")
    return evidence


def cli() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("demo", "decode"))
    p.add_argument("--side", type=int, choices=(64, 128), default=64)
    p.add_argument("--cycles", type=int, default=2)
    p.add_argument("--updates", type=int, default=2)
    p.add_argument("--out", type=Path)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--archive", type=Path)
    args = p.parse_args()
    if args.command == "demo":
        print(json.dumps(demo(args.side, args.updates, args.cycles, args.out), indent=2, sort_keys=True))
    else:
        if args.checkpoint is None or args.archive is None or args.out is None:
            p.error("decode requires --checkpoint, --archive, and --out")
        reproducible()
        digest = sha256(args.checkpoint.read_bytes())
        model = load_checkpoint(args.checkpoint, args.side, expected_sha=digest)
        z = unpack_latent(args.archive.read_bytes(), model, digest)
        y, _ = run_from_latent(model, z, cycles=args.cycles)  # No original-reference guard!
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_bytes(original_bytes(y))
        print(json.dumps({"output_uint8_bytes": args.side ** 3, "output_sha256": sha256(original_bytes(y)), "cycles": args.cycles}))


if __name__ == "__main__":
    cli()
