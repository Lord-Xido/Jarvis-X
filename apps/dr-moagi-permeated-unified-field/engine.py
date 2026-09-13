#!/usr/bin/env python3
"""Permeated unified-field Dr Moagi volumetric interpreter.

A dense differentiable research backend for bounded active tiles.  It combines:
- one registered 3D field containing all channel groups;
- multimodal projection into that common field;
- local 3D convolution and global FFT-domain spatial mixing;
- damped fixed-point relaxation with explicit equilibrium telemetry;
- differentiable topological warping driven by reconstruction stress;
- kinetic gradient-flow integration with optional torchdiffeq.

"Instantaneous global mixing" means a global spatial receptive field is available
within one numerical permeation iteration through the FFT path.  It is not a
claim of instantaneous physical propagation.
"""
from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # Optional dependency; deterministic Euler fallback is always available.
    from torchdiffeq import odeint as _odeint
except ImportError:  # pragma: no cover - environment dependent
    _odeint = None


class UnifiedVoxelField(nn.Module):
    """Persistent registered 3D field with differentiable trilinear warping."""

    def __init__(
        self,
        res: int,
        total_channels: int,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        if res <= 0 or total_channels <= 0:
            raise ValueError("res and total_channels must be positive")
        self.res = int(res)
        self.total_channels = int(total_channels)
        self.register_buffer(
            "grid",
            torch.randn(1, total_channels, res, res, res, device=device) * 0.02,
            persistent=True,
        )
        z, y, x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, res, device=device),
            torch.linspace(-1.0, 1.0, res, device=device),
            torch.linspace(-1.0, 1.0, res, device=device),
            indexing="ij",
        )
        self.register_buffer(
            "base_grid",
            torch.stack((x, y, z), dim=-1).unsqueeze(0),
            persistent=False,
        )

    def _validate_grid(self, value: torch.Tensor) -> None:
        expected_tail = (self.res, self.res, self.res)
        if value.ndim != 5:
            raise ValueError("grid must have shape [B,C,D,H,W]")
        if value.shape[1] != self.total_channels:
            raise ValueError(f"expected {self.total_channels} channels")
        if tuple(value.shape[-3:]) != expected_tail:
            raise ValueError(f"expected spatial resolution {self.res}^3")

    def set_grid(self, value: torch.Tensor) -> None:
        self._validate_grid(value)
        self.grid = value.detach().clone()

    def warp(
        self,
        flow_field: torch.Tensor,
        *,
        grid: torch.Tensor | None = None,
        commit: bool = False,
    ) -> torch.Tensor:
        source = self.grid if grid is None else grid
        self._validate_grid(source)
        expected = (source.shape[0], 3, self.res, self.res, self.res)
        if tuple(flow_field.shape) != expected:
            raise ValueError("flow_field must have shape [B,3,D,H,W] matching the field")
        base = self.base_grid.expand(source.shape[0], -1, -1, -1, -1)
        displacement = flow_field.permute(0, 2, 3, 4, 1)
        sampling_grid = torch.clamp(base + displacement, -1.0, 1.0)
        warped = F.grid_sample(
            source,
            sampling_grid,
            mode="bilinear",
            padding_mode="reflection",
            align_corners=True,
        )
        if commit:
            self.set_grid(warped)
        return warped


class MultimodalProjector(nn.Module):
    """Project image/video/audio/features into one fixed-width 3D channel space."""

    def __init__(
        self,
        img_channels: int = 3,
        video_channels: int = 3,
        audio_dim: int = 128,
        feature_dim: int = 128,
        target_res: int = 16,
        total_channels: int = 112,
        seed_res: int = 4,
    ) -> None:
        super().__init__()
        if target_res <= 0 or total_channels <= 0 or seed_res <= 0:
            raise ValueError("target_res, total_channels and seed_res must be positive")
        self.target_res = int(target_res)
        self.total_channels = int(total_channels)
        self.seed_res = int(min(seed_res, target_res))

        self.image_encoder = nn.Sequential(
            nn.Conv2d(img_channels, total_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(total_channels),
            nn.SiLU(inplace=True),
        )
        self.depth_basis = nn.Parameter(
            torch.zeros(1, total_channels, target_res, 1, 1)
        )

        self.video_encoder = nn.Sequential(
            nn.Conv3d(video_channels, total_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(total_channels),
            nn.SiLU(inplace=True),
        )

        seed_volume = total_channels * self.seed_res**3
        self.audio_fc = nn.Linear(audio_dim, seed_volume)
        self.feature_fc = nn.Linear(feature_dim, seed_volume)

    def project_image(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4:
            raise ValueError("image must have shape [B,C,H,W]")
        plane = F.interpolate(
            image,
            size=(self.target_res, self.target_res),
            mode="bilinear",
            align_corners=False,
        )
        plane = self.image_encoder(plane).unsqueeze(2)
        depth_scale = 1.0 + 0.1 * torch.tanh(self.depth_basis)
        return plane * depth_scale

    def project_video(self, video: torch.Tensor) -> torch.Tensor:
        if video.ndim != 5:
            raise ValueError("video must have shape [B,C,T,H,W]")
        volume = F.interpolate(
            video,
            size=(self.target_res, self.target_res, self.target_res),
            mode="trilinear",
            align_corners=False,
        )
        return self.video_encoder(volume)

    def project_audio(self, audio: torch.Tensor) -> torch.Tensor:
        return self._project_vector(audio, self.audio_fc, "audio")

    def project_features(self, features: torch.Tensor) -> torch.Tensor:
        """Feature vectors may represent text/code/sensor embeddings."""
        return self._project_vector(features, self.feature_fc, "features")

    def _project_vector(
        self,
        value: torch.Tensor,
        layer: nn.Linear,
        name: str,
    ) -> torch.Tensor:
        if value.ndim != 2:
            raise ValueError(f"{name} must have shape [B,F]")
        batch = value.shape[0]
        seed = layer(value).view(
            batch,
            self.total_channels,
            self.seed_res,
            self.seed_res,
            self.seed_res,
        )
        if self.seed_res == self.target_res:
            return seed
        return F.interpolate(
            seed,
            size=(self.target_res, self.target_res, self.target_res),
            mode="trilinear",
            align_corners=False,
        )

    @staticmethod
    def fuse(
        fields: Sequence[torch.Tensor],
        weights: Sequence[float] | None = None,
    ) -> torch.Tensor:
        if not fields:
            raise ValueError("at least one field is required")
        shape = fields[0].shape
        if any(field.shape != shape for field in fields):
            raise ValueError("all fields must share the same shape")
        if weights is None:
            normalized = [1.0 / len(fields)] * len(fields)
        else:
            if len(weights) != len(fields):
                raise ValueError("weights must match fields")
            total = float(sum(weights))
            if not math.isfinite(total) or total <= 0.0:
                raise ValueError("weights must have a positive finite sum")
            normalized = [float(weight) / total for weight in weights]
        out = torch.zeros_like(fields[0])
        for weight, field in zip(normalized, fields):
            out = out + weight * field
        return out


class PermeationOperator(nn.Module):
    """Shared local/global/channel transform used at every equilibrium iteration."""

    def __init__(
        self,
        channels: int,
        res: int,
        spectral_rank: int = 16,
        local_gain: float = 0.25,
        global_gain: float = 0.15,
        channel_gain: float = 0.25,
    ) -> None:
        super().__init__()
        if channels <= 0 or res <= 0:
            raise ValueError("channels and res must be positive")
        self.channels = int(channels)
        self.res = int(res)
        self.rank = max(1, min(int(spectral_rank), channels))
        self.local_gain = float(local_gain)
        self.global_gain = float(global_gain)
        self.channel_gain = float(channel_gain)

        self.local_depthwise = nn.Conv3d(
            channels, channels, kernel_size=3, padding=1, groups=channels
        )
        self.local_pointwise = nn.Conv3d(channels, channels, kernel_size=1)

        self.to_spectral_rank = nn.Conv3d(channels, self.rank, kernel_size=1, bias=False)
        self.from_spectral_rank = nn.Conv3d(self.rank, channels, kernel_size=1, bias=False)
        freq_shape = (1, self.rank, res, res, res // 2 + 1)
        self.spectral_delta_real = nn.Parameter(torch.zeros(freq_shape))
        self.spectral_delta_imag = nn.Parameter(torch.zeros(freq_shape))

        self.channel_mix = nn.Conv3d(channels, channels, kernel_size=1)
        self.core_modulation = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.SiLU(),
            nn.Linear(channels * 2, channels * 2),
        )
        self.activation = nn.GELU()

    def _global_mix(self, value: torch.Tensor) -> torch.Tensor:
        ranked = self.to_spectral_rank(value)
        spectrum = torch.fft.rfftn(ranked, dim=(-3, -2, -1), norm="ortho")
        transfer = 1.0 + 0.1 * torch.complex(
            torch.tanh(self.spectral_delta_real),
            torch.tanh(self.spectral_delta_imag),
        )
        mixed = spectrum * transfer
        global_rank = torch.fft.irfftn(
            mixed,
            s=(self.res, self.res, self.res),
            dim=(-3, -2, -1),
            norm="ortho",
        )
        return self.from_spectral_rank(global_rank)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 5 or value.shape[1] != self.channels:
            raise ValueError("value must have shape [B,C,D,H,W]")
        if tuple(value.shape[-3:]) != (self.res, self.res, self.res):
            raise ValueError("value resolution does not match permeator")

        local = self.local_pointwise(self.local_depthwise(value))
        global_mixed = self._global_mix(value)

        core = value.mean(dim=(-3, -2, -1))
        scale, shift = self.core_modulation(core).chunk(2, dim=1)
        scale = 1.0 + 0.1 * torch.tanh(scale).view(value.shape[0], -1, 1, 1, 1)
        shift = 0.1 * torch.tanh(shift).view(value.shape[0], -1, 1, 1, 1)
        saturated = value * scale + shift
        channels = self.channel_mix(saturated)

        candidate = (
            value
            + self.local_gain * local
            + self.global_gain * global_mixed
            + self.channel_gain * channels
        )
        return self.activation(candidate)


class TopologicalWarpNetwork(nn.Module):
    def __init__(self, in_ch: int, max_displacement: float = 0.15) -> None:
        super().__init__()
        self.max_displacement = float(max_displacement)
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(inplace=True),
            nn.Conv3d(32, 3, kernel_size=3, padding=1),
        )

    def forward(self, stress: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(stress)) * self.max_displacement


class PermeatedVolumetricInterpreter(nn.Module):
    """Unified 3D equilibrium engine with warp and kinetic feedback."""

    def __init__(
        self,
        res: int = 16,
        outer_ch: int = 16,
        mid_ch: int = 32,
        inner_ch: int = 64,
        device: str | torch.device = "cpu",
        spectral_rank: int = 16,
        fixed_point_damping: float = 0.5,
        max_update: float = 1.0,
        warp_weight: float = 0.25,
        warp_reg_weight: float = 0.001,
        kinetic_weight: float = 0.01,
        equilibrium_weight: float = 0.05,
        field_reconstruction_weight: float = 0.10,
        kinetic_dt: float = 0.01,
    ) -> None:
        super().__init__()
        if min(outer_ch, mid_ch, inner_ch) <= 0:
            raise ValueError("channel groups must be positive")
        if not 0.0 < fixed_point_damping <= 1.0:
            raise ValueError("fixed_point_damping must be in (0,1]")
        if max_update <= 0.0:
            raise ValueError("max_update must be positive")
        self.res = int(res)
        self.outer_ch = int(outer_ch)
        self.mid_ch = int(mid_ch)
        self.inner_ch = int(inner_ch)
        self.total_channels = outer_ch + mid_ch + inner_ch
        self.fixed_point_damping = float(fixed_point_damping)
        self.max_update = float(max_update)
        self.warp_weight = float(warp_weight)
        self.warp_reg_weight = float(warp_reg_weight)
        self.kinetic_weight = float(kinetic_weight)
        self.equilibrium_weight = float(equilibrium_weight)
        self.field_reconstruction_weight = float(field_reconstruction_weight)
        self.kinetic_dt = float(kinetic_dt)

        self.field = UnifiedVoxelField(res, self.total_channels, device=device)
        self.permeator = PermeationOperator(
            self.total_channels, res, spectral_rank=spectral_rank
        )
        self.warp_net = TopologicalWarpNetwork(self.total_channels)
        self.extract_outer = nn.Conv3d(self.total_channels, outer_ch, kernel_size=1)
        self.extract_mid = nn.Conv3d(self.total_channels, mid_ch, kernel_size=1)
        self.extract_inner = nn.Conv3d(self.total_channels, inner_ch, kernel_size=1)
        self.last_metrics: dict[str, float | int | bool] = {}

    def _validate_input(self, value: torch.Tensor) -> None:
        self.field._validate_grid(value)

    def relax(
        self,
        initial: torch.Tensor,
        *,
        max_iterations: int,
        convergence_tol: float,
    ) -> tuple[torch.Tensor, int, float, bool]:
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if convergence_tol < 0.0:
            raise ValueError("convergence_tol must be non-negative")
        value = initial
        residual_value = math.inf
        converged = False
        executed = 0
        for _ in range(max_iterations):
            proposal = self.permeator(value)
            raw_delta = proposal - value
            bounded_delta = self.max_update * torch.tanh(raw_delta / self.max_update)
            residual = bounded_delta.pow(2).mean().sqrt()
            residual_value = float(residual.detach())
            value = value + self.fixed_point_damping * bounded_delta
            executed += 1
            if convergence_tol > 0.0 and residual_value <= convergence_tol:
                converged = True
                break
        return value, executed, residual_value, converged

    def forward(
        self,
        input_field: torch.Tensor | None = None,
        iterations: int = 5,
        *,
        convergence_tol: float = 0.0,
        injection_gain: float = 1.0,
        commit_state: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        current = self.field.grid
        if input_field is not None:
            self._validate_input(input_field)
            initial = current.expand(input_field.shape[0], -1, -1, -1, -1) + injection_gain * input_field
        else:
            initial = current

        unified, executed, residual, converged = self.relax(
            initial,
            max_iterations=iterations,
            convergence_tol=convergence_tol,
        )
        outer = self.extract_outer(unified)
        mid = self.extract_mid(unified)
        inner = self.extract_inner(unified)
        core = unified.mean(dim=(-3, -2, -1), keepdim=True)

        if commit_state:
            self.field.set_grid(unified)
        self.last_metrics.update(
            {
                "iterations": executed,
                "fixed_point_residual": residual,
                "converged": converged,
            }
        )
        return outer, mid, inner, core, unified

    def _integrate_kinetics(
        self,
        state: torch.Tensor,
        force: torch.Tensor,
    ) -> torch.Tensor:
        detached_force = force.detach()
        if _odeint is None:
            return state - self.kinetic_dt * detached_force

        def kinetics(_time: torch.Tensor, _state: torch.Tensor) -> torch.Tensor:
            return -detached_force

        t_span = torch.tensor(
            [0.0, self.kinetic_dt],
            device=state.device,
            dtype=state.dtype,
        )
        solution = _odeint(kinetics, state, t_span, method="euler")
        return solution[-1]

    def optimize_step(
        self,
        target_outer: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        *,
        target_field: torch.Tensor | None = None,
        iterations: int = 5,
        convergence_tol: float = 0.0,
        grad_clip: float = 1.0,
    ) -> float:
        optimizer.zero_grad(set_to_none=True)
        pred_outer, _, _, _, unified = self.forward(
            iterations=iterations,
            convergence_tol=convergence_tol,
            commit_state=False,
        )
        if pred_outer.shape != target_outer.shape:
            raise ValueError("target_outer must exactly match the outer prediction shape")
        if target_field is not None:
            self._validate_input(target_field)
            if target_field.shape != unified.shape:
                raise ValueError("target_field must exactly match the unified field shape")

        reconstruction_loss = F.mse_loss(pred_outer, target_outer)

        stress = torch.autograd.grad(
            reconstruction_loss,
            unified,
            create_graph=False,
            retain_graph=True,
            only_inputs=True,
        )[0]
        displacement = self.warp_net(stress.detach())
        warped_field = self.field.warp(displacement, grid=unified, commit=False)
        warped_outer = self.extract_outer(warped_field)
        warp_reconstruction_loss = F.mse_loss(warped_outer, target_outer)
        warp_regularization = displacement.pow(2).mean()

        kinetic_field = self._integrate_kinetics(unified, stress)
        kinetic_outer = self.extract_outer(kinetic_field)
        kinetic_loss = F.mse_loss(kinetic_outer, target_outer)

        equilibrium_proposal = self.permeator(unified)
        equilibrium_loss = F.mse_loss(equilibrium_proposal, unified)

        if target_field is None:
            field_loss = torch.zeros((), device=unified.device, dtype=unified.dtype)
        else:
            field_loss = F.mse_loss(unified, target_field)

        total_loss = (
            reconstruction_loss
            + self.warp_weight * warp_reconstruction_loss
            + self.warp_reg_weight * warp_regularization
            + self.kinetic_weight * kinetic_loss
            + self.equilibrium_weight * equilibrium_loss
            + self.field_reconstruction_weight * field_loss
        )
        total_loss.backward()
        if grad_clip > 0.0:
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=grad_clip)
        optimizer.step()

        with torch.no_grad():
            committed = 0.75 * unified.detach() + 0.25 * warped_field.detach()
            self.field.set_grid(committed)

        self.last_metrics.update(
            {
                "total_loss": float(total_loss.detach()),
                "reconstruction_loss": float(reconstruction_loss.detach()),
                "warp_reconstruction_loss": float(warp_reconstruction_loss.detach()),
                "warp_regularization": float(warp_regularization.detach()),
                "kinetic_loss": float(kinetic_loss.detach()),
                "equilibrium_loss": float(equilibrium_loss.detach()),
                "field_loss": float(field_loss.detach()),
            }
        )
        return float(total_loss.detach())


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running permeated unified-field interpreter on: {device}")

    model = PermeatedVolumetricInterpreter(res=16, device=device).to(device)
    projector = MultimodalProjector(
        target_res=16,
        total_channels=model.total_channels,
    ).to(device)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(projector.parameters()),
        lr=5.0e-4,
    )

    image = torch.randn(1, 3, 64, 64, device=device)
    audio = torch.randn(1, 128, device=device)
    image_field = projector.project_image(image)
    audio_field = projector.project_audio(audio)
    combined = projector.fuse([image_field, audio_field])
    model.field.set_grid(combined)

    target_outer = combined[:, : model.outer_ch].contiguous()
    for step in range(8):
        loss = model.optimize_step(
            target_outer,
            optimizer,
            target_field=combined,
            iterations=5,
        )
        print(
            f"step {step + 1}/8 | loss={loss:.6f} | "
            f"eq_residual={model.last_metrics['fixed_point_residual']:.6f}"
        )

    with torch.no_grad():
        reconstruction, _, _, core, _ = model.forward(
            iterations=20,
            convergence_tol=1.0e-4,
        )
        mse = F.mse_loss(reconstruction, target_outer).item()
    print(f"final outer reconstruction MSE: {mse:.6f}")
    print(f"global core shape: {list(core.shape)}")
    print("permeation complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
