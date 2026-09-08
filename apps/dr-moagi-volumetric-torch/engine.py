#!/usr/bin/env python3
"""Differentiable 3D Dr Moagi volumetric autoencoder backend.

Dense Torch reference backend for recursive volumetric folding, dynamic 3D
hyper-convolution, topological warping, multimodal projection and kinetic
integration. It complements the sparse 1000x1000 logical runtime rather than
materializing that logical lattice densely.
"""
from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # Optional: the engine has a deterministic Euler fallback.
    from torchdiffeq import odeint as _odeint
except ImportError:  # pragma: no cover - depends on optional environment
    _odeint = None


class AdaptiveVoxelGrid3D(nn.Module):
    """Persistent 3D shell with registered state and trilinear coordinate warping."""

    def __init__(self, res: int, channels: int, device: str | torch.device = "cpu") -> None:
        super().__init__()
        if res <= 0 or channels <= 0:
            raise ValueError("res and channels must be positive")
        self.res = int(res)
        self.channels = int(channels)
        self.register_buffer(
            "grid", torch.randn(1, channels, res, res, res, device=device), persistent=True
        )
        z, y, x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, res, device=device),
            torch.linspace(-1.0, 1.0, res, device=device),
            torch.linspace(-1.0, 1.0, res, device=device),
            indexing="ij",
        )
        self.register_buffer(
            "base_grid", torch.stack((x, y, z), dim=-1).unsqueeze(0), persistent=False
        )

    def set_grid(self, value: torch.Tensor) -> None:
        self._validate_grid(value)
        self.grid = value.detach().clone()

    def _validate_grid(self, value: torch.Tensor) -> None:
        if value.ndim != 5:
            raise ValueError("grid must have shape [B,C,D,H,W]")
        if value.shape[1] != self.channels:
            raise ValueError(f"expected {self.channels} channels")
        if tuple(value.shape[-3:]) != (self.res, self.res, self.res):
            raise ValueError(f"expected spatial resolution {self.res}^3")

    def warp(
        self,
        flow_field: torch.Tensor,
        *,
        grid: torch.Tensor | None = None,
        commit: bool = False,
    ) -> torch.Tensor:
        """Warp ``grid`` by ``flow_field`` in normalized (dx,dy,dz) coordinates."""
        source = self.grid if grid is None else grid
        self._validate_grid(source)
        if flow_field.shape != (source.shape[0], 3, self.res, self.res, self.res):
            raise ValueError("flow_field must have shape [B,3,D,H,W] matching the shell")
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


class Multimodal3DProjectionAdapter(nn.Module):
    """Project image, video and vector modalities into a common 3D voxel field."""

    def __init__(
        self,
        img_channels: int = 3,
        video_channels: int = 3,
        audio_dim: int = 128,
        feature_dim: int = 128,
        target_res: int = 16,
        target_ch: int = 16,
    ) -> None:
        super().__init__()
        self.target_res = target_res
        self.target_ch = target_ch
        self.img_proj = nn.Sequential(
            nn.Conv2d(img_channels, target_ch * target_res, kernel_size=3, padding=1),
            nn.BatchNorm2d(target_ch * target_res),
            nn.SiLU(inplace=True),
        )
        self.video_proj = nn.Sequential(
            nn.Conv3d(video_channels, target_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(target_ch),
            nn.SiLU(inplace=True),
        )
        volume_dim = target_ch * target_res**3
        self.audio_fc = nn.Linear(audio_dim, volume_dim)
        self.feature_fc = nn.Linear(feature_dim, volume_dim)

    def project_image_2d(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4:
            raise ValueError("image must have shape [B,C,H,W]")
        batch = image.shape[0]
        x = F.interpolate(
            image,
            size=(self.target_res, self.target_res),
            mode="bilinear",
            align_corners=False,
        )
        x = self.img_proj(x)
        return x.view(
            batch,
            self.target_ch,
            self.target_res,
            self.target_res,
            self.target_res,
        )

    def project_video_3d(self, video: torch.Tensor) -> torch.Tensor:
        """[B,C,T,H,W] -> common 3D field, treating time as volumetric depth."""
        if video.ndim != 5:
            raise ValueError("video must have shape [B,C,T,H,W]")
        x = F.interpolate(
            video,
            size=(self.target_res, self.target_res, self.target_res),
            mode="trilinear",
            align_corners=False,
        )
        return self.video_proj(x)

    def project_audio_1d(self, audio: torch.Tensor) -> torch.Tensor:
        return self._project_vector(audio, self.audio_fc, "audio")

    def project_feature_1d(self, features: torch.Tensor) -> torch.Tensor:
        """Project text/code/sensor embeddings represented as feature vectors."""
        return self._project_vector(features, self.feature_fc, "features")

    def _project_vector(self, value: torch.Tensor, layer: nn.Linear, name: str) -> torch.Tensor:
        if value.ndim != 2:
            raise ValueError(f"{name} must have shape [B,F]")
        batch = value.shape[0]
        x = layer(value)
        return x.view(
            batch,
            self.target_ch,
            self.target_res,
            self.target_res,
            self.target_res,
        )

    @staticmethod
    def fuse(volumes: Sequence[torch.Tensor], weights: Sequence[float] | None = None) -> torch.Tensor:
        if not volumes:
            raise ValueError("at least one projected volume is required")
        reference_shape = volumes[0].shape
        if any(volume.shape != reference_shape for volume in volumes):
            raise ValueError("all projected volumes must share the same shape")
        if weights is None:
            normalized = [1.0 / len(volumes)] * len(volumes)
        else:
            if len(weights) != len(volumes):
                raise ValueError("weights must match volumes")
            total = float(sum(weights))
            if not math.isfinite(total) or total <= 0.0:
                raise ValueError("weights must have a positive finite sum")
            normalized = [float(weight) / total for weight in weights]
        fused = torch.zeros_like(volumes[0])
        for weight, volume in zip(normalized, volumes):
            fused = fused + weight * volume
        return fused


class Omnidirectional3DResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_ch)
        self.shortcut: nn.Module
        if in_ch == out_ch:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=1), nn.BatchNorm3d(out_ch)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = F.silu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.silu(out + residual, inplace=True)


class DynamicHyperNetwork3D(nn.Module):
    """Generate and apply per-sample 3D kernels conditioned on global core Omega*."""

    def __init__(
        self,
        latent_dim: int,
        kernel_size: int = 3,
        in_ch: int = 32,
        out_ch: int = 32,
    ) -> None:
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        output_dim = in_ch * out_ch * kernel_size**3
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.SiLU(), nn.Linear(256, output_dim)
        )
        self.weight_scale = 1.0 / math.sqrt(in_ch * kernel_size**3)

    def generate_weights(self, core_omega: torch.Tensor) -> torch.Tensor:
        batch = core_omega.shape[0]
        weights = self.mlp(core_omega.flatten(start_dim=1)) * self.weight_scale
        return weights.view(
            batch,
            self.out_ch,
            self.in_ch,
            self.kernel_size,
            self.kernel_size,
            self.kernel_size,
        )

    def forward(self, x: torch.Tensor, core_omega: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5 or x.shape[1] != self.in_ch:
            raise ValueError("x must be [B,in_ch,D,H,W]")
        if x.shape[0] != core_omega.shape[0]:
            raise ValueError("x and core_omega batch sizes must match")
        batch, _, depth, height, width = x.shape
        weights = self.generate_weights(core_omega).reshape(
            batch * self.out_ch,
            self.in_ch,
            self.kernel_size,
            self.kernel_size,
            self.kernel_size,
        )
        grouped = x.reshape(1, batch * self.in_ch, depth, height, width)
        out = F.conv3d(
            grouped,
            weights,
            padding=self.kernel_size // 2,
            groups=batch,
        )
        return out.view(batch, self.out_ch, depth, height, width)


class TopologicalWarpNetwork3D(nn.Module):
    def __init__(self, in_ch: int = 16, max_displacement: float = 0.15) -> None:
        super().__init__()
        self.max_displacement = max_displacement
        self.net = nn.Sequential(
            Omnidirectional3DResBlock(in_ch, 32), nn.Conv3d(32, 3, kernel_size=3, padding=1)
        )

    def forward(self, stress_tensor: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(stress_tensor)) * self.max_displacement


class VolumetricMultimodalInterpreter(nn.Module):
    """Recursive 3D encoder/decoder with topological and kinetic feedback."""

    def __init__(
        self,
        base_res: int = 16,
        device: str | torch.device = "cpu",
        correction_gain: float = 0.5,
        hyper_gain: float = 0.1,
        warp_weight: float = 0.25,
        kinetic_weight: float = 0.01,
        warp_reg_weight: float = 0.001,
        kinetic_dt: float = 0.01,
    ) -> None:
        super().__init__()
        if base_res < 8 or base_res % 4:
            raise ValueError("base_res must be >= 8 and divisible by 4")
        self.base_res = base_res
        self.correction_gain = correction_gain
        self.hyper_gain = hyper_gain
        self.warp_weight = warp_weight
        self.kinetic_weight = kinetic_weight
        self.warp_reg_weight = warp_reg_weight
        self.kinetic_dt = kinetic_dt

        self.shells = nn.ModuleDict(
            {
                "outer": AdaptiveVoxelGrid3D(base_res, 16, device=device),
                "mid": AdaptiveVoxelGrid3D(base_res // 2, 32, device=device),
                "inner": AdaptiveVoxelGrid3D(base_res // 4, 64, device=device),
            }
        )
        self.core_omega = nn.Parameter(torch.randn(1, 128, 2, 2, 2, device=device) * 0.02)

        self.enc_outer = Omnidirectional3DResBlock(16, 32)
        self.enc_mid = Omnidirectional3DResBlock(32, 64)
        inner_dim = 64 * (base_res // 4) ** 3
        core_dim = 128 * 8
        self.enc_inner_to_core = nn.Linear(inner_dim, core_dim)
        self.dec_core_to_inner = nn.Linear(core_dim, inner_dim)
        self.dec_inner = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            Omnidirectional3DResBlock(32, 32),
        )
        self.dec_mid = nn.Sequential(
            nn.ConvTranspose3d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            Omnidirectional3DResBlock(16, 16),
        )
        self.warp_net = TopologicalWarpNetwork3D(in_ch=16)
        self.hyper_net = DynamicHyperNetwork3D(
            latent_dim=core_dim, kernel_size=3, in_ch=32, out_ch=32
        )
        self.last_metrics: dict[str, float | int | bool] = {}

    def _encode(self, outer: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outer_feat = self.enc_outer(outer)
        mid_feat = F.max_pool3d(outer_feat, kernel_size=2)
        inner_feat = F.max_pool3d(self.enc_mid(mid_feat), kernel_size=2)
        core = self.enc_inner_to_core(inner_feat.flatten(start_dim=1)).view(-1, 128, 2, 2, 2)
        return inner_feat, core

    def _decode(
        self, core: torch.Tensor, inner_shape: torch.Size
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = core.shape[0]
        inner = self.dec_core_to_inner(core.flatten(start_dim=1)).view(
            batch, inner_shape[1], inner_shape[2], inner_shape[3], inner_shape[4]
        )
        mid = self.dec_inner(inner)
        mid = mid + self.hyper_gain * self.hyper_net(mid, core)
        outer = self.dec_mid(mid)
        return outer, mid, inner

    def forward(
        self,
        input_outer: torch.Tensor | None = None,
        recursive_depth: int = 2,
        *,
        commit_state: bool = True,
        convergence_tol: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if recursive_depth < 0:
            raise ValueError("recursive_depth must be non-negative")
        outer = self.shells["outer"].grid if input_outer is None else input_outer
        self.shells["outer"]._validate_grid(outer)
        inner_feat, core_init = self._encode(outer)
        core = core_init + self.core_omega

        executed = 0
        for _ in range(recursive_depth):
            pred_outer, _, _ = self._decode(core, inner_feat.shape)
            residual = outer - pred_outer
            _, residual_core = self._encode(residual)
            next_core = core + self.correction_gain * residual_core
            executed += 1
            if convergence_tol > 0.0:
                delta = (next_core - core).pow(2).mean().sqrt().detach().item()
                core = next_core
                if delta <= convergence_tol:
                    break
            else:
                core = next_core

        final_outer, final_mid, final_inner = self._decode(core, inner_feat.shape)
        if commit_state:
            self.shells["inner"].set_grid(final_inner)
            self.shells["mid"].set_grid(final_mid)
            self.shells["outer"].set_grid(final_outer)
        self.last_metrics["recursive_steps"] = executed
        return final_outer, final_mid, final_inner, core

    def _integrate_kinetics(
        self,
        pred_outer: torch.Tensor,
        stress_tensor: torch.Tensor,
        *,
        use_ode_solver: bool,
    ) -> torch.Tensor:
        force = -stress_tensor.detach()
        if use_ode_solver and _odeint is not None:
            times = torch.tensor(
                [0.0, self.kinetic_dt], device=pred_outer.device, dtype=pred_outer.dtype
            )

            def kinetics(_time: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
                return force

            return _odeint(kinetics, pred_outer, times, method="euler")[-1]
        return pred_outer + self.kinetic_dt * force

    def optimize_step(
        self,
        target_outer: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        *,
        recursive_depth: int = 2,
        use_ode_solver: bool = True,
        grad_clip: float = 1.0,
    ) -> float:
        self.shells["outer"]._validate_grid(target_outer)
        optimizer.zero_grad(set_to_none=True)
        input_grid = self.shells["outer"].grid.detach().clone().requires_grad_(True)
        pred_outer, pred_mid, pred_inner, _core = self.forward(
            input_grid, recursive_depth=recursive_depth, commit_state=False
        )
        recon_loss = F.mse_loss(pred_outer, target_outer)
        stress_tensor = torch.autograd.grad(
            recon_loss,
            pred_outer,
            retain_graph=True,
            create_graph=False,
            only_inputs=True,
        )[0].detach()

        displacement = self.warp_net(stress_tensor)
        warped_pred = self.shells["outer"].warp(displacement, grid=pred_outer, commit=False)
        warp_recon_loss = F.mse_loss(warped_pred, target_outer)
        warp_reg = displacement.pow(2).mean()

        kinetic_state = self._integrate_kinetics(
            pred_outer, stress_tensor, use_ode_solver=use_ode_solver
        )
        kinetic_loss = F.mse_loss(kinetic_state, target_outer)
        total_loss = (
            recon_loss
            + self.warp_weight * warp_recon_loss
            + self.warp_reg_weight * warp_reg
            + self.kinetic_weight * kinetic_loss
        )
        total_loss.backward()
        if grad_clip > 0.0:
            nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
        optimizer.step()

        self.shells["inner"].set_grid(pred_inner)
        self.shells["mid"].set_grid(pred_mid)
        self.shells["outer"].set_grid(warped_pred)
        self.last_metrics.update(
            {
                "loss": float(total_loss.detach()),
                "reconstruction_loss": float(recon_loss.detach()),
                "warp_reconstruction_loss": float(warp_recon_loss.detach()),
                "warp_regularization": float(warp_reg.detach()),
                "kinetic_loss": float(kinetic_loss.detach()),
                "used_torchdiffeq": bool(use_ode_solver and _odeint is not None),
            }
        )
        return float(total_loss.detach())


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing Dr Moagi volumetric interpreter on: {device}")
    model = VolumetricMultimodalInterpreter(base_res=16, device=device).to(device)
    adapter = Multimodal3DProjectionAdapter(target_res=16, target_ch=16).to(device)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(adapter.parameters()), lr=1.0e-3)
    image = torch.randn(1, 3, 64, 64, device=device)
    voxel_input = adapter.project_image_2d(image)
    model.shells["outer"].set_grid(voxel_input)
    print(f"Projected field: {list(voxel_input.shape)}")
    for step in range(5):
        loss = model.optimize_step(voxel_input, optimizer)
        print(f"step={step + 1} loss={loss:.6f} metrics={model.last_metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
