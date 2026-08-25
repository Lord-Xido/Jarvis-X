"""Gradient-driven inward self-optimization for the Jarvis-X Hugging Face model.

This module operationalizes the user-supplied 3D parameter-fold concept without
destructively decoding an entire model from a single 3-vector. Model parameters
are partitioned into bounded chunks, projected to 3D control tokens, evolved by a
radial inward-fold field, and pulled back to parameter space with a vector-
Jacobian product. The resulting geometry update is blended with ordinary
gradient descent and norm-bounded before commit.

The mechanism is experimental self-distillation, not a proof of globally optimal
parameters or learning without an objective.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import torch
from torch import nn
from torch.nn import functional as F

try:
    from .configuration_jarvisx import JarvisXConfig
    from .modeling_jarvisx import JarvisXModel, JarvisXOutput
except ImportError:  # Hugging Face dynamic module fallback.
    from configuration_jarvisx import JarvisXConfig
    from modeling_jarvisx import JarvisXModel, JarvisXOutput


@dataclass(frozen=True)
class InwardOptimizerBounds:
    """Numerical guardrails for the inward optimizer."""

    cube_extent: float = 1000.0
    fold_radius: float = 500.0
    dt: float = 0.02
    finite_difference_eps: float = 1.0e-3
    geometry_mix: float = 0.10
    max_update_to_param_ratio: float = 1.0e-3
    chunk_size: int = 256
    projector_hidden_dim: int = 128

    def __post_init__(self) -> None:
        if self.cube_extent <= 0.0:
            raise ValueError("cube_extent must be positive")
        if not 0.0 < self.fold_radius <= self.cube_extent:
            raise ValueError("fold_radius must lie in (0, cube_extent]")
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.finite_difference_eps <= 0.0:
            raise ValueError("finite_difference_eps must be positive")
        if not 0.0 <= self.geometry_mix <= 1.0:
            raise ValueError("geometry_mix must be in [0, 1]")
        if self.max_update_to_param_ratio <= 0.0:
            raise ValueError("max_update_to_param_ratio must be positive")
        if self.chunk_size < 3:
            raise ValueError("chunk_size must be at least 3")
        if self.projector_hidden_dim < 3:
            raise ValueError("projector_hidden_dim must be at least 3")


def inward_fold(
    points: torch.Tensor,
    *,
    cube_extent: float = 1000.0,
    fold_radius: float = 500.0,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """Invert points through a sphere centered in the logical 3D cube.

    Using ``R**2 / ||p-c||**2`` makes the sphere ``||p-c|| = R`` the fixed
    attractor shell. The small epsilon prevents the center singularity.
    """

    if points.shape[-1] != 3:
        raise ValueError("points must have trailing dimension 3")
    center = points.new_full((3,), cube_extent * 0.5)
    delta = points - center
    norm_sq = delta.square().sum(dim=-1, keepdim=True).clamp_min(eps)
    folded = center + (fold_radius * fold_radius) * delta / norm_sq
    return folded.clamp_(0.0, cube_extent)


class InwardSelfOptimizer(JarvisXModel):
    """Jarvis-X model with a bounded 3D kinetic parameter-control loop.

    The base Jarvis-X parameters are the optimization target. The 3D projection
    machinery and hyperparameter anchors are explicitly excluded from the target
    flattening so the parameter-vector dimensionality remains invariant.
    """

    _CONTROL_PREFIXES = ("param_projector.", "hyper_anchors.")

    def __init__(
        self,
        config: JarvisXConfig,
        *,
        bounds: Optional[InwardOptimizerBounds] = None,
        alpha: float = 1.0e-3,
        beta: float = 1.0e-2,
        eta: float = 1.0e-3,
        learning_rate: float = 1.0e-4,
    ) -> None:
        super().__init__(config)
        self.bounds = bounds or InwardOptimizerBounds()

        if min(alpha, beta, eta, learning_rate) <= 0.0:
            raise ValueError("alpha, beta, eta and learning_rate must be positive")

        hidden = self.bounds.projector_hidden_dim
        chunk = self.bounds.chunk_size
        self.param_projector = nn.Sequential(
            nn.Linear(chunk, hidden),
            nn.GELU(),
            nn.Linear(hidden, 3),
        )
        self.hyper_anchors = nn.ParameterDict(
            {
                "alpha": nn.Parameter(torch.tensor([float(alpha)])),
                "beta": nn.Parameter(torch.tensor([float(beta)])),
                "eta": nn.Parameter(torch.tensor([float(eta)])),
                "lr": nn.Parameter(torch.tensor([float(learning_rate)])),
            }
        )
        self._last_self_loss: Optional[float] = None

    def _target_named_parameters(self) -> list[tuple[str, nn.Parameter]]:
        targets: list[tuple[str, nn.Parameter]] = []
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            if any(name.startswith(prefix) for prefix in self._CONTROL_PREFIXES):
                continue
            targets.append((name, parameter))
        if not targets:
            raise RuntimeError("no trainable base-model parameters were found")
        return targets

    @staticmethod
    def _flatten(
        parameters: Sequence[nn.Parameter],
        gradients: Sequence[Optional[torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        flat_parameters = torch.cat([parameter.detach().reshape(-1) for parameter in parameters])
        flat_gradients = torch.cat(
            [
                (
                    torch.zeros_like(parameter)
                    if gradient is None
                    else gradient.detach().to(device=parameter.device, dtype=parameter.dtype)
                ).reshape(-1)
                for parameter, gradient in zip(parameters, gradients)
            ]
        )
        return flat_parameters, flat_gradients

    def _chunk(self, vector: torch.Tensor) -> tuple[torch.Tensor, int]:
        chunk_size = self.bounds.chunk_size
        remainder = vector.numel() % chunk_size
        pad = 0 if remainder == 0 else chunk_size - remainder
        if pad:
            vector = F.pad(vector, (0, pad))
        return vector.view(-1, chunk_size), pad

    def param_to_voxel(self, flat_parameters: torch.Tensor) -> torch.Tensor:
        """Project parameter chunks to bounded 3D token coordinates."""

        chunks, _ = self._chunk(flat_parameters)
        rms = chunks.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1.0e-8)
        normalized = chunks / rms
        unit = torch.sigmoid(self.param_projector(normalized))
        return unit * self.bounds.cube_extent

    def kinetic_param_step(
        self,
        flat_parameters: torch.Tensor,
        flat_gradients: torch.Tensor,
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        """Blend gradient descent with a bounded inward-fold control displacement."""

        if flat_parameters.shape != flat_gradients.shape:
            raise ValueError("flat_parameters and flat_gradients must have identical shape")

        geometry_input = flat_parameters.detach().requires_grad_(True)
        voxels = self.param_to_voxel(geometry_input)

        descent_probe = geometry_input - self.bounds.finite_difference_eps * flat_gradients
        perturbed_voxels = self.param_to_voxel(descent_probe)
        velocity = (perturbed_voxels - voxels) / self.bounds.finite_difference_eps

        alpha = self.hyper_anchors["alpha"].detach().clamp(1.0e-8, 1.0)
        beta = self.hyper_anchors["beta"].detach().clamp(1.0e-8, 10.0)
        eta = self.hyper_anchors["eta"].detach().clamp(1.0e-8, 1.0)
        learning_rate = self.hyper_anchors["lr"].detach().clamp(1.0e-8, 1.0)

        folded = inward_fold(
            voxels,
            cube_extent=self.bounds.cube_extent,
            fold_radius=self.bounds.fold_radius,
        )
        force = alpha * (folded - voxels) + eta * velocity
        velocity_new = velocity + self.bounds.dt * (-beta * velocity + force)
        voxel_new = (voxels + self.bounds.dt * velocity_new).clamp(
            0.0, self.bounds.cube_extent
        )
        voxel_delta = voxel_new - voxels

        pullback = torch.autograd.grad(
            outputs=voxels,
            inputs=geometry_input,
            grad_outputs=voxel_delta.detach(),
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )[0].detach()

        grad_norm = flat_gradients.norm().clamp_min(1.0e-12)
        pullback_norm = pullback.norm().clamp_min(1.0e-12)
        scaled_pullback = pullback * (grad_norm / pullback_norm)

        gradient_update = -learning_rate * flat_gradients
        geometry_update = learning_rate * self.bounds.geometry_mix * scaled_pullback
        update = gradient_update + geometry_update

        param_norm = flat_parameters.norm().clamp_min(1.0e-12)
        max_update = self.bounds.max_update_to_param_ratio * param_norm
        update_norm = update.norm().clamp_min(1.0e-12)
        scale = torch.minimum(update.new_tensor(1.0), max_update / update_norm)
        bounded_update = update * scale
        new_parameters = flat_parameters + bounded_update

        metrics = {
            "grad_norm": float(grad_norm.item()),
            "voxel_velocity": float(velocity.norm(dim=-1).mean().item()),
            "voxel_flux": float(voxel_delta.norm(dim=-1).mean().item()),
            "update_norm": float(bounded_update.norm().item()),
            "update_scale": float(scale.item()),
        }
        return new_parameters, metrics

    @torch.no_grad()
    def _scatter(self, parameters: Sequence[nn.Parameter], flat_parameters: torch.Tensor) -> None:
        offset = 0
        for parameter in parameters:
            count = parameter.numel()
            replacement = flat_parameters[offset : offset + count].view_as(parameter)
            parameter.copy_(replacement)
            offset += count
        if offset != flat_parameters.numel():
            raise RuntimeError("flat parameter scatter did not consume the full vector")

    @torch.no_grad()
    def _adapt_hyperparameters(self, current_loss: float) -> None:
        """Conservatively adapt anchors from the observed cycle-to-cycle loss trend."""

        previous = self._last_self_loss
        if previous is None:
            self._last_self_loss = current_loss
            return

        improved = current_loss < previous
        strength_factor = 1.002 if improved else 0.995
        damping_factor = 0.999 if improved else 1.005
        lr_factor = 1.001 if improved else 0.995

        self.hyper_anchors["alpha"].mul_(strength_factor).clamp_(1.0e-6, 0.1)
        self.hyper_anchors["eta"].mul_(strength_factor).clamp_(1.0e-6, 0.1)
        self.hyper_anchors["beta"].mul_(damping_factor).clamp_(1.0e-5, 1.0)
        self.hyper_anchors["lr"].mul_(lr_factor).clamp_(1.0e-7, 1.0e-2)
        self._last_self_loss = current_loss

    def self_optimise_step(self, loss: torch.Tensor) -> Dict[str, float]:
        """Execute one bounded inward optimization cycle."""

        target_pairs = self._target_named_parameters()
        parameters = [parameter for _, parameter in target_pairs]
        gradients = torch.autograd.grad(
            loss,
            parameters,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )
        flat_parameters, flat_gradients = self._flatten(parameters, gradients)
        new_parameters, kinetic = self.kinetic_param_step(flat_parameters, flat_gradients)
        self._scatter(parameters, new_parameters)

        loss_value = float(loss.detach().item())
        self._adapt_hyperparameters(loss_value)
        kinetic.update(
            {
                "loss": loss_value,
                "alpha": float(self.hyper_anchors["alpha"].item()),
                "beta": float(self.hyper_anchors["beta"].item()),
                "eta": float(self.hyper_anchors["eta"].item()),
                "lr": float(self.hyper_anchors["lr"].item()),
                "param_flux": float(flat_gradients.norm().item() / 1.0e6),
            }
        )
        return kinetic

    def reconstruct(self, input_state: torch.Tensor) -> torch.Tensor:
        output = self.forward(input_state, return_dict=True)
        if not isinstance(output, JarvisXOutput) or output.reconstruction is None:
            raise RuntimeError("Jarvis-X forward pass did not return a reconstruction")
        return output.reconstruction

    def forward_self_optimising(
        self,
        input_state: torch.Tensor,
        *,
        num_cycles: int = 10,
        inward_mix: float = 0.5,
        mix_every: int = 2,
    ) -> tuple[torch.Tensor, tuple[Dict[str, float], ...]]:
        """Run iterative self-distillation and bounded 3D parameter refinement."""

        if num_cycles <= 0:
            raise ValueError("num_cycles must be positive")
        if not 0.0 <= inward_mix <= 1.0:
            raise ValueError("inward_mix must be in [0, 1]")
        if mix_every <= 0:
            raise ValueError("mix_every must be positive")

        state = input_state.detach()
        reports: list[Dict[str, float]] = []
        reconstruction = self.reconstruct(state)

        for cycle in range(num_cycles):
            loss = F.mse_loss(reconstruction, state)
            report = self.self_optimise_step(loss)
            report["cycle"] = float(cycle)
            reports.append(report)

            reconstruction = self.reconstruct(state)
            if cycle % mix_every == 0:
                state = (
                    (1.0 - inward_mix) * state + inward_mix * reconstruction.detach()
                ).detach()
            else:
                state = state.detach()

        return reconstruction, tuple(reports)


def build_token_state_optimizer(
    *,
    hidden_dim: int = 256,
    latent_dim: int = 64,
    device: Optional[torch.device] = None,
) -> InwardSelfOptimizer:
    """Construct the six-channel position/velocity demo model."""

    config = JarvisXConfig(
        input_dim=6,
        output_dim=6,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        transition_layers=2,
    )
    model = InwardSelfOptimizer(config)
    if device is not None:
        model = model.to(device)
    return model


if __name__ == "__main__":
    torch.manual_seed(7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_token_state_optimizer(device=device)

    batch = 1024
    position = torch.rand(batch, 3, device=device) * 1000.0
    velocity = (torch.rand(batch, 3, device=device) - 0.5) * 20.0
    state = torch.cat((position, velocity), dim=-1)

    print("Starting bounded inward self-optimization...")
    for epoch in range(5):
        refined, reports = model.forward_self_optimising(state, num_cycles=20)
        state = refined.detach()
        final_loss = F.mse_loss(model.reconstruct(state), state).item()
        print(
            f"epoch={epoch} loss={final_loss:.6f} "
            f"alpha={reports[-1]['alpha']:.6g} "
            f"beta={reports[-1]['beta']:.6g} "
            f"eta={reports[-1]['eta']:.6g}"
        )

    print("Inward loop completed as bounded self-distillation.")
