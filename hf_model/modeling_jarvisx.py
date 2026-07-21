from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F
from transformers import PreTrainedModel
from transformers.utils import ModelOutput

try:
    from .configuration_jarvisx import JarvisXConfig
except ImportError:  # Hugging Face dynamic module fallback.
    from configuration_jarvisx import JarvisXConfig


@dataclass
class JarvisXOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    last_hidden_state: Optional[torch.Tensor] = None
    latent_state: Optional[torch.Tensor] = None
    reconstruction: Optional[torch.Tensor] = None
    omega_state: Optional[torch.Tensor] = None
    lambda_gate: Optional[torch.Tensor] = None
    prediction_error: Optional[torch.Tensor] = None


class JarvisXModel(PreTrainedModel):
    """Bounded recurrent autoencoder implementing Jarvis-X state evolution.

    Per step:
        prediction = P(state)
        error      = prediction - encoded_observation
        omega'     = decay * omega + rate * correction(-error)
        proposal   = state + prediction - error + omega'
        state'     = Lambda(state, proposal)
    """

    config_class = JarvisXConfig
    base_model_prefix = "jarvisx"
    main_input_name = "input_values"

    def __init__(self, config: JarvisXConfig) -> None:
        super().__init__(config)

        self.encoder = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_eps),
        )

        transition = []
        for _ in range(config.transition_layers):
            transition.extend(
                [
                    nn.Linear(config.hidden_dim, config.hidden_dim),
                    nn.GELU(),
                    nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_eps),
                ]
            )
        self.predictor = nn.Sequential(*transition)

        self.omega_projection = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.Tanh(),
        )
        self.lambda_gate = nn.Linear(config.hidden_dim * 4, config.hidden_dim)
        self.state_norm = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_eps)
        self.latent_projection = nn.Sequential(
            nn.Linear(config.hidden_dim, config.latent_dim),
            nn.Tanh(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_eps),
            nn.Linear(config.hidden_dim, config.output_dim),
        )
        self.post_init()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(
        self,
        input_values: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
        omega_state: Optional[torch.Tensor] = None,
        target_values: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[JarvisXOutput, Tuple[torch.Tensor, ...]]:
        return_dict = return_dict if return_dict is not None else getattr(
            self.config, "return_dict", True
        )

        if input_values is None:
            raise ValueError("input_values is required.")

        squeeze_sequence = False
        if input_values.ndim == 2:
            input_values = input_values.unsqueeze(1)
            squeeze_sequence = True
        if input_values.ndim != 3:
            raise ValueError(
                "input_values must have shape [batch, input_dim] or "
                "[batch, sequence, input_dim]."
            )
        if input_values.shape[-1] != self.config.input_dim:
            raise ValueError(
                f"Expected input_dim={self.config.input_dim}, "
                f"received {input_values.shape[-1]}."
            )

        batch_size, sequence_length, _ = input_values.shape
        reference = input_values
        expected_state_shape = (batch_size, self.config.hidden_dim)

        if initial_state is None:
            state = reference.new_zeros(expected_state_shape)
        else:
            if tuple(initial_state.shape) != expected_state_shape:
                raise ValueError(f"initial_state must have shape {expected_state_shape}.")
            state = initial_state.to(device=reference.device, dtype=reference.dtype)

        if omega_state is None:
            omega = reference.new_zeros(expected_state_shape)
        else:
            if tuple(omega_state.shape) != expected_state_shape:
                raise ValueError(f"omega_state must have shape {expected_state_shape}.")
            omega = omega_state.to(device=reference.device, dtype=reference.dtype)

        hidden_steps = []
        latent_steps = []
        reconstruction_steps = []
        gate_steps = []
        error_steps = []

        for step in range(sequence_length):
            encoded = self.encoder(input_values[:, step, :])
            prediction = self.predictor(state)
            error = prediction - encoded

            correction = self.omega_projection(-error)
            omega = self.config.omega_decay * omega + self.config.omega_rate * correction

            proposal = state + prediction - error + omega
            gate_features = torch.cat((state, encoded, prediction, omega), dim=-1)
            gate = torch.sigmoid(self.lambda_gate(gate_features))
            state = self.state_norm((1.0 - gate) * state + gate * proposal)

            latent = self.latent_projection(state)
            reconstruction = self.decoder(latent)

            hidden_steps.append(state)
            latent_steps.append(latent)
            reconstruction_steps.append(reconstruction)
            gate_steps.append(gate)
            error_steps.append(error)

        hidden_sequence = torch.stack(hidden_steps, dim=1)
        latent_sequence = torch.stack(latent_steps, dim=1)
        reconstruction_sequence = torch.stack(reconstruction_steps, dim=1)
        gate_sequence = torch.stack(gate_steps, dim=1)
        error_sequence = torch.stack(error_steps, dim=1)

        loss = None
        if target_values is not None:
            if target_values.ndim == 2:
                target_values = target_values.unsqueeze(1)
            if tuple(target_values.shape) != tuple(reconstruction_sequence.shape):
                raise ValueError(
                    "target_values must match reconstruction shape "
                    f"{tuple(reconstruction_sequence.shape)}."
                )
            loss = F.mse_loss(reconstruction_sequence, target_values)
        elif self.config.output_dim == self.config.input_dim:
            loss = F.mse_loss(reconstruction_sequence, input_values)

        if squeeze_sequence:
            hidden_sequence = hidden_sequence[:, 0, :]
            latent_sequence = latent_sequence[:, 0, :]
            reconstruction_sequence = reconstruction_sequence[:, 0, :]
            gate_sequence = gate_sequence[:, 0, :]
            error_sequence = error_sequence[:, 0, :]

        if not return_dict:
            values = (
                hidden_sequence,
                latent_sequence,
                reconstruction_sequence,
                omega,
                gate_sequence,
                error_sequence,
            )
            return ((loss,) + values) if loss is not None else values

        return JarvisXOutput(
            loss=loss,
            last_hidden_state=hidden_sequence,
            latent_state=latent_sequence,
            reconstruction=reconstruction_sequence,
            omega_state=omega,
            lambda_gate=gate_sequence,
            prediction_error=error_sequence,
        )
