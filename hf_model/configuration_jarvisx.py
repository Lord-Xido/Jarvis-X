from typing import Any

from transformers import PreTrainedConfig


class JarvisXConfig(PreTrainedConfig):
    """Configuration for the Jarvis-X auto-encoding state-transition model."""

    model_type = "jarvisx"

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 512,
        latent_dim: int = 128,
        output_dim: int = 512,
        transition_layers: int = 2,
        omega_decay: float = 0.95,
        omega_rate: float = 0.10,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-5,
        **kwargs: Any,
    ) -> None:
        if min(input_dim, hidden_dim, latent_dim, output_dim) <= 0:
            raise ValueError("All model dimensions must be positive.")
        if transition_layers < 1:
            raise ValueError("transition_layers must be at least 1.")
        if not 0.0 <= omega_decay <= 1.0:
            raise ValueError("omega_decay must be in [0, 1].")
        if not 0.0 <= omega_rate <= 1.0:
            raise ValueError("omega_rate must be in [0, 1].")

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.latent_dim = int(latent_dim)
        self.output_dim = int(output_dim)
        self.transition_layers = int(transition_layers)
        self.omega_decay = float(omega_decay)
        self.omega_rate = float(omega_rate)
        self.initializer_range = float(initializer_range)
        self.layer_norm_eps = float(layer_norm_eps)
        super().__init__(**kwargs)
