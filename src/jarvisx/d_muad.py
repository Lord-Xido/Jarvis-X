"""Executable contracts for the Dr Moagi Unified Auto-Encoding Dynamics.

D-MUAD v2.0-C is a compressive, recurrent, physics-regularized, dual-branch
geometric autoencoding architecture.  This module does not implement a neural
network framework.  It makes the architecture's shape rules, arithmetic
budgets, loss aggregation, optimizer recurrence, and semantic boundaries
executable and testable with the Python standard library.
"""

from dataclasses import dataclass
import math
from typing import Dict, Tuple


Shape5D = Tuple[int, int, int, int, int]
Shape2D = Tuple[int, int]


def padded_extent(size: int, multiple: int = 8) -> int:
    """Return the smallest positive multiple greater than or equal to size."""
    if not isinstance(size, int):
        raise TypeError("Extent must be an integer")
    if not isinstance(multiple, int):
        raise TypeError("Padding multiple must be an integer")
    if size <= 0:
        raise ValueError("Extent must be positive")
    if multiple <= 0:
        raise ValueError("Padding multiple must be positive")
    return ((size + multiple - 1) // multiple) * multiple


@dataclass(frozen=True)
class DMUADConfig:
    """Canonical architectural constants for D-MUAD v2.0-C."""

    input_channels: int = 4
    embedding_channels: int = 64
    encoder_channels: Tuple[int, int, int] = (256, 512, 1024)
    latent_dim: int = 1024
    decoder_channels: Tuple[int, int, int] = (512, 128, 5)
    recurrent_hidden_dim: int = 1024
    positional_encoding_dim: int = 63
    nerf_hidden_dim: int = 256
    nerf_samples_per_ray: int = 64
    padding_multiple: int = 8

    def __post_init__(self) -> None:
        positive = (
            self.input_channels,
            self.embedding_channels,
            self.latent_dim,
            self.recurrent_hidden_dim,
            self.positional_encoding_dim,
            self.nerf_hidden_dim,
            self.nerf_samples_per_ray,
            self.padding_multiple,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("All D-MUAD dimensions must be positive")
        if any(value <= 0 for value in self.encoder_channels):
            raise ValueError("Encoder channels must be positive")
        if any(value <= 0 for value in self.decoder_channels):
            raise ValueError("Decoder channels must be positive")


@dataclass(frozen=True)
class DMUADContract:
    """Derived tensor contract for one D-MUAD input batch."""

    raw: Shape5D
    padded: Shape5D
    embedded: Shape5D
    h1: Shape5D
    h2: Shape5D
    h3: Shape5D
    pooled: Shape2D
    latent: Shape2D
    volumetric_base: Shape5D
    up1: Shape5D
    up2: Shape5D
    volumetric_output: Shape5D
    padding: Tuple[int, int, int]
    raw_elements_per_sample: int
    latent_elements_per_sample: int
    dense_bottleneck_weights: int

    @property
    def compressive(self) -> bool:
        return self.raw_elements_per_sample > self.latent_elements_per_sample

    @property
    def exact_global_inverse_possible(self) -> bool:
        """The canonical encoder is not globally invertible.

        Strided convolutions, ReLU, global pooling, and a finite latent bottleneck
        are many-to-one even for small inputs.  This property remains explicit so
        callers cannot mistake reconstruction for a bijective codec.
        """
        return False


def derive_contract(raw: Shape5D, config: DMUADConfig = DMUADConfig()) -> DMUADContract:
    """Derive all canonical tensor shapes from a raw B,C,T,H,W shape."""
    if len(raw) != 5:
        raise ValueError("D-MUAD input shape must be B,C,T,H,W")
    if any(not isinstance(value, int) for value in raw):
        raise TypeError("All shape components must be integers")
    if any(value <= 0 for value in raw):
        raise ValueError("All shape components must be positive")

    batch, channels, time, height, width = raw
    if channels != config.input_channels:
        raise ValueError(
            "D-MUAD requires {0} input channels; received {1}".format(
                config.input_channels, channels
            )
        )

    time_p = padded_extent(time, config.padding_multiple)
    height_p = padded_extent(height, config.padding_multiple)
    width_p = padded_extent(width, config.padding_multiple)
    padded = (batch, channels, time_p, height_p, width_p)
    embedded = (batch, config.embedding_channels, time_p, height_p, width_p)

    c1, c2, c3 = config.encoder_channels
    h1 = (batch, c1, time_p // 2, height_p // 2, width_p // 2)
    h2 = (batch, c2, time_p // 4, height_p // 4, width_p // 4)
    h3 = (batch, c3, time_p // 4, height_p // 4, width_p // 4)
    pooled = (batch, c3)
    latent = (batch, config.latent_dim)

    volumetric_base = h3
    d1, d2, d_out = config.decoder_channels
    up1 = (batch, d1, time_p // 2, height_p // 2, width_p // 2)
    up2 = (batch, d2, time_p, height_p, width_p)
    volumetric_output = (batch, d_out, time_p, height_p, width_p)

    base_values_per_sample = c3 * (time_p // 4) * (height_p // 4) * (width_p // 4)
    dense_bottleneck_weights = config.latent_dim * base_values_per_sample
    raw_elements_per_sample = channels * time_p * height_p * width_p

    return DMUADContract(
        raw=raw,
        padded=padded,
        embedded=embedded,
        h1=h1,
        h2=h2,
        h3=h3,
        pooled=pooled,
        latent=latent,
        volumetric_base=volumetric_base,
        up1=up1,
        up2=up2,
        volumetric_output=volumetric_output,
        padding=(time_p - time, height_p - height, width_p - width),
        raw_elements_per_sample=raw_elements_per_sample,
        latent_elements_per_sample=config.latent_dim,
        dense_bottleneck_weights=dense_bottleneck_weights,
    )


def conv3d_macs(
    output: Shape5D,
    input_channels: int,
    kernel: Tuple[int, int, int],
) -> int:
    """Multiply-accumulate count for a dense 3D convolution."""
    batch, output_channels, time, height, width = output
    kt, kh, kw = kernel
    values = (
        batch,
        output_channels,
        time,
        height,
        width,
        input_channels,
        kt,
        kh,
        kw,
    )
    if any(value <= 0 for value in values):
        raise ValueError("Convolution dimensions must be positive")
    return (
        batch
        * output_channels
        * time
        * height
        * width
        * input_channels
        * kt
        * kh
        * kw
    )


def separable_embedding_macs(contract: DMUADContract) -> int:
    """MACs for a 3x3 spatial stage followed by a length-3 temporal stage."""
    batch, embedded_channels, time, height, width = contract.embedded
    input_channels = contract.padded[1]
    spatial = batch * time * height * width * embedded_channels * input_channels * 9
    temporal = batch * time * height * width * embedded_channels * 3
    return spatial + temporal


def encoder_macs(contract: DMUADContract) -> int:
    """MAC count for the three canonical encoder convolutions."""
    return (
        conv3d_macs(contract.h1, contract.embedded[1], (3, 3, 3))
        + conv3d_macs(contract.h2, contract.h1[1], (3, 3, 3))
        + conv3d_macs(contract.h3, contract.h2[1], (3, 3, 3))
    )


@dataclass(frozen=True)
class LossWeights:
    sdf: float = 1.0
    appearance: float = 0.5
    physics: float = 0.1
    eikonal: float = 0.1
    rendering: float = 1.0

    def __post_init__(self) -> None:
        if any(value < 0.0 or not math.isfinite(value) for value in self.as_tuple()):
            raise ValueError("Loss weights must be finite and non-negative")

    def as_tuple(self) -> Tuple[float, float, float, float, float]:
        return (
            self.sdf,
            self.appearance,
            self.physics,
            self.eikonal,
            self.rendering,
        )


def aggregate_loss(
    sdf: float,
    appearance: float,
    physics: float,
    eikonal: float,
    rendering: float,
    weights: LossWeights = LossWeights(),
) -> float:
    """Aggregate the corrected D-MUAD loss functional in fp64 host arithmetic."""
    components = (sdf, appearance, physics, eikonal, rendering)
    if any(value < 0.0 or not math.isfinite(value) for value in components):
        raise ValueError("Loss components must be finite and non-negative")
    return sum(weight * value for weight, value in zip(weights.as_tuple(), components))


@dataclass(frozen=True)
class AdamScalarState:
    """One scalar parameter and its Adam moments."""

    parameter: float
    first_moment: float = 0.0
    second_moment: float = 0.0
    step: int = 0


def adam_scalar_step(
    state: AdamScalarState,
    gradient: float,
    learning_rate: float = 1e-4,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
) -> AdamScalarState:
    """Execute one exact scalar Adam recurrence."""
    scalars = (state.parameter, gradient, learning_rate, beta1, beta2, epsilon)
    if any(not math.isfinite(value) for value in scalars):
        raise ValueError("Adam inputs must be finite")
    if learning_rate <= 0.0 or epsilon <= 0.0:
        raise ValueError("Learning rate and epsilon must be positive")
    if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
        raise ValueError("Adam beta values must lie in [0, 1)")

    step = state.step + 1
    first = beta1 * state.first_moment + (1.0 - beta1) * gradient
    second = beta2 * state.second_moment + (1.0 - beta2) * gradient * gradient
    first_hat = first / (1.0 - beta1**step)
    second_hat = second / (1.0 - beta2**step)
    parameter = state.parameter - learning_rate * first_hat / (math.sqrt(second_hat) + epsilon)
    return AdamScalarState(parameter, first, second, step)


def constitutional_boundaries() -> Dict[str, str]:
    """Return the claims that are valid for D-MUAD v2.0-C."""
    return {
        "representation": "compressive and distributionally reconstructive",
        "inverse": "approximate decoder, not a global exact inverse",
        "differentiability": "training core is differentiable almost everywhere",
        "egress": "thresholding, uint8 conversion, and marching cubes are discrete",
        "determinism": "conditional on fixed state, parameters, kernels, and reduction order",
        "optimization": "closed recurrence, not a closed-form global optimum",
        "arithmetic_budget": "input-shape and implementation dependent",
    }
