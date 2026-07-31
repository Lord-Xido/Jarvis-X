"""Public API for the Dr. Moagi Unified Autoencoding reference."""

from .uea_dynamics import DrMoagiUEA, EvolutionTrace, ForcingMode, SignalBounds
from .uea_model import (
    AffineSignalOperation,
    FixedPointReport,
    GaussianPosterior,
    LinearGaussianAutoencoder,
    LossBreakdown,
    Matrix3,
    MoagiCoefficients,
    OperationSet,
    Signal3D,
    SignalMetric,
    Vector3,
    phase_difference,
    signal_residual,
    signal_squared_error,
    wrap_phase,
)

__all__ = [
    "AffineSignalOperation",
    "DrMoagiUEA",
    "EvolutionTrace",
    "FixedPointReport",
    "ForcingMode",
    "GaussianPosterior",
    "LinearGaussianAutoencoder",
    "LossBreakdown",
    "Matrix3",
    "MoagiCoefficients",
    "OperationSet",
    "Signal3D",
    "SignalBounds",
    "SignalMetric",
    "Vector3",
    "phase_difference",
    "signal_residual",
    "signal_squared_error",
    "wrap_phase",
]
