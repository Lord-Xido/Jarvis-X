"""Minimal Dr. Moagi UEA objective and equilibrium example."""

from jarvisx.unified_autoencoding import (
    DrMoagiUEA,
    MoagiCoefficients,
    Signal3D,
    SignalBounds,
)

engine = DrMoagiUEA(
    coefficients=MoagiCoefficients(
        beta=1.0e-3,
        gamma=0.25,
        lambda_m=0.02,
        lambda_f=0.01,
        lambda_n=0.00,
    )
)

signal = Signal3D(frequency=440.0, amplitude=0.8, phase=0.25)
loss = engine.loss((signal,))
fixed_point = engine.fixed_point_report(signal)

trace = engine.run_to_equilibrium(
    signal,
    time_step=1.0e-4,
    max_steps=10,
    forcing_mode="delta",
    bounds=SignalBounds(
        minimum_frequency=0.0,
        minimum_amplitude=0.0,
        maximum_amplitude=1.0,
    ),
)

print("loss:", loss.total)
print("fixed-point residual:", fixed_point.maximum_rms)
print("dynamic steps:", trace.steps)
print("final state:", trace.states[-1])
