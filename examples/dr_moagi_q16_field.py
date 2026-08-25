from jarvisx.dr_moagi_q16_field import (
    DrMoagiQ16Config,
    DrMoagiQ16FieldRuntime,
    Q16Interval,
    q_from_float,
    q_to_float,
    temporal_compression_law,
)

runtime = DrMoagiQ16FieldRuntime(
    DrMoagiQ16Config(
        side=16,
        lambda_inverse_raw=q_from_float(1.0),
        gamma_gain_raw=q_from_float(0.05),
        eta_amplitude_raw=q_from_float(0.0),
        seed=7,
    )
)
runtime.load_float({(8, 8, 8): 0.25})

report = runtime.step_field(
    psi_raw={(8, 8, 8): q_from_float(0.10)},
    phi_kernel={(0, 0, 0): q_from_float(0.50)},
    adaptive_gradient_raw={(8, 8, 8): q_from_float(0.05)},
    constraints={(8, 8, 8): Q16Interval(0, q_from_float(0.75))},
)

print("tick:", report.tick)
print("Xi:", {coord: q_to_float(raw) for coord, raw in report.output.items()})
print("ledger:", report.ledger_hash)
print("clock-law:", temporal_compression_law(0.45))
