"""Minimal end-to-end Dr Moagi 3D auto-execution example."""

from jarvisx.dr_moagi_autoexec import DrMoagiAutoExecutionEngine


engine = DrMoagiAutoExecutionEngine()
engine.load(
    {
        (32, 32, 32): 1.0,
        (33, 32, 32): 0.75,
        (31, 32, 32): 0.75,
        (32, 33, 32): 0.50,
        (32, 31, 32): 0.50,
    }
)

for report in engine.run(4):
    print(
        f"cycle={report.cycle} "
        f"committed={report.committed} "
        f"active={report.active_cells_after} "
        f"latent={report.latent_cells} "
        f"mse={report.reconstruction_mse:.6f} "
        f"promoted={report.policy_promoted}"
    )

print(engine.status())
