from __future__ import annotations

import numpy as np

from jarvisx.qsol_kinetic_runtime import KineticConfig, QSOLKineticRuntime


def main() -> None:
    volume = np.zeros((8, 8, 8), dtype=np.uint8)
    volume[2, 3, 4] = 255
    volume[2, 3, 5] = 240
    volume[3, 3, 4] = 250
    volume[3, 4, 4] = 245

    runtime = QSOLKineticRuntime(
        KineticConfig(
            latent_dim=64,
            max_cycles=160,
            latent_tolerance=1.0e-3,
        )
    )
    result = runtime.run(volume)

    print("QSOL KINETIC RUNTIME")
    print(f"converged={result.converged}")
    print(f"cycles={result.cycles}")
    print(f"bytes_transferred={result.bytes_transferred}")
    print(f"latent_norm={np.linalg.norm(result.latent):.6f}")
    print(f"final_mse={result.receipts[-1].authoritative_mse:.9f}")
    print(f"latent_rms={result.receipts[-1].latent_rms:.9f}")
    print(f"state_hash={result.receipts[-1].state_hash}")
    print(f"commits={sum(receipt.committed for receipt in result.receipts)}")
    print(f"rollbacks={sum(not receipt.committed for receipt in result.receipts)}")


if __name__ == "__main__":
    main()
