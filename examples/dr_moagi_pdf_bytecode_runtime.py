"""Build, verify, and execute a bounded PDF-carried DM3D bytecode package."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from jarvisx.dr_moagi_pdf_bytecode import (
    build_pdf_package,
    canonical_autoencoder_program,
    make_seed_volume,
    run_pdf_package,
)


def main() -> None:
    program = canonical_autoencoder_program(pool=2, refinement_passes=4)
    with tempfile.TemporaryDirectory(prefix="jarvisx-dm3d-") as directory:
        package = Path(directory) / "dr-moagi-bytecode.pdf"
        manifest = build_pdf_package(package, program)
        result = run_pdf_package(package, make_seed_volume(16))
        print(
            json.dumps(
                {
                    "manifest": manifest.__dict__,
                    "result": result.report(),
                },
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
