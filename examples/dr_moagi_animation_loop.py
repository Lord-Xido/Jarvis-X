"""Run the bounded Dr Moagi 3D animation auto-execution loop."""

from __future__ import annotations

import json

from jarvisx.dr_moagi_animation_loop import (
    canonical_animation_loop_program,
    execute_auto_loop,
)
from jarvisx.dr_moagi_pdf_bytecode import make_seed_volume


def main() -> None:
    program = canonical_animation_loop_program(cycles=4, refinement_passes=6)
    result = execute_auto_loop(program, make_seed_volume(16))
    print(json.dumps(result.report(), indent=2))


if __name__ == "__main__":
    main()
