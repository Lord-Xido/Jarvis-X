from __future__ import annotations

import numpy as np

from .ann import TinyAutoencoder
from .compiler import Assembler
from .vm import VANNVirtualMachine


DEMO_SOURCE = """
LOAD_INPUT
NORMALIZE
VOXELIZE
PREFETCH3D
ENCODE3D
PREDICT
COMPARE
UPDATE_OMEGA
PROJECT_LAMBDA
VERIFY
COMMIT
DECODE3D
PROJECT_LAMBDA
STAGE
VERIFY
COMMIT
RENDER
SAMPLE_METRICS
OPTIMIZE_POLICY
JOURNAL
HALT
""".strip()


def build_demo_vm(*, output_sink=print) -> VANNVirtualMachine:
    model = TinyAutoencoder(input_dim=12, latent_dim=4, seed=7)
    program = Assembler().assemble(DEMO_SOURCE)
    vm = VANNVirtualMachine(model, output_sink=output_sink)
    vm.load_program(program.instructions)
    return vm


def demo_input() -> np.ndarray:
    return np.asarray(
        [[0.1, 0.2, 0.9, 0.8, 0.15, 0.05, 0.7, 0.65, 0.4, 0.3, 0.95, 0.55]],
        dtype=np.float32,
    )


def run_demo() -> dict[str, object]:
    vm = build_demo_vm()
    vm.set_input(demo_input())
    return vm.run().__dict__
