"""VANN-ROM Ω³ SDK.

Reference implementations of the sparse 3D ROM bytecode processor and the
Aether sparse 4D multimodal auto-encoding engine.
"""

from .aether import (
    AetherConfig,
    AetherEngine,
    AetherInput,
    AetherLoss,
    AetherLossWeights,
    AetherOutput,
    AetherPolicy,
    AetherResult,
    GraphTensor,
    Sparse4DField,
    morton4d_decode,
    morton4d_encode,
    synthetic_aether_input,
)
from .ann import TinyAutoencoder
from .compiler import Assembler
from .geometry import Address3D, morton3d_decode, morton3d_encode
from .isa import Instruction, NumericFormat, Opcode, Phase
from .rom import Sparse3DROM, VoxelPage
from .vm import VANNVirtualMachine, VMConfig, VMResult

__all__ = [
    "Address3D",
    "AetherConfig",
    "AetherEngine",
    "AetherInput",
    "AetherLoss",
    "AetherLossWeights",
    "AetherOutput",
    "AetherPolicy",
    "AetherResult",
    "Assembler",
    "GraphTensor",
    "Instruction",
    "NumericFormat",
    "Opcode",
    "Phase",
    "Sparse3DROM",
    "Sparse4DField",
    "TinyAutoencoder",
    "VANNVirtualMachine",
    "VMConfig",
    "VMResult",
    "VoxelPage",
    "morton3d_encode",
    "morton3d_decode",
    "morton4d_encode",
    "morton4d_decode",
    "synthetic_aether_input",
]

__version__ = "0.3.0"
