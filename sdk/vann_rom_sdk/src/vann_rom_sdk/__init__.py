"""VANN-ROM Ω³ SDK.

A reference implementation of a sparse 3D ROM bytecode virtual ANN processor.
"""

from .ann import TinyAutoencoder
from .compiler import Assembler
from .geometry import Address3D, morton3d_decode, morton3d_encode
from .isa import Instruction, NumericFormat, Opcode, Phase
from .rom import Sparse3DROM, VoxelPage
from .vm import VANNVirtualMachine, VMConfig, VMResult

__all__ = [
    "Address3D",
    "Assembler",
    "Instruction",
    "NumericFormat",
    "Opcode",
    "Phase",
    "Sparse3DROM",
    "TinyAutoencoder",
    "VANNVirtualMachine",
    "VMConfig",
    "VMResult",
    "VoxelPage",
    "morton3d_encode",
    "morton3d_decode",
]

__version__ = "0.2.0"
