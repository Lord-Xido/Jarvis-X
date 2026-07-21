"""Jarvis-X deterministic virtual machine and sparse 3-D automaton."""

from .automaton import (
    ADDRESS_DEPTH,
    AXIS_SIZE,
    RADIX,
    VIRTUAL_CELL_EXPONENT,
    BoundedMechanicsOptimizer,
    CellState,
    Coordinate3D,
    DeterministicAutoencoder,
    Mechanics,
    OptimizationResult,
    Sparse3DAutomaton,
    StepMetrics,
    make_echo_injections,
)

__version__ = "0.2.0"

__all__ = [
    "ADDRESS_DEPTH",
    "AXIS_SIZE",
    "RADIX",
    "VIRTUAL_CELL_EXPONENT",
    "BoundedMechanicsOptimizer",
    "CellState",
    "Coordinate3D",
    "DeterministicAutoencoder",
    "Mechanics",
    "OptimizationResult",
    "Sparse3DAutomaton",
    "StepMetrics",
    "make_echo_injections",
]
