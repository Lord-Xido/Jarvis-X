"""Jarvis-X deterministic VM and sparse geometric automata."""

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
from .tetration_field import (
    BASE,
    BRICK_SIZE,
    CHANNELS,
    EDGE,
    BrickAutoencoderMoE,
    BrickState,
    FieldMechanics,
    FieldStepMetrics,
    SparseHashDirectory,
    TetrationAddress,
    TetrationFieldAutomaton as ReferenceTetrationFieldAutomaton,
    TetrationUniverse,
    make_brick_pulse,
)
from .operational_field import OperationalTetrationFieldAutomaton

TetrationFieldAutomaton = OperationalTetrationFieldAutomaton

__version__ = "0.3.0"

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
    "BASE",
    "BRICK_SIZE",
    "CHANNELS",
    "EDGE",
    "BrickAutoencoderMoE",
    "BrickState",
    "FieldMechanics",
    "FieldStepMetrics",
    "SparseHashDirectory",
    "TetrationAddress",
    "ReferenceTetrationFieldAutomaton",
    "OperationalTetrationFieldAutomaton",
    "TetrationFieldAutomaton",
    "TetrationUniverse",
    "make_brick_pulse",
]
