"""Data contracts for the bounded Moagi OmegaFold reference resolver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple


State = Tuple[float, ...]
Transition = Callable[[State], State]
Residual = Callable[[State], float]
ClosedForm = Callable[[State], State]


@dataclass(frozen=True)
class FoldConfig:
    """Deterministic execution limits for one OmegaFold resolution."""

    max_iterations: int = 256
    tolerance: float = 1.0e-9
    quantization_digits: int = 12

    def __post_init__(self) -> None:
        if self.max_iterations < 0:
            raise ValueError("max_iterations must be non-negative")
        if self.tolerance < 0.0:
            raise ValueError("tolerance must be non-negative")
        if self.quantization_digits < 0:
            raise ValueError("quantization_digits must be non-negative")


@dataclass(frozen=True)
class FoldProblem:
    """A finite state-transition problem with an independently measured residual."""

    name: str
    initial_state: State
    transition: Transition
    residual: Residual
    closed_form: Optional[ClosedForm] = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("problem name must be non-empty")
        if not self.initial_state:
            raise ValueError("initial_state must contain at least one value")


@dataclass(frozen=True)
class FoldCertificate:
    """Evidence describing how and why a bounded resolution terminated."""

    problem_name: str
    method: str
    iterations: int
    converged: bool
    residual: float
    terminal_reason: str
    state_digest: str
    config_digest: str


@dataclass(frozen=True)
class FoldResult:
    """Terminal state, residual trace and verification certificate."""

    state: State
    residual_trace: Tuple[float, ...]
    certificate: FoldCertificate
