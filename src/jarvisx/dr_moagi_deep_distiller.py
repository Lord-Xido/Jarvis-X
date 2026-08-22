"""Dr Moagi Deep Distiller (DM-DD): IP-locked residual auto-iteration.

DM-DD is a bounded sparse reference implementation of the operational law::

    Z_t         = E_Theta(X_t)
    X_hat_t     = D_Theta(Z_t)
    E_t         = X_t - X_hat_t
    Omega_t+1   = rho * Omega_t + (1-rho) * E_t
    Theta'_t+1  = Theta_t - eta * grad_Theta ||E_t||^2
    X'_t+1      = X_hat_t + omega * Omega_t+1
    (X,Omega,Theta)_t+1 = Pi_Lambda(X',Omega',Theta')

The constitutional lock is transactional: state, residual memory, and parameters
commit together or none of them commit. Rejected proposals may be journaled for
audit, but never become authoritative runtime state.

The implementation is intentionally sparse and finite. ``logical_side`` describes
a logical lattice; only active coordinates are materialized. No dense N^3 volume
or unbudgeted parameter/code expansion is permitted.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Callable, Mapping

from .dr_moagi_autoexec import AutoExecPolicy, HashChainJournal, SparseParser3D
from .dr_moagi_field_runtime import Coordinate, SparseField


@dataclass(frozen=True)
class DeepDistillerTheta:
    """Minimal learnable codec parameters for the reference distiller.

    The encoder/decoder gains make the residual gradient explicit and testable.
    Production encoders may replace these scalar maps while preserving the same
    propose -> Pi_Lambda -> atomic-commit contract.
    """

    encoder_gain: float = 0.80
    decoder_gain: float = 0.80


@dataclass(frozen=True)
class DeepDistillerConfig:
    logical_side: int = 1000
    max_active_cells: int = 50_000
    max_latent_cells: int = 25_000
    value_min: float = -1.0
    value_max: float = 1.0
    rho: float = 0.90
    omega_gain: float = 0.25
    learning_rate: float = 0.05
    theta_min: float = 0.05
    theta_max: float = 2.0
    theta_max_delta: float = 0.10
    residual_tolerance: float = 1.0e-6
    latent_prune_epsilon: float = 0.0
    memory_prune_epsilon: float = 0.0
    max_iterations: int = 64
    policy: AutoExecPolicy = field(default_factory=AutoExecPolicy)

    def __post_init__(self) -> None:
        for name in ("logical_side", "max_active_cells", "max_latent_cells", "max_iterations"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_latent_cells > self.max_active_cells:
            raise ValueError("max_latent_cells cannot exceed max_active_cells")
        for name in (
            "value_min",
            "value_max",
            "rho",
            "omega_gain",
            "learning_rate",
            "theta_min",
            "theta_max",
            "theta_max_delta",
            "residual_tolerance",
            "latent_prune_epsilon",
            "memory_prune_epsilon",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if self.value_min >= self.value_max:
            raise ValueError("value_min must be smaller than value_max")
        if not 0.0 <= self.rho < 1.0:
            raise ValueError("rho must be in [0, 1)")
        if self.omega_gain < 0.0:
            raise ValueError("omega_gain must be non-negative")
        if self.learning_rate < 0.0:
            raise ValueError("learning_rate must be non-negative")
        if self.theta_min <= 0.0 or self.theta_min >= self.theta_max:
            raise ValueError("theta bounds must satisfy 0 < theta_min < theta_max")
        if self.theta_max_delta <= 0.0:
            raise ValueError("theta_max_delta must be positive")
        if self.residual_tolerance < 0.0:
            raise ValueError("residual_tolerance must be non-negative")
        if self.latent_prune_epsilon < 0.0 or self.memory_prune_epsilon < 0.0:
            raise ValueError("prune epsilons must be non-negative")


@dataclass(frozen=True)
class DeepDistillerCandidate:
    state: SparseField
    omega: SparseField
    theta: DeepDistillerTheta
    latent_cells: int


DeepDistillerGate = Callable[[DeepDistillerCandidate], bool]


@dataclass(frozen=True)
class DeepDistillerReport:
    iteration: int
    committed: bool
    converged: bool
    active_cells: int
    latent_cells: int
    residual_rms: float
    omega_rms: float
    loss: float
    grad_encoder: float
    grad_decoder: float
    encoder_gain: float
    decoder_gain: float
    gate_passed: bool
    rejection_reason: str | None
    state_hash: str
    theta_hash: str
    journal_hash: str = ""

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class DeepDistiller:
    """Transactional residual auto-iteration network.

    ``Pi_Lambda`` is represented by :meth:`pi_lambda`. The method validates the
    complete proposal (X, Omega, Theta) before :meth:`step` mutates any
    authoritative state.
    """

    LAW_ID = "DM-DD"
    PRODUCT_NAME = "Dr Moagi Deep Distiller"
    COMMIT_POLICY = "Pi_Lambda-atomic"

    def __init__(
        self,
        config: DeepDistillerConfig | None = None,
        *,
        theta: DeepDistillerTheta | None = None,
        gate: DeepDistillerGate | None = None,
        journal_path: str | Path | None = None,
    ) -> None:
        self.config = config or DeepDistillerConfig()
        self._theta = theta or DeepDistillerTheta()
        self._validate_theta(self._theta)
        self.gate = gate
        self.parser = SparseParser3D(
            side=self.config.logical_side,
            max_active_cells=self.config.max_active_cells,
            value_min=self.config.value_min,
            value_max=self.config.value_max,
            prune_epsilon=self.config.policy.prune_epsilon,
        )
        self.journal = HashChainJournal(journal_path)
        self._state: SparseField = {}
        self._omega: SparseField = {}
        self._iteration = 0
        self._loaded = False
        self.reports: list[DeepDistillerReport] = []

    @property
    def theta(self) -> DeepDistillerTheta:
        return self._theta

    def load(self, source: Mapping[Coordinate, float]) -> SparseField:
        """Admit initial state through parser + Pi_Lambda before authority."""
        parsed = self.parser.parse(source)
        initial = DeepDistillerCandidate(
            state=dict(parsed), omega={}, theta=self._theta, latent_cells=0
        )
        passed, reason = self.pi_lambda(initial)
        if not passed:
            raise ValueError(f"initial state rejected by Pi_Lambda: {reason}")
        self._state = dict(parsed)
        self._omega = {}
        self._iteration = 0
        self.reports.clear()
        self._loaded = True
        return self.snapshot()

    def snapshot(self) -> SparseField:
        self._require_loaded()
        return dict(self._state)

    def omega_snapshot(self) -> SparseField:
        self._require_loaded()
        return dict(self._omega)

    def encode(self, state: Mapping[Coordinate, float], theta: DeepDistillerTheta) -> SparseField:
        """Distill X -> Z with bounded deterministic sparse top-K support."""
        eps = self.config.latent_prune_epsilon
        scored: list[tuple[float, Coordinate, float]] = []
        for coordinate, raw in state.items():
            value = theta.encoder_gain * float(raw)
            if abs(value) > eps:
                scored.append((abs(value), coordinate, value))
        scored.sort(key=lambda item: (-item[0], item[1]))
        selected = scored[: self.config.max_latent_cells]
        return {coordinate: value for _, coordinate, value in selected}

    @staticmethod
    def decode(latent: Mapping[Coordinate, float], theta: DeepDistillerTheta) -> SparseField:
        return {
            coordinate: theta.decoder_gain * float(value)
            for coordinate, value in latent.items()
        }

    @staticmethod
    def residual(
        state: Mapping[Coordinate, float], reconstructed: Mapping[Coordinate, float]
    ) -> SparseField:
        result: SparseField = {}
        for coordinate in set(state) | set(reconstructed):
            value = float(state.get(coordinate, 0.0)) - float(reconstructed.get(coordinate, 0.0))
            if value != 0.0:
                result[coordinate] = value
        return result

    def omega_update(self, residual: Mapping[Coordinate, float]) -> SparseField:
        retain = self.config.rho
        inject = 1.0 - retain
        eps = self.config.memory_prune_epsilon
        result: SparseField = {}
        for coordinate in set(self._omega) | set(residual):
            value = retain * float(self._omega.get(coordinate, 0.0)) + inject * float(
                residual.get(coordinate, 0.0)
            )
            if abs(value) > eps:
                result[coordinate] = value
        return result

    def residual_gradient(
        self,
        state: Mapping[Coordinate, float],
        latent: Mapping[Coordinate, float],
        residual: Mapping[Coordinate, float],
        theta: DeepDistillerTheta,
    ) -> tuple[float, float]:
        """Gradient of mean squared residual on retained latent support.

        Top-K support selection is treated as fixed during one tick (a standard
        straight-through reference approximation for this discrete bottleneck).
        """
        count = max(1, len(set(state) | set(residual)))
        grad_encoder = 0.0
        grad_decoder = 0.0
        for coordinate, z_value in latent.items():
            x_value = float(state.get(coordinate, 0.0))
            error = float(residual.get(coordinate, 0.0))
            grad_encoder += -2.0 * theta.decoder_gain * x_value * error / count
            grad_decoder += -2.0 * float(z_value) * error / count
        return grad_encoder, grad_decoder

    def theta_candidate(
        self, theta: DeepDistillerTheta, grad_encoder: float, grad_decoder: float
    ) -> DeepDistillerTheta:
        eta = self.config.learning_rate
        cap = self.config.theta_max_delta

        def update(value: float, gradient: float) -> float:
            delta = max(-cap, min(cap, -eta * gradient))
            return max(self.config.theta_min, min(self.config.theta_max, value + delta))

        return DeepDistillerTheta(
            encoder_gain=update(theta.encoder_gain, grad_encoder),
            decoder_gain=update(theta.decoder_gain, grad_decoder),
        )

    def state_candidate(
        self,
        reconstructed: Mapping[Coordinate, float],
        omega: Mapping[Coordinate, float],
    ) -> SparseField:
        result: SparseField = {}
        eps = self.config.policy.prune_epsilon
        for coordinate in set(reconstructed) | set(omega):
            value = float(reconstructed.get(coordinate, 0.0)) + self.config.omega_gain * float(
                omega.get(coordinate, 0.0)
            )
            if abs(value) > eps:
                result[coordinate] = value
        return result

    def pi_lambda(self, candidate: DeepDistillerCandidate) -> tuple[bool, str | None]:
        """Constitutional admissibility gate over the complete candidate."""
        if len(candidate.state) > self.config.max_active_cells:
            return False, "active-cell budget exceeded"
        if candidate.latent_cells > self.config.max_latent_cells:
            return False, "latent-cell budget exceeded"
        if not self._finite(candidate.state) or not self._finite(candidate.omega):
            return False, "non-finite state or Omega"
        for value in candidate.state.values():
            if not self.config.value_min <= float(value) <= self.config.value_max:
                return False, "state value outside constitutional bounds"
        try:
            self._validate_theta(candidate.theta)
        except (TypeError, ValueError) as exc:
            return False, f"Theta rejected: {exc}"
        if self.gate is not None and not bool(self.gate(candidate)):
            return False, "external Pi_Lambda policy rejected candidate"
        return True, None

    def step(self) -> DeepDistillerReport:
        self._require_loaded()
        current = dict(self._state)
        current_theta = self._theta

        latent = self.encode(current, current_theta)
        reconstructed = self.decode(latent, current_theta)
        error = self.residual(current, reconstructed)
        omega_candidate = self.omega_update(error)
        grad_encoder, grad_decoder = self.residual_gradient(
            current, latent, error, current_theta
        )
        theta_candidate = self.theta_candidate(current_theta, grad_encoder, grad_decoder)
        next_state = self.state_candidate(reconstructed, omega_candidate)
        candidate = DeepDistillerCandidate(
            state=next_state,
            omega=omega_candidate,
            theta=theta_candidate,
            latent_cells=len(latent),
        )

        residual_rms = self._rms(error)
        omega_rms = self._rms(omega_candidate)
        loss = residual_rms * residual_rms
        gate_passed, rejection_reason = self.pi_lambda(candidate)
        committed = gate_passed

        if committed:
            # Atomic authoritative commit boundary.
            self._state = dict(candidate.state)
            self._omega = dict(candidate.omega)
            self._theta = candidate.theta
            self._iteration += 1

        converged = bool(committed and residual_rms <= self.config.residual_tolerance)
        authoritative_state = self._state if committed else current
        authoritative_theta = self._theta if committed else current_theta
        provisional = DeepDistillerReport(
            iteration=self._iteration,
            committed=committed,
            converged=converged,
            active_cells=len(authoritative_state),
            latent_cells=len(latent),
            residual_rms=residual_rms,
            omega_rms=omega_rms,
            loss=loss,
            grad_encoder=grad_encoder,
            grad_decoder=grad_decoder,
            encoder_gain=authoritative_theta.encoder_gain,
            decoder_gain=authoritative_theta.decoder_gain,
            gate_passed=gate_passed,
            rejection_reason=rejection_reason,
            state_hash=self._state_hash(authoritative_state),
            theta_hash=self._theta_hash(authoritative_theta),
        )
        record = provisional.as_dict()
        record.pop("journal_hash", None)
        digest = self.journal.append(record)
        report = replace(provisional, journal_hash=digest)
        self.reports.append(report)
        return report

    def run(self, max_iterations: int | None = None) -> tuple[DeepDistillerReport, ...]:
        self._require_loaded()
        limit = self.config.max_iterations if max_iterations is None else max_iterations
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("max_iterations must be a positive integer")
        for _ in range(limit):
            report = self.step()
            if report.converged or not report.committed:
                break
        return tuple(self.reports)

    def status(self) -> dict[str, object]:
        self._require_loaded()
        latest = self.reports[-1] if self.reports else None
        return {
            "law": self.LAW_ID,
            "product": self.PRODUCT_NAME,
            "ip_locked": True,
            "commit_policy": self.COMMIT_POLICY,
            "iteration": self._iteration,
            "logical_side": self.config.logical_side,
            "materialization": "sparse-active-support-only",
            "active_cells": len(self._state),
            "max_active_cells": self.config.max_active_cells,
            "max_latent_cells": self.config.max_latent_cells,
            "theta": asdict(self._theta),
            "residual_rms": latest.residual_rms if latest else None,
            "converged": latest.converged if latest else False,
            "state_hash": self._state_hash(self._state),
            "theta_hash": self._theta_hash(self._theta),
            "journal_head": self.journal.head,
            "journal_valid": self.journal.verify(),
        }

    def _validate_theta(self, theta: DeepDistillerTheta) -> None:
        for name in ("encoder_gain", "decoder_gain"):
            value = getattr(theta, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
            if not self.config.theta_min <= float(value) <= self.config.theta_max:
                raise ValueError(f"{name} outside configured Theta bounds")

    @staticmethod
    def _finite(field: Mapping[Coordinate, float]) -> bool:
        return all(math.isfinite(float(value)) for value in field.values())

    @staticmethod
    def _rms(field: Mapping[Coordinate, float]) -> float:
        if not field:
            return 0.0
        return math.sqrt(sum(float(value) ** 2 for value in field.values()) / len(field))

    @staticmethod
    def _state_hash(field: Mapping[Coordinate, float]) -> str:
        canonical = [
            [coordinate[0], coordinate[1], coordinate[2], float(field[coordinate])]
            for coordinate in sorted(field)
        ]
        payload = json.dumps(canonical, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _theta_hash(theta: DeepDistillerTheta) -> str:
        payload = json.dumps(asdict(theta), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _require_loaded(self) -> None:
        if not self._loaded:
            raise RuntimeError("load a sparse 3D field before Deep Distiller execution")
