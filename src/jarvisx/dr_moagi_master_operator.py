"""Executable reference for the Dr Moagi fully operational 3D master operator.

The module gives finite, auditable semantics to the symbolic system

    S_Moagi = {P, M, mu, E_phi, D_theta, inward_loop}

without materialising the logical 10^6 x 10^6 x 10^6 domain. Only active
coordinates are resident. The 1 GB/ns multiplex is represented as a logical
bandwidth budget plus deterministic lane receipts; it is not a benchmark claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

Coordinate = tuple[int, int, int]
SparseField = dict[Coordinate, float]


@dataclass(frozen=True)
class MoagiMasterConfig:
    logical_side: int = 1_000_000
    latent_dim: int = 8
    lanes: int = 32
    bandwidth_budget_bps: int = 10**18
    beta_kl: float = 1.0e-3
    fixed_point_tolerance: float = 1.0e-8
    max_iterations: int = 64
    max_active_cells: int = 100_000
    value_bound: float = 1.0e6
    residual_decay: float = 0.98
    residual_floor: float = 0.2
    residual_reset: float = 5.6
    residual_initial: float = 5.6
    tau_nodes: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    tau_weights: tuple[float, ...] = (0.2, 0.2, 0.2, 0.2, 0.2)

    def __post_init__(self) -> None:
        if self.logical_side <= 0:
            raise ValueError("logical_side must be positive")
        if self.latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if self.lanes <= 0:
            raise ValueError("lanes must be positive")
        if self.bandwidth_budget_bps <= 0:
            raise ValueError("bandwidth_budget_bps must be positive")
        if self.max_iterations <= 0 or self.max_active_cells <= 0:
            raise ValueError("iteration and active-cell budgets must be positive")
        if len(self.tau_nodes) == 0 or len(self.tau_nodes) != len(self.tau_weights):
            raise ValueError("tau_nodes and tau_weights must have the same non-zero length")
        if any((not math.isfinite(v)) for v in self.tau_nodes + self.tau_weights):
            raise ValueError("quadrature nodes/weights must be finite")
        if any(weight < 0.0 for weight in self.tau_weights):
            raise ValueError("tau_weights must be non-negative")
        if sum(self.tau_weights) <= 0.0:
            raise ValueError("tau_weights must have positive mass")
        for name in (
            "beta_kl",
            "fixed_point_tolerance",
            "value_bound",
            "residual_decay",
            "residual_floor",
            "residual_reset",
            "residual_initial",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.beta_kl < 0.0:
            raise ValueError("beta_kl must be non-negative")
        if self.fixed_point_tolerance < 0.0:
            raise ValueError("fixed_point_tolerance must be non-negative")
        if self.value_bound <= 0.0:
            raise ValueError("value_bound must be positive")
        if not 0.0 < self.residual_decay < 1.0:
            raise ValueError("residual_decay must be in (0, 1)")
        if self.residual_floor < 0.0 or self.residual_reset <= self.residual_floor:
            raise ValueError("residual reset must exceed the non-negative floor")
        if self.residual_initial < 0.0:
            raise ValueError("residual_initial must be non-negative")

    @property
    def logical_voxels(self) -> int:
        return self.logical_side**3


@dataclass(frozen=True)
class LossState:
    reconstruction_mse: float
    kl_divergence: float
    total: float


@dataclass(frozen=True)
class MultiplexReceipt:
    symbolic_measure: str
    lane_count: int
    total_bandwidth_budget_bps: int
    lane_bandwidth_budget_bps: tuple[int, ...]
    lane_digests: tuple[str, ...]
    resident_active_cells: int


@dataclass(frozen=True)
class MasterStepReport:
    iteration: int
    active_cells: int
    latent_dim: int
    operator_delta_rms: float
    autoencoder_consistency_rms: float
    loss: LossState
    residual_control: float
    residual_reset_applied: bool
    converged: bool
    state_hash: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class DrMoagiMasterOperator:
    """Sparse reference implementation of P_t[M] and the inward recurrence."""

    LAW_ID = "S_Moagi::P_t[M]"

    def __init__(self, config: MoagiMasterConfig | None = None) -> None:
        self.config = config or MoagiMasterConfig()
        self._state: SparseField = {}
        self._iteration = 0
        self._residual_control = self.config.residual_initial
        self.reports: list[MasterStepReport] = []

    def load(self, field: Mapping[Coordinate, float]) -> SparseField:
        if not field:
            raise ValueError("field must contain at least one active coordinate")
        if len(field) > self.config.max_active_cells:
            raise RuntimeError("active-cell budget exceeded")
        parsed: SparseField = {}
        for raw_coordinate, raw_value in field.items():
            coordinate = self._validate_coordinate(raw_coordinate)
            if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                raise TypeError("field values must be numeric")
            value = float(raw_value)
            if not math.isfinite(value):
                raise ValueError("field values must be finite")
            if abs(value) > self.config.value_bound:
                raise ValueError("field value exceeds configured bound")
            parsed[coordinate] = value
        self._state = dict(sorted(parsed.items()))
        self._iteration = 0
        self._residual_control = self.config.residual_initial
        self.reports.clear()
        return self.snapshot()

    def snapshot(self) -> SparseField:
        self._require_loaded()
        return dict(self._state)

    def encode(self, field: Mapping[Coordinate, float]) -> tuple[float, ...]:
        """E_phi: sparse logical 10^18-voxel field -> R^8 by deterministic bucket means."""
        sums = [0.0] * self.config.latent_dim
        counts = [0] * self.config.latent_dim
        for coordinate, value in field.items():
            bucket = self._bucket(coordinate)
            sums[bucket] += float(value)
            counts[bucket] += 1
        return tuple(
            sums[i] / counts[i] if counts[i] else 0.0
            for i in range(self.config.latent_dim)
        )

    def decode(self, latent: Sequence[float], support: Sequence[Coordinate]) -> SparseField:
        """D_theta: R^8 -> the requested sparse support of the logical 3D domain."""
        if len(latent) != self.config.latent_dim:
            raise ValueError("latent width does not match latent_dim")
        values = tuple(float(v) for v in latent)
        if any(not math.isfinite(v) for v in values):
            raise ValueError("latent values must be finite")
        return {coordinate: values[self._bucket(coordinate)] for coordinate in support}

    def loss_and_gradient(
        self, field: Mapping[Coordinate, float]
    ) -> tuple[LossState, SparseField]:
        """Return L = mean(||X-X_hat||^2) + beta*D_KL and its analytic gradient."""
        support = tuple(sorted(field))
        if not support:
            raise ValueError("loss requires non-empty support")
        latent = self.encode(field)
        reconstruction = self.decode(latent, support)
        n = len(support)
        residual = {
            coordinate: float(field[coordinate]) - reconstruction[coordinate]
            for coordinate in support
        }
        mse = sum(value * value for value in residual.values()) / n

        probabilities = self._softmax(latent)
        dim = self.config.latent_dim
        kl = sum(p * math.log(p * dim) for p in probabilities)
        total = mse + self.config.beta_kl * kl

        counts = [0] * dim
        for coordinate in support:
            counts[self._bucket(coordinate)] += 1
        grad_kl_z = tuple(
            p * (math.log(p * dim) - kl) for p in probabilities
        )
        gradient: SparseField = {}
        for coordinate in support:
            bucket = self._bucket(coordinate)
            mse_gradient = 2.0 * residual[coordinate] / n
            kl_gradient = grad_kl_z[bucket] / counts[bucket]
            gradient[coordinate] = (
                mse_gradient + self.config.beta_kl * kl_gradient
            )
        return LossState(mse, kl, total), gradient

    def master_operator(self, field: Mapping[Coordinate, float]) -> SparseField:
        """Apply P_t[M](X) using the configured finite measure over tau."""
        support = tuple(sorted(field))
        _, gradient = self.loss_and_gradient(field)
        mass = sum(self.config.tau_weights)
        result = {coordinate: 0.0 for coordinate in support}

        for tau, raw_weight in zip(self.config.tau_nodes, self.config.tau_weights):
            weight = raw_weight / mass
            modulated: SparseField = {}
            for coordinate in support:
                exponent = max(
                    -60.0, min(60.0, -gradient[coordinate] * tau)
                )
                modulated[coordinate] = (
                    float(field[coordinate]) * math.exp(exponent)
                )
            latent = self.encode(modulated)
            decoded = self.decode(latent, support)
            for coordinate in support:
                result[coordinate] += weight * decoded[coordinate]

        bound = self.config.value_bound
        return {
            coordinate: max(-bound, min(bound, value))
            for coordinate, value in result.items()
        }

    def step(self) -> MasterStepReport:
        self._require_loaded()
        current = self.snapshot()
        candidate = self.master_operator(current)
        delta = self._rms(current, candidate)
        ae_reconstruction = self.decode(
            self.encode(candidate), tuple(sorted(candidate))
        )
        ae_consistency = self._rms(candidate, ae_reconstruction)
        loss, _ = self.loss_and_gradient(candidate)

        next_residual = self.config.residual_decay * self._residual_control
        reset_applied = next_residual < self.config.residual_floor
        if reset_applied:
            next_residual = self.config.residual_reset
        self._residual_control = next_residual

        self._state = candidate
        self._iteration += 1
        converged = (
            delta <= self.config.fixed_point_tolerance
            and ae_consistency <= self.config.fixed_point_tolerance
        )
        report = MasterStepReport(
            iteration=self._iteration,
            active_cells=len(candidate),
            latent_dim=self.config.latent_dim,
            operator_delta_rms=delta,
            autoencoder_consistency_rms=ae_consistency,
            loss=loss,
            residual_control=self._residual_control,
            residual_reset_applied=reset_applied,
            converged=converged,
            state_hash=self._state_hash(candidate),
        )
        self.reports.append(report)
        return report

    def run_until_fixed_point(self) -> tuple[MasterStepReport, ...]:
        reports: list[MasterStepReport] = []
        for _ in range(self.config.max_iterations):
            report = self.step()
            reports.append(report)
            if report.converged:
                break
        return tuple(reports)

    def multiplex_receipt(self) -> MultiplexReceipt:
        """Return mu(X)=tensor_product_l X^l as symbolic lane receipts."""
        self._require_loaded()
        lane_digests: list[str] = []
        for lane in range(self.config.lanes):
            payload = [
                [*coordinate, value]
                for coordinate, value in sorted(self._state.items())
                if self._lane(coordinate) == lane
            ]
            encoded = json.dumps(
                payload, separators=(",", ":"), sort_keys=False
            ).encode("utf-8")
            lane_digests.append(hashlib.sha256(encoded).hexdigest())

        base, remainder = divmod(
            self.config.bandwidth_budget_bps, self.config.lanes
        )
        budgets = tuple(
            base + (1 if lane < remainder else 0)
            for lane in range(self.config.lanes)
        )
        return MultiplexReceipt(
            symbolic_measure=(
                f"tensor_product(l=0..{self.config.lanes - 1}) X^l"
            ),
            lane_count=self.config.lanes,
            total_bandwidth_budget_bps=self.config.bandwidth_budget_bps,
            lane_bandwidth_budget_bps=budgets,
            lane_digests=tuple(lane_digests),
            resident_active_cells=len(self._state),
        )

    def status(self) -> dict[str, object]:
        self._require_loaded()
        return {
            "law": self.LAW_ID,
            "logical_shape": [self.config.logical_side] * 3,
            "logical_voxels": str(self.config.logical_voxels),
            "resident_active_cells": len(self._state),
            "latent_dim": self.config.latent_dim,
            "measure": "normalized discrete quadrature over tau",
            "multiplex": f"{self.config.lanes} symbolic lanes",
            "bandwidth_budget_bps": str(self.config.bandwidth_budget_bps),
            "bandwidth_semantics": (
                "configured logical budget; not measured physical throughput"
            ),
            "fixed_point": (
                "X* = P_t[M](X*) with AE consistency "
                "X*=D_theta(E_phi(X*))"
            ),
            "residual_control": self._residual_control,
        }

    def _validate_coordinate(self, coordinate: object) -> Coordinate:
        if not isinstance(coordinate, (tuple, list)) or len(coordinate) != 3:
            raise TypeError(
                "coordinate must contain exactly three integer axes"
            )
        x, y, z = coordinate
        for axis in (x, y, z):
            if isinstance(axis, bool) or not isinstance(axis, int):
                raise TypeError("coordinate axes must be integers")
            if not 0 <= axis < self.config.logical_side:
                raise ValueError("coordinate outside logical 3D domain")
        return int(x), int(y), int(z)

    def _bucket(self, coordinate: Coordinate) -> int:
        x, y, z = coordinate
        return (
            (x * 73_856_093)
            ^ (y * 19_349_663)
            ^ (z * 83_492_791)
        ) % self.config.latent_dim

    def _lane(self, coordinate: Coordinate) -> int:
        x, y, z = coordinate
        return (
            (x * 2_654_435_761)
            ^ (y * 2_246_822_519)
            ^ (z * 3_266_489_917)
        ) % self.config.lanes

    @staticmethod
    def _softmax(values: Sequence[float]) -> tuple[float, ...]:
        maximum = max(values)
        exps = tuple(math.exp(value - maximum) for value in values)
        total = sum(exps)
        return tuple(value / total for value in exps)

    @staticmethod
    def _rms(
        left: Mapping[Coordinate, float],
        right: Mapping[Coordinate, float],
    ) -> float:
        support = sorted(set(left) | set(right))
        if not support:
            return 0.0
        return math.sqrt(
            sum(
                (
                    float(left.get(coordinate, 0.0))
                    - float(right.get(coordinate, 0.0))
                )
                ** 2
                for coordinate in support
            )
            / len(support)
        )

    @staticmethod
    def _state_hash(field: Mapping[Coordinate, float]) -> str:
        payload = [
            [*coordinate, float(value)]
            for coordinate, value in sorted(field.items())
        ]
        encoded = json.dumps(
            payload, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _require_loaded(self) -> None:
        if not self._state:
            raise RuntimeError(
                "load a non-empty sparse field before execution"
            )


def _demo() -> dict[str, object]:
    engine = DrMoagiMasterOperator(MoagiMasterConfig(beta_kl=0.0))
    engine.load(
        {
            (0, 0, 0): 0.75,
            (1, 0, 0): -0.25,
            (2, 3, 5): 0.5,
        }
    )
    reports = engine.run_until_fixed_point()
    return {
        "status": engine.status(),
        "last_report": reports[-1].as_dict(),
        "multiplex": asdict(engine.multiplex_receipt()),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Dr Moagi sparse 3D master-operator reference"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="run the deterministic bounded demo",
    )
    args = parser.parse_args(argv)
    if not args.demo:
        parser.error("use --demo")
    print(json.dumps(_demo(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
