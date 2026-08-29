"""End-to-end Dr Moagi 3D phase-space + implicit-field + runtime optimizer.

This executable closes the current Jarvis-X research loop without conflating
physical phase evolution, neural learning, and runtime optimization:

    relativistic 3D phase state
      -> measured phase update
      -> implicit shell representation
      -> differentiable geometry update
      -> measured execution benchmark
      -> bounded runtime candidate search
      -> canonical kinetic transaction gate
      -> commit/rollback receipt
      -> next cycle

The phase state is authoritative. Neural error never directly changes the phase
integrator. Wall-clock measurements never enter autograd. Runtime candidates are
promoted only through Jarvis-X's canonical candidate-first transaction law.

Install and run:

    pip install -e '.[torch]'
    python examples/dr_moagi_phase3d_runtime.py --device cpu

On supported CUDA systems, PyTorch/Inductor may lower compiled candidates to
Triton-generated kernels. This module does not contain hand-written Triton kernels.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from jarvisx.kinetic_runtime import KineticReceipt, KineticTransactionEngine, ValidatorResult
from self_referential_triton_engine import (
    BenchmarkResult,
    ExecutionMetrics,
    RuntimeAutotuner,
    RuntimeConfig,
    SelfReferentialTritonEngine,
    _run_chunked,
    _sync,
)


C_LIGHT = 299_792_458.0


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


@dataclass(frozen=True)
class PhaseFieldConfig:
    """Dimensionally explicit central field and deterministic torus initializer."""

    node_count: int = 16_384
    major_radius_m: float = 2.0
    minor_radius_m: float = 0.5
    inverse_square_strength_n_m2: float = 15.0e6
    spring_constant_n_per_m: float = 5.0e6
    base_mass_kg: float = 1.0
    mass_step_kg: float = 0.1
    mass_period: int = 7
    damping_per_second: float = 0.0
    dt_seconds: float = 1.0e-5

    def __post_init__(self) -> None:
        if self.node_count <= 0:
            raise ValueError("node_count must be positive")
        if self.major_radius_m <= 0.0 or self.minor_radius_m <= 0.0:
            raise ValueError("torus radii must be positive")
        if self.inverse_square_strength_n_m2 <= 0.0:
            raise ValueError("inverse_square_strength_n_m2 must be positive")
        if self.spring_constant_n_per_m <= 0.0:
            raise ValueError("spring_constant_n_per_m must be positive")
        if self.base_mass_kg <= 0.0 or self.mass_step_kg < 0.0:
            raise ValueError("masses must be positive with non-negative step")
        if self.mass_period <= 0:
            raise ValueError("mass_period must be positive")
        if self.damping_per_second < 0.0:
            raise ValueError("damping_per_second must be non-negative")
        if self.dt_seconds <= 0.0:
            raise ValueError("dt_seconds must be positive")

    @property
    def equilibrium_radius_m(self) -> float:
        return (
            self.inverse_square_strength_n_m2 / self.spring_constant_n_per_m
        ) ** (1.0 / 3.0)


@dataclass(frozen=True)
class PhaseTelemetry:
    node_updates_per_second: float
    latency_ms: float
    max_velocity_m_s: float
    mean_radius_m: float
    shell_rmse_m: float
    kinetic_energy_j: float
    potential_energy_j: float
    mechanical_energy_j: float
    relative_energy_drift: float


@dataclass(frozen=True)
class RuntimeSearchCandidate:
    incumbent: BenchmarkResult
    best: BenchmarkResult
    evaluated_candidates: int


@dataclass(frozen=True)
class CycleReport:
    cycle: int
    phase: PhaseTelemetry
    geometry_loss: float
    geometry_rmse: float
    runtime: BenchmarkResult
    runtime_promoted: bool
    runtime_decision: str
    runtime_receipt_hash: str


class RelativisticPhaseField3D:
    """Vectorized N x 3 relativistic momentum integrator.

    The conservative radial field is

        F_r = A / r^2 - k r

    with A in N*m^2 and k in N/m. Optional linear momentum damping adds
    ``-zeta * p`` to dp/dt. With damping disabled the integrator is a
    semi-implicit Hamiltonian reference; with damping enabled the shell becomes
    an attracting dissipative equilibrium.
    """

    def __init__(
        self,
        config: PhaseFieldConfig,
        device: torch.device,
        *,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        self.config = config
        self.device = device
        self.dtype = dtype
        self.position = self._torus_surface(config.node_count)
        self.momentum = torch.zeros_like(self.position)
        indices = torch.arange(config.node_count, device=device, dtype=torch.int64)
        masses = config.base_mass_kg + config.mass_step_kg * (
            indices.remainder(config.mass_period).to(dtype)
        )
        self.mass = masses[:, None]
        self.initial_mechanical_energy = float(self.mechanical_energy().item())

    def _torus_surface(self, count: int) -> torch.Tensor:
        """Sample a true two-parameter torus surface deterministically."""

        n_theta = int(math.ceil(math.sqrt(count)))
        n_phi = int(math.ceil(count / n_theta))
        theta = torch.arange(n_theta, device=self.device, dtype=self.dtype)
        phi = torch.arange(n_phi, device=self.device, dtype=self.dtype)
        theta = theta * (2.0 * math.pi / n_theta)
        phi = phi * (2.0 * math.pi / n_phi)
        theta_grid, phi_grid = torch.meshgrid(theta, phi, indexing="ij")

        rho = self.config.major_radius_m + self.config.minor_radius_m * torch.cos(phi_grid)
        x = rho * torch.cos(theta_grid)
        y = rho * torch.sin(theta_grid)
        z = self.config.minor_radius_m * torch.sin(phi_grid)
        return torch.stack((x, y, z), dim=-1).reshape(-1, 3)[:count].contiguous()

    def radial_force(self) -> torch.Tensor:
        radius = torch.linalg.vector_norm(self.position, dim=-1, keepdim=True).clamp_min(
            1.0e-12
        )
        direction = self.position / radius
        magnitude = (
            self.config.inverse_square_strength_n_m2 / radius.square()
            - self.config.spring_constant_n_per_m * radius
        )
        force = magnitude * direction
        if self.config.damping_per_second > 0.0:
            force = force - self.config.damping_per_second * self.momentum
        return force

    def gamma(self) -> torch.Tensor:
        p2 = self.momentum.square().sum(dim=-1, keepdim=True)
        denominator = self.mass.square() * (C_LIGHT**2)
        return torch.sqrt(1.0 + p2 / denominator)

    def velocity(self) -> torch.Tensor:
        return self.momentum / (self.gamma() * self.mass)

    def kinetic_energy(self) -> torch.Tensor:
        """Stable relativistic kinetic energy sum, p^2 / (m * (gamma + 1))."""

        p2 = self.momentum.square().sum(dim=-1, keepdim=True)
        return (p2 / (self.mass * (self.gamma() + 1.0))).sum()

    def potential_energy(self) -> torch.Tensor:
        radius = torch.linalg.vector_norm(self.position, dim=-1).clamp_min(1.0e-12)
        potential = (
            self.config.inverse_square_strength_n_m2 / radius
            + 0.5 * self.config.spring_constant_n_per_m * radius.square()
        )
        return potential.sum()

    def mechanical_energy(self) -> torch.Tensor:
        return self.kinetic_energy() + self.potential_energy()

    @torch.inference_mode()
    def step(self) -> PhaseTelemetry:
        _sync(self.device)
        started = time.perf_counter()

        force = self.radial_force()
        self.momentum.add_(force, alpha=self.config.dt_seconds)
        velocity = self.velocity()
        self.position.add_(velocity, alpha=self.config.dt_seconds)

        _sync(self.device)
        elapsed = time.perf_counter() - started

        radius = torch.linalg.vector_norm(self.position, dim=-1)
        kinetic = float(self.kinetic_energy().item())
        potential = float(self.potential_energy().item())
        mechanical = kinetic + potential
        reference = max(abs(self.initial_mechanical_energy), 1.0e-12)
        drift = (mechanical - self.initial_mechanical_energy) / reference
        shell_error = radius - self.config.equilibrium_radius_m

        return PhaseTelemetry(
            node_updates_per_second=self.config.node_count / max(elapsed, 1.0e-12),
            latency_ms=elapsed * 1_000.0,
            max_velocity_m_s=float(torch.linalg.vector_norm(velocity, dim=-1).max().item()),
            mean_radius_m=float(radius.mean().item()),
            shell_rmse_m=float(torch.sqrt(torch.mean(shell_error.square())).item()),
            kinetic_energy_j=kinetic,
            potential_energy_j=potential,
            mechanical_energy_j=mechanical,
            relative_energy_drift=drift,
        )

    def neural_coordinates(self) -> torch.Tensor:
        """Return one B x N x 3 float32 view for the neural field."""

        return self.position.to(dtype=torch.float32).unsqueeze(0)


class Phase3DSelfOptimizingRuntime:
    """Close phase dynamics, neural learning, profiling, and canonical promotion."""

    def __init__(
        self,
        phase_config: PhaseFieldConfig,
        *,
        device: str = "cuda",
        target_qps: float = 1_000_000.0,
        runtime_config: RuntimeConfig | None = None,
        peak_memory_budget_mb: float = math.inf,
        min_runtime_improvement: float = 0.01,
        latent_dim: int = 128,
        hidden_dim: int = 256,
    ) -> None:
        self.device = torch.device(
            device if device == "cpu" or torch.cuda.is_available() else "cpu"
        )
        self.phase = RelativisticPhaseField3D(phase_config, self.device)
        self.engine = SelfReferentialTritonEngine(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)
        self.neural_optimizer = torch.optim.AdamW(
            self.engine.parameters(),
            lr=1.0e-3,
            weight_decay=1.0e-5,
        )
        self.target_qps = float(target_qps)
        self.runtime_config = runtime_config or RuntimeConfig()
        self.peak_memory_budget_mb = float(peak_memory_budget_mb)
        self.min_runtime_improvement = float(min_runtime_improvement)
        self.exec_state = ExecutionMetrics(
            throughput_qps=0.5 * self.target_qps,
            latency_ms=2.0,
        )
        self.autotuner = RuntimeAutotuner(
            self.engine,
            self.device,
            warmup=1,
            repeats=3,
        )

        self._transaction_coords: torch.Tensor | None = None
        self._transaction_metrics: torch.Tensor | None = None
        self._transaction_reference: torch.Tensor | None = None
        self.last_runtime_receipt: KineticReceipt | None = None
        self.runtime_transaction = KineticTransactionEngine[
            RuntimeConfig,
            BenchmarkResult,
            tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            RuntimeSearchCandidate,
        ](
            snapshot=lambda config: RuntimeConfig(config.chunk_size, config.compile_mode),
            observe=self._runtime_observe,
            encode=self._runtime_encode,
            propose=self._runtime_propose,
            shadow=self._runtime_shadow,
            validators=(
                self._runtime_semantic_validator,
                self._runtime_memory_validator,
                self._runtime_performance_validator,
            ),
            commit=lambda _config, candidate: candidate.best.config,
            rollback=lambda config: config,
            state_identity=lambda config: {
                "chunk_size": config.chunk_size,
                "compile_mode": config.compile_mode,
            },
            candidate_identity=self._runtime_candidate_identity,
        )

    def _target_sdf(self, coords: torch.Tensor) -> torch.Tensor:
        radius = self.phase.config.equilibrium_radius_m
        return torch.linalg.vector_norm(coords, dim=-1, keepdim=True) - radius

    def train_geometry(self, coords: torch.Tensor, steps: int) -> float:
        loss_value = math.inf
        for _ in range(steps):
            normalized = self.exec_state.normalized(
                self.target_qps,
                self.device,
                coords.shape[0],
            )
            target = self._target_sdf(coords)
            prediction = self.engine(coords, normalized)
            loss = F.smooth_l1_loss(prediction, target)
            self.neural_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.neural_optimizer.step()
            loss_value = float(loss.detach().item())
        return loss_value

    @torch.inference_mode()
    def geometry_rmse(self, coords: torch.Tensor) -> float:
        normalized = self.exec_state.normalized(
            self.target_qps,
            self.device,
            coords.shape[0],
        )
        prediction = self.engine(coords, normalized)
        target = self._target_sdf(coords)
        return float(torch.sqrt(torch.mean((prediction - target).square())).item())

    @torch.inference_mode()
    def _benchmark(
        self,
        config: RuntimeConfig,
        coords: torch.Tensor,
        metrics: torch.Tensor,
        reference: torch.Tensor,
    ) -> BenchmarkResult:
        return self.autotuner.benchmark(config, coords, metrics, reference)

    @torch.inference_mode()
    def _benchmark_incumbent(self, coords: torch.Tensor) -> BenchmarkResult:
        normalized = self.exec_state.normalized(
            self.target_qps,
            self.device,
            coords.shape[0],
        )
        reference = _run_chunked(
            self.engine,
            coords,
            normalized,
            self.runtime_config.chunk_size,
        )
        return self._benchmark(self.runtime_config, coords, normalized, reference)

    def _transaction_payload(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if (
            self._transaction_coords is None
            or self._transaction_metrics is None
            or self._transaction_reference is None
        ):
            raise RuntimeError("runtime transaction payload is not initialized")
        return (
            self._transaction_coords,
            self._transaction_metrics,
            self._transaction_reference,
        )

    @torch.inference_mode()
    def _runtime_observe(self, config: RuntimeConfig) -> BenchmarkResult:
        coords, metrics, reference = self._transaction_payload()
        return self._benchmark(config, coords, metrics, reference)

    def _runtime_encode(
        self,
        config: RuntimeConfig,
        observation: BenchmarkResult,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del config, observation
        return self._transaction_payload()

    @torch.inference_mode()
    def _runtime_propose(
        self,
        config: RuntimeConfig,
        observation: BenchmarkResult,
        encoded: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> RuntimeSearchCandidate:
        coords, metrics, reference = encoded
        results = [
            self._benchmark(candidate, coords, metrics, reference)
            for candidate in self.autotuner.candidates(config)
        ]
        admissible = [result for result in results if result.accepted]
        best = (
            max(admissible, key=lambda item: item.metrics.throughput_qps)
            if admissible
            else observation
        )
        return RuntimeSearchCandidate(
            incumbent=observation,
            best=best,
            evaluated_candidates=len(results),
        )

    def _runtime_shadow(
        self,
        config: RuntimeConfig,
        candidate: RuntimeSearchCandidate,
    ) -> dict[str, object]:
        del config
        return {
            "evaluated_candidates": candidate.evaluated_candidates,
            "incumbent_qps": _finite_or_none(candidate.incumbent.metrics.throughput_qps),
            "candidate_qps": _finite_or_none(candidate.best.metrics.throughput_qps),
            "candidate_latency_ms": _finite_or_none(candidate.best.metrics.latency_ms),
            "candidate_peak_memory_mb": _finite_or_none(
                candidate.best.metrics.peak_memory_mb
            ),
            "candidate_semantic_error": _finite_or_none(candidate.best.max_abs_error),
            "candidate_compile_mode": candidate.best.config.compile_mode,
            "candidate_effective_mode": candidate.best.effective_mode,
            "candidate_chunk_size": candidate.best.config.chunk_size,
        }

    def _runtime_semantic_validator(
        self,
        config: RuntimeConfig,
        candidate: RuntimeSearchCandidate,
    ) -> ValidatorResult:
        del config
        passed = candidate.best.accepted and math.isfinite(candidate.best.max_abs_error)
        return ValidatorResult(
            name="phase3d_semantic_equivalence",
            passed=passed,
            metrics={"max_abs_error": _finite_or_none(candidate.best.max_abs_error)},
            reason="semantic tolerance satisfied" if passed else "candidate semantic check failed",
        )

    def _runtime_memory_validator(
        self,
        config: RuntimeConfig,
        candidate: RuntimeSearchCandidate,
    ) -> ValidatorResult:
        del config
        peak = candidate.best.metrics.peak_memory_mb
        passed = math.isfinite(peak) and peak <= self.peak_memory_budget_mb
        return ValidatorResult(
            name="phase3d_memory_budget",
            passed=passed,
            metrics={
                "peak_memory_mb": _finite_or_none(peak),
                "budget_mb": _finite_or_none(self.peak_memory_budget_mb),
            },
            reason="within memory budget" if passed else "memory budget exceeded",
        )

    def _runtime_performance_validator(
        self,
        config: RuntimeConfig,
        candidate: RuntimeSearchCandidate,
    ) -> ValidatorResult:
        incumbent = candidate.incumbent
        best = candidate.best
        incumbent_qps = incumbent.metrics.throughput_qps
        best_qps = best.metrics.throughput_qps

        if not incumbent.accepted or not math.isfinite(incumbent_qps):
            passed = best.accepted and best.config != config and math.isfinite(best_qps)
            required = None
            reason = "recover unavailable incumbent" if passed else "no valid replacement"
        else:
            required_value = incumbent_qps * (1.0 + self.min_runtime_improvement)
            required = required_value
            passed = (
                best.config != config
                and math.isfinite(best_qps)
                and best_qps >= required_value
            )
            reason = "minimum throughput improvement satisfied" if passed else "retain incumbent"

        return ValidatorResult(
            name="phase3d_runtime_improvement",
            passed=passed,
            metrics={
                "incumbent_qps": _finite_or_none(incumbent_qps),
                "candidate_qps": _finite_or_none(best_qps),
                "required_qps": _finite_or_none(required) if required is not None else None,
                "min_relative_improvement": self.min_runtime_improvement,
            },
            reason=reason,
        )

    def _runtime_candidate_identity(self, candidate: RuntimeSearchCandidate) -> dict[str, object]:
        return {
            "evaluated_candidates": candidate.evaluated_candidates,
            "incumbent": {
                "chunk_size": candidate.incumbent.config.chunk_size,
                "compile_mode": candidate.incumbent.config.compile_mode,
                "accepted": candidate.incumbent.accepted,
                "qps": _finite_or_none(candidate.incumbent.metrics.throughput_qps),
            },
            "best": {
                "chunk_size": candidate.best.config.chunk_size,
                "compile_mode": candidate.best.config.compile_mode,
                "effective_mode": candidate.best.effective_mode,
                "accepted": candidate.best.accepted,
                "qps": _finite_or_none(candidate.best.metrics.throughput_qps),
                "latency_ms": _finite_or_none(candidate.best.metrics.latency_ms),
                "peak_memory_mb": _finite_or_none(candidate.best.metrics.peak_memory_mb),
                "max_abs_error": _finite_or_none(candidate.best.max_abs_error),
            },
        }

    @torch.inference_mode()
    def _tune_runtime(self, coords: torch.Tensor) -> tuple[BenchmarkResult, bool, KineticReceipt]:
        """Run runtime selection through the canonical Jarvis-X transaction law."""

        normalized = self.exec_state.normalized(
            self.target_qps,
            self.device,
            coords.shape[0],
        )
        reference = _run_chunked(
            self.engine,
            coords,
            normalized,
            self.runtime_config.chunk_size,
        )
        self._transaction_coords = coords
        self._transaction_metrics = normalized
        self._transaction_reference = reference

        result = self.runtime_transaction.step(self.runtime_config)
        self.runtime_config = result.state
        self.last_runtime_receipt = result.receipt
        reported = result.candidate.best if result.committed else result.candidate.incumbent
        return reported, result.committed, result.receipt

    def cycle(self, cycle_index: int, *, train_steps: int, tune: bool) -> CycleReport:
        phase_report = self.phase.step()
        coords = self.phase.neural_coordinates()
        geometry_loss = self.train_geometry(coords, train_steps)

        measured = self._benchmark_incumbent(coords)
        if measured.accepted:
            self.exec_state = measured.metrics

        promoted = False
        decision = "not_scheduled"
        receipt_hash = self.runtime_transaction.previous_receipt_hash
        if tune:
            measured, promoted, receipt = self._tune_runtime(coords)
            decision = receipt.decision
            receipt_hash = receipt.receipt_hash
            if measured.accepted:
                self.exec_state = measured.metrics

        rmse = self.geometry_rmse(coords)
        return CycleReport(
            cycle=cycle_index,
            phase=phase_report,
            geometry_loss=geometry_loss,
            geometry_rmse=rmse,
            runtime=measured,
            runtime_promoted=promoted,
            runtime_decision=decision,
            runtime_receipt_hash=receipt_hash,
        )

    def run(self, cycles: int, *, train_steps: int = 1, tune_every: int = 2) -> None:
        print(
            "DM-vOmegaXi+ Phase3D runtime | "
            f"device={self.device.type.upper()} "
            f"nodes={self.phase.config.node_count:,} "
            f"r*={self.phase.config.equilibrium_radius_m:.6f} m"
        )
        print(
            "measurement contract: node_updates/s and q/s are measured; "
            "runtime changes require canonical kinetic receipts"
        )

        for cycle_index in range(1, cycles + 1):
            report = self.cycle(
                cycle_index,
                train_steps=train_steps,
                tune=(tune_every > 0 and cycle_index % tune_every == 0),
            )
            phase = report.phase
            runtime = report.runtime
            print(
                f"cycle={cycle_index:03d} "
                f"phase={phase.node_updates_per_second:,.0f} nodes/s "
                f"r_mean={phase.mean_radius_m:.6f} m "
                f"shell_rmse={phase.shell_rmse_m:.6f} m "
                f"v_max={phase.max_velocity_m_s:,.2f} m/s "
                f"dE/E0={phase.relative_energy_drift:+.3e} "
                f"field_loss={report.geometry_loss:.6f} "
                f"field_rmse={report.geometry_rmse:.6f} "
                f"runtime={runtime.metrics.throughput_qps:,.0f} q/s "
                f"latency={runtime.metrics.latency_ms:.3f} ms "
                f"mem={runtime.metrics.peak_memory_mb:.1f} MiB "
                f"mode={runtime.effective_mode} "
                f"chunk={self.runtime_config.chunk_size} "
                f"decision={report.runtime_decision} "
                f"receipt={report.runtime_receipt_hash[:12]}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycles", type=int, default=5)
    parser.add_argument("--nodes", type=int, default=16_384)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--dt", type=float, default=1.0e-5)
    parser.add_argument("--damping", type=float, default=0.0)
    parser.add_argument("--train-steps", type=int, default=1)
    parser.add_argument("--tune-every", type=int, default=2)
    parser.add_argument("--target-qps", type=float, default=1_000_000.0)
    parser.add_argument("--chunk-size", type=int, default=65_536)
    parser.add_argument(
        "--compile-mode",
        choices=("eager", "default", "reduce-overhead", "max-autotune"),
        default="default",
    )
    parser.add_argument("--peak-memory-mb", type=float, default=math.inf)
    parser.add_argument("--min-runtime-improvement", type=float, default=0.01)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    phase_config = PhaseFieldConfig(
        node_count=args.nodes,
        dt_seconds=args.dt,
        damping_per_second=args.damping,
    )
    runtime = Phase3DSelfOptimizingRuntime(
        phase_config,
        device=args.device,
        target_qps=args.target_qps,
        runtime_config=RuntimeConfig(
            chunk_size=args.chunk_size,
            compile_mode=args.compile_mode,
        ),
        peak_memory_budget_mb=args.peak_memory_mb,
        min_runtime_improvement=args.min_runtime_improvement,
    )
    runtime.run(
        args.cycles,
        train_steps=args.train_steps,
        tune_every=args.tune_every,
    )


if __name__ == "__main__":
    main()
