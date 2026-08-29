"""Self-referential 3D implicit field with a separate runtime autotuning loop.

This example keeps two optimization spaces deliberately distinct:

1. ``theta`` -- neural parameters updated by differentiable geometry loss.
2. ``c``     -- runtime configuration selected by measured benchmark evidence.

Measured wall-clock throughput is not differentiable, so it is never inserted into
``total_loss`` as if autograd could optimize it. Instead, measured telemetry is
normalized and fed back into the next neural evaluation, while an outer controller
benchmarks bounded runtime candidates and promotes only semantically equivalent
variants.

Install the optional PyTorch dependency before running:

    pip install -e '.[torch]'
    python examples/self_referential_triton_engine.py

On CUDA, ``torch.compile``/Inductor may lower eligible graph regions to Triton
kernels. This file does not claim to be a hand-written Triton kernel.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


torch.set_float32_matmul_precision("high")


@dataclass(frozen=True)
class ExecutionMetrics:
    throughput_qps: float
    latency_ms: float
    peak_memory_mb: float = 0.0

    def normalized(
        self,
        target_qps: float,
        device: torch.device,
        batch_size: int,
    ) -> torch.Tensor:
        """Return numerically bounded execution-state features for the meta encoder."""

        target = max(float(target_qps), 1.0)
        throughput = math.log1p(max(self.throughput_qps, 0.0) / target)
        latency = math.log1p(max(self.latency_ms, 0.0))
        return torch.tensor(
            [[throughput, latency]],
            dtype=torch.float32,
            device=device,
        ).expand(batch_size, -1)


@dataclass(frozen=True, order=True)
class RuntimeConfig:
    chunk_size: int = 65_536
    compile_mode: str = "default"

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.compile_mode not in {"eager", "default", "reduce-overhead", "max-autotune"}:
            raise ValueError("unsupported compile_mode")


@dataclass(frozen=True)
class BenchmarkResult:
    config: RuntimeConfig
    metrics: ExecutionMetrics
    max_abs_error: float
    accepted: bool
    effective_mode: str


@dataclass(frozen=True)
class CompiledRunner:
    run: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    effective_mode: str
    available: bool = True


class SelfReferentialTritonEngine(nn.Module):
    """3D implicit field conditioned on normalized execution telemetry."""

    def __init__(self, latent_dim: int = 256, hidden_dim: int = 512) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.spatial_proj = nn.Linear(3, 64)
        self.meta_encoder = nn.Sequential(
            nn.Linear(66, 128),
            nn.SiLU(),
            nn.Linear(128, latent_dim),
        )
        self.core_pipeline = nn.Sequential(
            nn.Linear(latent_dim + 64, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, coords: torch.Tensor, exec_metrics: torch.Tensor) -> torch.Tensor:
        if coords.ndim != 3 or coords.shape[-1] != 3:
            raise ValueError("coords must have shape (B, N, 3)")
        if exec_metrics.ndim != 2 or exec_metrics.shape != (coords.shape[0], 2):
            raise ValueError("exec_metrics must have shape (B, 2)")

        spatial_feats = F.silu(self.spatial_proj(coords))
        metrics_expanded = exec_metrics[:, None, :].expand(-1, coords.shape[1], -1)
        z_meta = self.meta_encoder(torch.cat((spatial_feats, metrics_expanded), dim=-1))
        return self.core_pipeline(torch.cat((z_meta, spatial_feats), dim=-1))


def sphere_sdf(coords: torch.Tensor, radius: float = 0.65) -> torch.Tensor:
    """Analytic non-trivial SDF target used by the reference training loop."""

    return torch.linalg.vector_norm(coords, dim=-1, keepdim=True) - radius


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _compile(engine: nn.Module, config: RuntimeConfig) -> CompiledRunner:
    if config.compile_mode == "eager":
        return CompiledRunner(engine, "eager")
    if not hasattr(torch, "compile"):
        return CompiledRunner(engine, "eager", available=False)
    try:
        compiled = torch.compile(engine, mode=config.compile_mode)
    except Exception as exc:  # pragma: no cover - backend availability is platform specific
        print(f"[compile] {config.compile_mode!r} unavailable ({exc})")
        return CompiledRunner(engine, "eager", available=False)
    return CompiledRunner(compiled, config.compile_mode)


def _run_chunked(
    runner: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    coords: torch.Tensor,
    metrics: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    for start in range(0, coords.shape[1], chunk_size):
        end = min(coords.shape[1], start + chunk_size)
        outputs.append(runner(coords[:, start:end, :], metrics))
    return torch.cat(outputs, dim=1)


class RuntimeAutotuner:
    """Bounded, measurement-driven optimizer over runtime configuration only."""

    def __init__(
        self,
        engine: SelfReferentialTritonEngine,
        device: torch.device,
        *,
        semantic_tolerance: float = 5.0e-4,
        warmup: int = 2,
        repeats: int = 3,
    ) -> None:
        self.engine = engine
        self.device = device
        self.semantic_tolerance = semantic_tolerance
        self.warmup = warmup
        self.repeats = repeats
        self._cache: dict[RuntimeConfig, CompiledRunner] = {}

    def runner(self, config: RuntimeConfig) -> CompiledRunner:
        if config not in self._cache:
            self._cache[config] = _compile(self.engine, config)
        return self._cache[config]

    def candidates(self, incumbent: RuntimeConfig) -> tuple[RuntimeConfig, ...]:
        chunks = sorted(
            {
                max(1_024, incumbent.chunk_size // 2),
                incumbent.chunk_size,
                min(262_144, incumbent.chunk_size * 2),
            }
        )
        modes = (incumbent.compile_mode, "eager", "default", "reduce-overhead")
        unique: dict[RuntimeConfig, None] = {}
        for chunk in chunks:
            for mode in modes:
                unique[RuntimeConfig(chunk, mode)] = None
        return tuple(unique)

    @torch.inference_mode()
    def benchmark(
        self,
        config: RuntimeConfig,
        coords: torch.Tensor,
        metrics: torch.Tensor,
        reference: torch.Tensor,
    ) -> BenchmarkResult:
        variant = self.runner(config)
        if not variant.available:
            return BenchmarkResult(
                config=config,
                metrics=ExecutionMetrics(0.0, math.inf, 0.0),
                max_abs_error=math.inf,
                accepted=False,
                effective_mode=variant.effective_mode,
            )

        try:
            for _ in range(self.warmup):
                _run_chunked(variant.run, coords, metrics, config.chunk_size)
                _sync(self.device)
        except Exception as exc:  # pragma: no cover - backend failures are platform specific
            print(f"[benchmark] {config} failed during warm-up ({exc})")
            return BenchmarkResult(
                config=config,
                metrics=ExecutionMetrics(0.0, math.inf, 0.0),
                max_abs_error=math.inf,
                accepted=False,
                effective_mode=variant.effective_mode,
            )

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        durations: list[float] = []
        output = reference
        try:
            for _ in range(self.repeats):
                _sync(self.device)
                started = time.perf_counter()
                output = _run_chunked(variant.run, coords, metrics, config.chunk_size)
                _sync(self.device)
                durations.append(time.perf_counter() - started)
        except Exception as exc:  # pragma: no cover - backend failures are platform specific
            print(f"[benchmark] {config} failed during measurement ({exc})")
            return BenchmarkResult(
                config=config,
                metrics=ExecutionMetrics(0.0, math.inf, 0.0),
                max_abs_error=math.inf,
                accepted=False,
                effective_mode=variant.effective_mode,
            )

        duration = min(durations)
        queries = coords.shape[0] * coords.shape[1]
        throughput = queries / max(duration, 1.0e-9)
        peak_mb = 0.0
        if self.device.type == "cuda":
            peak_mb = torch.cuda.max_memory_allocated(self.device) / (1024.0 * 1024.0)

        max_abs_error = float((output - reference).abs().max().item())
        accepted = math.isfinite(throughput) and max_abs_error <= self.semantic_tolerance
        return BenchmarkResult(
            config=config,
            metrics=ExecutionMetrics(
                throughput_qps=throughput,
                latency_ms=duration * 1_000.0,
                peak_memory_mb=peak_mb,
            ),
            max_abs_error=max_abs_error,
            accepted=accepted,
            effective_mode=variant.effective_mode,
        )

    @torch.inference_mode()
    def tune(
        self,
        incumbent: RuntimeConfig,
        coords: torch.Tensor,
        metrics: torch.Tensor,
    ) -> BenchmarkResult:
        """Benchmark incumbent and candidates under one identical telemetry snapshot."""

        reference = _run_chunked(self.engine, coords, metrics, incumbent.chunk_size)
        results = [
            self.benchmark(candidate, coords, metrics, reference)
            for candidate in self.candidates(incumbent)
        ]
        admissible = [result for result in results if result.accepted]
        if not admissible:
            return BenchmarkResult(
                config=incumbent,
                metrics=ExecutionMetrics(0.0, math.inf, 0.0),
                max_abs_error=math.inf,
                accepted=False,
                effective_mode="unavailable",
            )

        incumbent_result = next(
            (result for result in admissible if result.config == incumbent),
            None,
        )
        best = max(admissible, key=lambda item: item.metrics.throughput_qps)
        if incumbent_result is not None and (
            best.metrics.throughput_qps <= incumbent_result.metrics.throughput_qps
        ):
            return incumbent_result
        return best


class SelfRefiningOptimizer:
    """Closed loop: gradient geometry learning + measured runtime configuration search."""

    def __init__(
        self,
        engine: SelfReferentialTritonEngine,
        *,
        target_qps: float = 1_000_000.0,
        device: str = "cuda",
        runtime_config: RuntimeConfig | None = None,
    ) -> None:
        self.device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
        self.engine = engine.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.engine.parameters(),
            lr=1.0e-3,
            weight_decay=1.0e-5,
        )
        self.target_qps = float(target_qps)
        self.runtime_config = runtime_config or RuntimeConfig()
        self.autotuner = RuntimeAutotuner(self.engine, self.device)
        self.exec_state = ExecutionMetrics(
            throughput_qps=0.5 * self.target_qps,
            latency_ms=2.0,
        )

    def _train_step(self, coords: torch.Tensor) -> float:
        normalized = self.exec_state.normalized(self.target_qps, self.device, coords.shape[0])
        target = sphere_sdf(coords)
        prediction = self.engine(coords, normalized)
        geometry_loss = F.smooth_l1_loss(prediction, target)

        self.optimizer.zero_grad(set_to_none=True)
        geometry_loss.backward()
        self.optimizer.step()
        return float(geometry_loss.detach().item())

    @torch.inference_mode()
    def _measure_incumbent(self, coords: torch.Tensor) -> BenchmarkResult:
        normalized = self.exec_state.normalized(self.target_qps, self.device, coords.shape[0])
        reference = _run_chunked(self.engine, coords, normalized, self.runtime_config.chunk_size)
        return self.autotuner.benchmark(
            self.runtime_config,
            coords,
            normalized,
            reference,
        )

    def run_self_optimization_cycle(
        self,
        cycles: int = 5,
        *,
        batch_size: int = 4,
        train_queries: int = 8_192,
        benchmark_queries: int = 32_768,
        tune_every: int = 2,
    ) -> None:
        print(
            f"Starting dual-loop optimization on {self.device.type.upper()} | "
            f"target={self.target_qps:,.0f} queries/s"
        )

        benchmark_coords = torch.randn(batch_size, benchmark_queries, 3, device=self.device)

        for cycle in range(1, cycles + 1):
            train_coords = torch.randn(batch_size, train_queries, 3, device=self.device)
            geometry_loss = self._train_step(train_coords)

            measured = self._measure_incumbent(benchmark_coords)
            if measured.accepted:
                self.exec_state = measured.metrics

            if tune_every > 0 and cycle % tune_every == 0:
                normalized = self.exec_state.normalized(
                    self.target_qps,
                    self.device,
                    benchmark_coords.shape[0],
                )
                tuned = self.autotuner.tune(
                    self.runtime_config,
                    benchmark_coords,
                    normalized,
                )
                if tuned.accepted:
                    self.runtime_config = tuned.config
                    self.exec_state = tuned.metrics
                    measured = tuned

            print(
                f"cycle={cycle:02d} "
                f"geometry_loss={geometry_loss:.6f} "
                f"throughput={measured.metrics.throughput_qps:,.0f} q/s "
                f"latency={measured.metrics.latency_ms:.3f} ms "
                f"peak_mem={measured.metrics.peak_memory_mb:.1f} MiB "
                f"chunk={self.runtime_config.chunk_size} "
                f"requested={self.runtime_config.compile_mode} "
                f"effective={measured.effective_mode} "
                f"semantic_error={measured.max_abs_error:.2e}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycles", type=int, default=5)
    parser.add_argument("--target-qps", type=float, default=1_000_000.0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--compile-mode",
        default="default",
        choices=("eager", "default", "reduce-overhead", "max-autotune"),
    )
    parser.add_argument("--chunk-size", type=int, default=65_536)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    engine = SelfReferentialTritonEngine()
    loop = SelfRefiningOptimizer(
        engine,
        target_qps=args.target_qps,
        device=args.device,
        runtime_config=RuntimeConfig(
            chunk_size=args.chunk_size,
            compile_mode=args.compile_mode,
        ),
    )
    loop.run_self_optimization_cycle(cycles=args.cycles)


if __name__ == "__main__":
    main()
