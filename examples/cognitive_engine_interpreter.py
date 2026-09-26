#!/usr/bin/env python3
"""
End-to-End Operational & Mathematical Python Interpreter for the Cognitive Engine
===================================================================================
A self-contained implementation of the 3D Auto-Encoding Cognitive Engine,
featuring harmonic carrier signals, latent bottleneck compression, real
multi-head self-attention over a rolling KV context, differential particle
kinematics on a Horn Torus manifold, and an interactive REPL shell.

Patched relative to v1:
  * warp_speed now drives the carrier clocks (phase accumulators + Nyquist cap)
  * multi-head attention over a real KV context (no more degenerate softmax)
  * normalized fidelity (1 - MSE/input_energy) instead of a floor at 88%
  * residual reconstruction path so the decoder can actually fit the input
  * collision-aware two-panel ASCII oscilloscope
  * configurable per-step angular velocity cap for the toroidal kinematics
"""

import math
import time
import sys
import random
import json
from collections import deque
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional, Deque

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

TWO_PI = 2.0 * math.pi


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class EngineConfig:
    """Hyperparameters and operational parameters for the Cognitive Engine."""
    input_dim: int = 16               # High-dimensional input state d
    bottleneck_dim: int = 16          # Compressed latent dimension k (k << d)
    num_heads: int = 4                # Multi-head attention head count (k must be divisible)
    context_size: int = 8             # Rolling KV context window size
    warp_speed: float = 1000.0        # Time warp multiplier Ms
    particle_count: int = 1200        # Number of token particles N
    f_encoder: float = 1.0            # 1.0 kHz Base Encoder clock
    f_latent: float = 0.5             # 0.5 kHz Sub-harmonic Bottleneck clock
    f_attn: float = 4.0               # 4.0 kHz Super-harmonic Attention clock
    decoder_phase_offset: float = 0.4 # Decoder phase lag (radians)
    dt_step: float = 0.001            # Wall-clock time per integration step (s)
    inward_boost: bool = True         # Enable inward singularity particle acceleration
    nyquist_safety: float = 0.25      # Samples per fastest clock cycle at max sim advance
    max_angular_velocity: float = 0.5 # rad/step cap for particle kinematics (stability)
    history_size: int = 200           # Rolling history buffer length


# ---------------------------------------------------------------------------
# Particle token
# ---------------------------------------------------------------------------
@dataclass
class ParticleToken:
    """Discrete token particle traveling along the toroidal manifold."""
    id: int
    theta: float
    phi: float
    speed_factor: float
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    domain: str = "Encoder"
    color_rgb: Tuple[int, int, int] = (56, 189, 248)


# ---------------------------------------------------------------------------
# Linear algebra (NumPy-accelerated when available)
# ---------------------------------------------------------------------------
class LinearAlgebraOps:
    """Matrix operations with a NumPy fast-path and pure-Python fallback."""

    @staticmethod
    def zeros(rows: int, cols: int) -> List[List[float]]:
        return [[0.0 for _ in range(cols)] for _ in range(rows)]

    @staticmethod
    def random_matrix(rows: int, cols: int, scale: float = 0.1) -> List[List[float]]:
        return [[(random.random() * 2.0 - 1.0) * scale for _ in range(cols)]
                for _ in range(rows)]

    @staticmethod
    def mat_vec_mul(matrix: List[List[float]], vector: List[float]) -> List[float]:
        if HAS_NUMPY:
            return (np.asarray(matrix, dtype=np.float64) @
                    np.asarray(vector, dtype=np.float64)).tolist()
        rows = len(matrix)
        cols = len(vector)
        out = [0.0] * rows
        for i in range(rows):
            row = matrix[i]
            s = 0.0
            for j in range(cols):
                s += row[j] * vector[j]
            out[i] = s
        return out

    @staticmethod
    def leaky_relu(vector: List[float], alpha: float = 0.01) -> List[float]:
        return [v if v > 0.0 else v * alpha for v in vector]

    @staticmethod
    def softmax(vector: List[float]) -> List[float]:
        if not vector:
            return []
        max_v = max(vector)
        exps = [math.exp(v - max_v) for v in vector]
        s = sum(exps)
        if s <= 0.0:
            return [1.0 / len(vector)] * len(vector)
        return [e / s for e in exps]

    @staticmethod
    def rms(vector: List[float]) -> float:
        if not vector:
            return 0.0
        return math.sqrt(sum(v * v for v in vector) / len(vector))

    @staticmethod
    def entropy(weights: List[float]) -> float:
        h = 0.0
        for w in weights:
            if w > 1e-12:
                h -= w * math.log(w)
        return h


# ---------------------------------------------------------------------------
# Harmonic carrier clocks — phase accumulators, warp-aware
# ---------------------------------------------------------------------------
class HarmonicCarrierClocks:
    """
    Phase-accumulator clock source. Each layer's carrier advances by
    2*pi*f*dt_sim per step, where dt_sim = min(dt_step * warp, safety/f_max).
    The Nyquist-style cap prevents visual aliasing at extreme warp values.
    """

    def __init__(self, config: EngineConfig):
        self.config = config
        self.reset()

    def reset(self):
        self.phase = {"enc": 0.0, "lat": 0.0, "attn": 0.0, "dec": 0.0}
        self._last_dt_sim = 0.0

    @property
    def fastest_clock_hz(self) -> float:
        return max(self.config.f_encoder, self.config.f_latent, self.config.f_attn)

    def effective_warp(self) -> float:
        """Real warp after Nyquist-safety clamping of dt_sim."""
        dt_ideal = self.config.dt_step * self.config.warp_speed
        dt_cap = self.config.nyquist_safety / self.fastest_clock_hz
        dt_sim = min(dt_ideal, dt_cap)
        return dt_sim / self.config.dt_step

    def advance(self, dt_wall: float) -> None:
        """Advance all clock phases by one integration step."""
        dt_ideal = dt_wall * self.config.warp_speed
        dt_cap = self.config.nyquist_safety / self.fastest_clock_hz
        dt_sim = min(dt_ideal, dt_cap)
        self._last_dt_sim = dt_sim

        f = self.config
        self.phase["enc"]  = (self.phase["enc"]  + TWO_PI * f.f_encoder * dt_sim) % TWO_PI
        self.phase["lat"]  = (self.phase["lat"]  + TWO_PI * f.f_latent  * dt_sim) % TWO_PI
        self.phase["attn"] = (self.phase["attn"] + TWO_PI * f.f_attn    * dt_sim) % TWO_PI
        self.phase["dec"]  = (self.phase["dec"]  + TWO_PI * f.f_encoder * dt_sim) % TWO_PI

    def evaluate(self) -> Dict[str, float]:
        off = self.config.decoder_phase_offset
        return {
            "S_enc":  math.sin(self.phase["enc"]),
            "S_lat":  math.sin(self.phase["lat"]),
            "S_attn": math.sin(self.phase["attn"]),
            "S_dec":  math.sin(self.phase["dec"] - off),
        }


# ---------------------------------------------------------------------------
# Multi-head attention with a rolling KV context
# ---------------------------------------------------------------------------
class MultiHeadAttention:
    """
    Scaled dot-product multi-head attention over a rolling context of latent
    vectors. The current latent is the query; the query plus its recent past
    forms the key/value set. Weights are shared across the H heads via row
    slicing, i.e. W_Q[h*d_head:(h+1)*d_head] is the projection for head h.
    """

    def __init__(self, k: int, num_heads: int, context_size: int):
        # Pick the largest divisor of k that is <= num_heads
        while num_heads > 1 and k % num_heads != 0:
            num_heads -= 1
        self.k = k
        self.h = num_heads
        self.d_head = k // num_heads
        self.ctx_size = context_size

        self.W_Q = LinearAlgebraOps.random_matrix(k, k, scale=0.3)
        self.W_K = LinearAlgebraOps.random_matrix(k, k, scale=0.3)
        self.W_V = LinearAlgebraOps.random_matrix(k, k, scale=0.3)
        self.W_O = LinearAlgebraOps.random_matrix(k, k, scale=0.3)

        self.buffer: Deque[List[float]] = deque(maxlen=context_size)
        self.last_weights: List[float] = []
        self.last_entropy: float = 0.0

    def forward(self, z: List[float], S_attn: float) -> Tuple[List[float], List[float]]:
        kv_set = list(self.buffer) + [z]
        if len(kv_set) < 2:
            self.buffer.append(z)
            self.last_weights = [1.0]
            self.last_entropy = 0.0
            return [v * S_attn for v in z], self.last_weights

        Q = LinearAlgebraOps.mat_vec_mul(self.W_Q, z)

        # Precompute K/V once, then slice per head
        K_all = [LinearAlgebraOps.mat_vec_mul(self.W_K, kv) for kv in kv_set]
        V_all = [LinearAlgebraOps.mat_vec_mul(self.W_V, kv) for kv in kv_set]

        head_outputs: List[float] = []
        per_head_weights: List[List[float]] = []

        for h in range(self.h):
            off = h * self.d_head
            q_h = Q[off:off + self.d_head]

            scores = []
            for K in K_all:
                k_h = K[off:off + self.d_head]
                s = 0.0
                for i in range(self.d_head):
                    s += q_h[i] * k_h[i]
                scores.append(s / math.sqrt(self.d_head))

            w = LinearAlgebraOps.softmax(scores)
            per_head_weights.append(w)

            head_out = [0.0] * self.d_head
            for n, wn in enumerate(w):
                v_h = V_all[n][off:off + self.d_head]
                for i in range(self.d_head):
                    head_out[i] += wn * v_h[i]
            head_outputs.extend(head_out)

        z_attn_raw = LinearAlgebraOps.mat_vec_mul(self.W_O, head_outputs)
        z_attn = [v * S_attn for v in z_attn_raw]

        # Aggregate for telemetry: mean weight per context slot across heads
        n_ctx = len(kv_set)
        mean_w = [sum(per_head_weights[h][n] for h in range(self.h)) / self.h
                  for n in range(n_ctx)]
        self.last_weights = mean_w
        self.last_entropy = LinearAlgebraOps.entropy(mean_w)

        self.buffer.append(z)
        return z_attn, mean_w


# ---------------------------------------------------------------------------
# Neural operator core
# ---------------------------------------------------------------------------
class NeuralOperatorCore:
    """Encoder → bottleneck → multi-head attention → residual decoder."""

    def __init__(self, config: EngineConfig):
        self.config = config
        self.d = config.input_dim
        self.k = config.bottleneck_dim

        # Encoder
        self.W_e = LinearAlgebraOps.random_matrix(self.d, self.d, scale=0.2)
        self.b_e = [0.05] * self.d

        # Bottleneck compression
        self.W_z = LinearAlgebraOps.random_matrix(self.k, self.d, scale=0.2)
        self.b_z = [0.01] * self.k

        # Attention
        self.attn = MultiHeadAttention(self.k, config.num_heads, config.context_size)

        # Decoder with residual head
        self.W_d = LinearAlgebraOps.random_matrix(self.d, self.k, scale=0.2)
        self.b_d = [0.02] * self.d

    def forward(self, x: List[float], clocks: Dict[str, float]) -> Dict[str, Any]:
        d, k = self.d, self.k

        # 1) Encoder
        raw_h = LinearAlgebraOps.mat_vec_mul(self.W_e, x)
        h_e = [raw_h[i] + self.b_e[i] for i in range(d)]
        h_e_act = LinearAlgebraOps.leaky_relu(h_e)
        h_e_mod = [v * clocks["S_enc"] for v in h_e_act]

        # 2) Latent bottleneck
        raw_z = LinearAlgebraOps.mat_vec_mul(self.W_z, h_e_mod)
        z_unact = [raw_z[i] + self.b_z[i] for i in range(k)]
        z_act = LinearAlgebraOps.leaky_relu(z_unact)
        z_lat = [v * clocks["S_lat"] for v in z_act]

        # 3) Multi-head self-attention over rolling context
        z_attn, attn_w = self.attn.forward(z_lat, clocks["S_attn"])

        # 4) Decoder with residual path on the bottleneck-projected signal
        raw_dec = LinearAlgebraOps.mat_vec_mul(self.W_d, z_attn)
        # Residual: project z_lat to input space and add, gated by decoder clock
        residual = LinearAlgebraOps.mat_vec_mul(self.W_d, z_lat)
        x_hat = [
            clocks["S_dec"] * (LinearAlgebraOps.leaky_relu([raw_dec[i] + self.b_d[i]])[0]
                               + 0.5 * residual[i])
            for i in range(d)
        ]

        # 5) Normalized reconstruction loss
        mse = sum((x[i] - x_hat[i]) ** 2 for i in range(d)) / d
        input_energy = sum(x[i] * x[i] for i in range(d)) / d + 1e-9
        nrmse = mse / input_energy
        fidelity = 100.0 * max(0.0, 1.0 - nrmse)

        # 6) Telemetry norms
        h_norm = LinearAlgebraOps.rms(h_e_mod)
        z_norm = LinearAlgebraOps.rms(z_lat)
        attn_norm = LinearAlgebraOps.rms(z_attn)

        return {
            "x_input": x,
            "h_encoder": h_e_mod,
            "z_latent": z_lat,
            "z_attn": z_attn,
            "attn_weights": attn_w,
            "attn_entropy": self.attn.last_entropy,
            "x_reconstructed": x_hat,
            "loss_mse": mse,
            "loss_nrmse": nrmse,
            "fidelity_pct": fidelity,
            "h_norm": h_norm,
            "z_norm": z_norm,
            "attn_norm": attn_norm,
        }


# ---------------------------------------------------------------------------
# Toroidal kinematics on a Horn Torus (R == r)
# ---------------------------------------------------------------------------
class ToroidalKinematicsSolver:
    """Coupled ODEs for particle tokens on the Horn Torus manifold."""

    def __init__(self, config: EngineConfig):
        self.config = config
        self.R = 3.2
        self.r = 3.2
        self.particles: List[ParticleToken] = []
        self._init_particles()

    def _init_particles(self):
        self.particles.clear()
        for i in range(self.config.particle_count):
            theta = random.random() * TWO_PI
            phi = random.random() * TWO_PI
            p = ParticleToken(id=i, theta=theta, phi=phi,
                              speed_factor=random.uniform(0.5, 1.3))
            self._update_particle_position(p)
            self.particles.append(p)

    def _update_particle_position(self, p: ParticleToken):
        inward = (1.0 - 0.3 * math.sin(p.phi)) if self.config.inward_boost else 1.0
        eff_R = (self.R + self.r * math.cos(p.phi)) * inward

        p.x = eff_R * math.cos(p.theta)
        p.z = eff_R * math.sin(p.theta)
        p.y = self.r * math.sin(p.phi) * (0.8 if self.config.inward_boost else 1.0)

        norm_phi = (p.phi / TWO_PI) % 1.0
        if norm_phi < 0.25:
            p.domain = "Encoder Outer Shell"
            p.color_rgb = (56, 189, 248)
        elif norm_phi < 0.50:
            p.domain = "Latent Bottleneck Gate"
            p.color_rgb = (16, 185, 129)
        elif norm_phi < 0.75:
            p.domain = "Recurrent Attention Core"
            p.color_rgb = (232, 121, 249)
        else:
            p.domain = "Decoder Axis Path"
            p.color_rgb = (245, 158, 11)

    def step_integration(self, dt_wall: float):
        """
        Euler-forward integration. Effective wall-time warp is applied to the
        angular rates, then capped per-step to keep the display coherent.
        """
        warp = self.config.warp_speed
        cap = self.config.max_angular_velocity

        for p in self.particles:
            boost = (1.0 + 2.0 * (math.sin(p.phi * 0.5) ** 2)
                     if self.config.inward_boost else 1.0)

            omega_theta = 0.4 * warp * p.speed_factor * boost * dt_wall
            omega_phi = 1.2 * warp * p.speed_factor * boost * dt_wall

            omega_theta = min(omega_theta, cap)
            omega_phi = min(omega_phi, cap)

            p.theta = (p.theta + omega_theta) % TWO_PI
            p.phi = (p.phi + omega_phi) % TWO_PI
            self._update_particle_position(p)


# ---------------------------------------------------------------------------
# ASCII terminal visualizer
# ---------------------------------------------------------------------------
class TerminalVisualizer:

    @staticmethod
    def _render_subplot(title: str, series: List[Tuple[str, Deque[float]]],
                        width: int, height: int,
                        value_range: Optional[Tuple[float, float]],
                        legend: str):
        print(f"\n=== {title} ===")
        canvas = [[" " for _ in range(width)] for _ in range(height)]
        mid = height // 2

        for char, data in series:
            recent = list(data)[-width:]
            if not recent:
                continue
            if value_range is not None:
                vmin, vmax = value_range
            else:
                vmin, vmax = min(recent), max(recent)
                if vmax - vmin < 1e-9:
                    vmax = vmin + 1.0

            span = vmax - vmin
            for x, v in enumerate(recent):
                norm = (v - vmin) / span * 2.0 - 1.0  # [-1, 1]
                y = mid - int(round(norm * (mid - 1)))
                y = max(0, min(height - 1, y))
                if canvas[y][x] == " ":
                    canvas[y][x] = char
                elif canvas[y][x] != char:
                    canvas[y][x] = "#"

        print("┌" + "─" * width + "┐")
        for row in canvas:
            print("│" + "".join(row) + "│")
        print("└" + "─" * width + "┘")
        print(legend)

    @staticmethod
    def render_oscilloscope(histories: Dict[str, Deque[float]], width: int = 72):
        TerminalVisualizer._render_subplot(
            "CARRIER CLOCKS (warp-modulated)",
            [("E", histories["S_enc"]),
             ("L", histories["S_lat"]),
             ("A", histories["S_attn"]),
             ("D", histories["S_dec"])],
            width=width, height=9, value_range=(-1.1, 1.1),
            legend="E=Encoder 1.0kHz | L=Latent 0.5kHz | A=Attention 4.0kHz | D=Decoder"
        )

        TerminalVisualizer._render_subplot(
            "ACTIVITY NORMS  (auto-scaled)",
            [("h", histories["h_norm"]),
             ("z", histories["z_norm"]),
             ("a", histories["attn_norm"]),
             ("*", histories["loss"])],
            width=width, height=7, value_range=None,
            legend="h=‖h_encoder‖_rms | z=‖z_latent‖_rms | a=‖z_attn‖_rms | *=MSE loss"
        )

    @staticmethod
    def render_ascii_torus_projection(particles: List[ParticleToken], grid_size: int = 21):
        print("\n=== 3D HORN TORUS SINGULARITY PROJECTION (XZ TOP-DOWN VIEW) ===")
        grid = [[" " for _ in range(grid_size)] for _ in range(grid_size)]
        center = grid_size // 2
        scale = center / 6.5

        stride = max(1, len(particles) // 300)
        for p in particles[::stride]:
            gx = int(round(center + p.x * scale))
            gz = int(round(center + p.z * scale))
            if 0 <= gx < grid_size and 0 <= gz < grid_size:
                grid[gz][gx] = p.domain[0]

        grid[center][center] = "O"
        print("┌" + "─" * grid_size + "┐")
        for row in grid:
            print("│" + "".join(row) + "│")
        print("└" + "─" * grid_size + "┘")
        print("Legend: E=Encoder | L=Latent | A=Attention | D=Decoder | O=Origin")


# ---------------------------------------------------------------------------
# Cognitive Engine Interpreter — orchestrates everything
# ---------------------------------------------------------------------------
class CognitiveEngineInterpreter:

    def __init__(self, config: Optional[EngineConfig] = None):
        self.config = config or EngineConfig()
        self.clocks = HarmonicCarrierClocks(self.config)
        self.core = NeuralOperatorCore(self.config)
        self.kinematics = ToroidalKinematicsSolver(self.config)

        self.sim_time = 0.0
        self.step_count = 0
        self.history: Dict[str, Deque[float]] = {
            "S_enc": deque(maxlen=self.config.history_size),
            "S_lat": deque(maxlen=self.config.history_size),
            "S_attn": deque(maxlen=self.config.history_size),
            "S_dec": deque(maxlen=self.config.history_size),
            "h_norm": deque(maxlen=self.config.history_size),
            "z_norm": deque(maxlen=self.config.history_size),
            "attn_norm": deque(maxlen=self.config.history_size),
            "loss": deque(maxlen=self.config.history_size),
            "fidelity": deque(maxlen=self.config.history_size),
        }

    # -------- configuration mutators --------
    def set_bottleneck_dimension(self, new_dim: int):
        if new_dim < 1:
            print(f"[!] Bottleneck dimension must be positive; got {new_dim}.")
            return
        self.config.bottleneck_dim = new_dim
        self.core = NeuralOperatorCore(self.config)  # rebuilds attention too
        print(f"[✓] Compression dimension updated to k = {new_dim}D. "
              f"(Attention context reset; head count = {self.core.attn.h}.)")

    def effective_warp(self) -> float:
        return self.clocks.effective_warp()

    # -------- core step --------
    def step(self, input_vector: Optional[List[float]] = None) -> Dict[str, Any]:
        dt = self.config.dt_step

        # Advance the phase-accumulator clocks (this consumes warp_speed)
        self.clocks.advance(dt)
        clock_signals = self.clocks.evaluate()

        # Structured two-tone synthetic input — decoupled from warp so the
        # signal is legible in the oscilloscope at any warp setting.
        if input_vector is None:
            t = self.step_count * 0.1
            input_vector = [
                math.sin(t + i * 0.3) + 0.5 * math.sin(2.3 * t + i * 0.7)
                for i in range(self.config.input_dim)
            ]

        # Forward pass
        res = self.core.forward(input_vector, clock_signals)

        # Particle kinematics
        self.kinematics.step_integration(dt)

        self.sim_time += dt
        self.step_count += 1

        # Telemetry
        self.history["S_enc"].append(clock_signals["S_enc"])
        self.history["S_lat"].append(clock_signals["S_lat"])
        self.history["S_attn"].append(clock_signals["S_attn"])
        self.history["S_dec"].append(clock_signals["S_dec"])
        self.history["h_norm"].append(res["h_norm"])
        self.history["z_norm"].append(res["z_norm"])
        self.history["attn_norm"].append(res["attn_norm"])
        self.history["loss"].append(res["loss_mse"])
        self.history["fidelity"].append(res["fidelity_pct"])

        return res

    def run_sequence(self, steps: int = 100) -> Dict[str, Any]:
        start = time.time()
        last = {}
        for _ in range(steps):
            last = self.step()
        elapsed = time.time() - start
        return {
            "completed_steps": steps,
            "sim_time_sec": self.sim_time,
            "elapsed_wall_sec": elapsed,
            "steps_per_sec": steps / max(1e-6, elapsed),
            "final_loss": last.get("loss_mse", 0.0),
            "final_nrmse": last.get("loss_nrmse", 0.0),
            "final_fidelity": last.get("fidelity_pct", 0.0),
            "attn_entropy": last.get("attn_entropy", 0.0),
        }

    def export_telemetry_json(self) -> str:
        latest = {k: (v[-1] if v else None) for k, v in self.history.items()}
        telemetry = {
            "config": {
                "input_dim": self.config.input_dim,
                "bottleneck_dim": self.config.bottleneck_dim,
                "num_heads": self.core.attn.h,
                "context_size": self.config.context_size,
                "warp_speed_requested": self.config.warp_speed,
                "warp_speed_effective": self.effective_warp(),
                "particle_count": self.config.particle_count,
                "inward_boost": self.config.inward_boost,
            },
            "runtime": {
                "sim_time": self.sim_time,
                "step_count": self.step_count,
                "latest": latest,
            },
            "clocks": self.clocks.evaluate(),
        }
        return json.dumps(telemetry, indent=2)


# ---------------------------------------------------------------------------
# Interactive REPL
# ---------------------------------------------------------------------------
HELP_TEXT = """
AVAILABLE COMMANDS:
  step [n]         - Run n simulation steps (default n=1)
  warp [speed]     - Set requested warp multiplier (effective warp is
                     Nyquist-limited and reported back)
  dim [k]          - Set bottleneck dimension k (rebuilds attention)
  scope            - Render ASCII oscilloscope (carrier + activity panels)
  torus            - Render Horn Torus ASCII particle projection
  eval v1,v2,...   - Pass a custom vector into the encoder
  json             - Export telemetry as JSON
  plot             - Open Matplotlib 3D window (if installed)
  help             - Show this message
  exit / quit      - Terminate
"""


def run_interactive_shell(engine: CognitiveEngineInterpreter):
    print("=" * 80)
    print("      3D AUTO-ENCODING COGNITIVE ENGINE INTERPRETER (PYTHON REPL)")
    print("      Warp-Scaled Harmonic Clocks • Multi-Head Attention • Horn Torus")
    print("=" * 80)
    print("Type 'help' for available commands or 'exit' to quit.\n")

    while True:
        try:
            raw = input("CognitiveEngine> ").strip()
            if not raw:
                continue
            parts = raw.split()
            cmd = parts[0].lower()

            if cmd in ("exit", "quit"):
                print("Exiting Cognitive Engine Interpreter. System shutdown clean.")
                return

            elif cmd == "help":
                print(HELP_TEXT)

            elif cmd == "step":
                n = int(parts[1]) if len(parts) > 1 else 1
                res = engine.run_sequence(n)
                print(f"[✓] Advanced {n} steps | Sim t={res['sim_time_sec']:.3f}s | "
                      f"MSE={res['final_loss']:.5f} | NRMSE={res['final_nrmse']:.4f} | "
                      f"Fidelity={res['final_fidelity']:.2f}% | "
                      f"H(attn)={res['attn_entropy']:.3f}")

            elif cmd == "warp":
                if len(parts) > 1:
                    w = max(1.0, float(parts[1]))
                    engine.config.warp_speed = w
                    eff = engine.effective_warp()
                    print(f"[✓] Warp requested = {w:.0f}x | effective = {eff:.2f}x "
                          f"(Nyquist-limited by f_attn={engine.config.f_attn} kHz)")
                else:
                    print(f"Requested warp: {engine.config.warp_speed:.0f}x | "
                          f"effective: {engine.effective_warp():.2f}x")

            elif cmd == "dim":
                if len(parts) > 1:
                    engine.set_bottleneck_dimension(int(parts[1]))
                else:
                    print(f"Current k = {engine.config.bottleneck_dim}D, "
                          f"heads = {engine.core.attn.h}, "
                          f"context = {engine.config.context_size}")

            elif cmd == "scope":
                TerminalVisualizer.render_oscilloscope(engine.history)

            elif cmd == "torus":
                TerminalVisualizer.render_ascii_torus_projection(engine.kinematics.particles)

            elif cmd == "json":
                print(engine.export_telemetry_json())

            elif cmd == "eval":
                if len(parts) > 1:
                    vals = [float(v) for v in parts[1].split(",")]
                    if len(vals) < engine.config.input_dim:
                        vals += [0.0] * (engine.config.input_dim - len(vals))
                    else:
                        vals = vals[:engine.config.input_dim]
                    res = engine.step(vals)
                    print("[✓] Forward Pass Complete:")
                    print(f"    MSE      : {res['loss_mse']:.6f}")
                    print(f"    NRMSE    : {res['loss_nrmse']:.6f}")
                    print(f"    Fidelity : {res['fidelity_pct']:.2f}%")
                    print(f"    H(attn)  : {res['attn_entropy']:.4f}")
                    print(f"    x̂[:4]    : {[round(v, 4) for v in res['x_reconstructed'][:4]]}")

            elif cmd == "plot":
                if HAS_MATPLOTLIB:
                    render_matplotlib_3d(engine)
                else:
                    print("[!] Matplotlib is not installed.")

            else:
                print(f"[!] Unknown command '{cmd}'. Type 'help'.")

        except Exception as err:
            print(f"[!] Error evaluating command: {err}")


# ---------------------------------------------------------------------------
# Matplotlib 3D visualisation (optional)
# ---------------------------------------------------------------------------
def render_matplotlib_3d(engine: CognitiveEngineInterpreter):
    if not HAS_MATPLOTLIB:
        return
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#030712")
    fig.patch.set_facecolor("#030712")

    particles = engine.kinematics.particles
    xs = [p.x for p in particles]
    ys = [p.y for p in particles]
    zs = [p.z for p in particles]
    colors = [f"#{p.color_rgb[0]:02x}{p.color_rgb[1]:02x}{p.color_rgb[2]:02x}"
              for p in particles]

    ax.scatter(xs, zs, ys, c=colors, s=12, alpha=0.8, edgecolors="none")
    ax.set_title(
        f"3D Cognitive Engine Manifold "
        f"(k={engine.config.bottleneck_dim}D, "
        f"h={engine.core.attn.h}, "
        f"{engine.config.warp_speed:.0f}x requested / "
        f"{engine.effective_warp():.1f}x effective)",
        color="#38bdf8", fontsize=12,
    )
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    print("[*] Initializing 3D Auto-Encoding Cognitive Engine Python Core...")
    config = EngineConfig(
        input_dim=16,
        bottleneck_dim=16,
        num_heads=4,
        context_size=8,
        warp_speed=1000.0,
        particle_count=1200,
    )
    engine = CognitiveEngineInterpreter(config)

    print("[*] Running initial 100-step simulation benchmark...")
    bench = engine.run_sequence(100)
    print(f"[✓] Benchmark: {bench['steps_per_sec']:.1f} steps/sec | "
          f"MSE={bench['final_loss']:.5f} | NRMSE={bench['final_nrmse']:.4f} | "
          f"Fidelity={bench['final_fidelity']:.2f}% | "
          f"H(attn)={bench['attn_entropy']:.3f}")
    print(f"[✓] Effective warp: {engine.effective_warp():.2f}x "
          f"(requested {config.warp_speed:.0f}x)\n")

    run_interactive_shell(engine)


if __name__ == "__main__":
    main()
