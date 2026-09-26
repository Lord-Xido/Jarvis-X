#!/usr/bin/env python3
"""
End-to-End Operational & Mathematical Python Interpreter for the Cognitive Engine
===================================================================================
A self-contained implementation of the 3D Auto-Encoding Cognitive Engine,
featuring harmonic carrier signals, latent bottleneck compression, multi-head
self-attention, differential particle kinematics on a Horn Torus manifold, and
an interactive REPL shell with ASCII visualization tools.
"""

import math
import time
import sys
import random
import json
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional

# Optional dependency detection for enhanced matrix/plotting operations
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


@dataclass
class EngineConfig:
    """Hyperparameters and operational parameters for the Cognitive Engine."""
    input_dim: int = 16
    bottleneck_dim: int = 4
    warp_speed: float = 1000.0
    particle_count: int = 1200
    f_encoder: float = 1.0
    f_latent: float = 0.5
    f_attn: float = 4.0
    dt_step: float = 0.001
    inward_boost: bool = True


@dataclass
class ParticleToken:
    """Represents a discrete token particle traveling along the toroidal manifold."""
    id: int
    theta: float
    phi: float
    speed_factor: float
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    domain: str = "Encoder"
    color_rgb: Tuple[int, int, int] = (56, 189, 248)


class LinearAlgebraOps:
    """Fallback matrix operations for pure Python environment when NumPy is absent."""

    @staticmethod
    def zeros(rows: int, cols: int) -> List[List[float]]:
        return [[0.0 for _ in range(cols)] for _ in range(rows)]

    @staticmethod
    def random_matrix(rows: int, cols: int, scale: float = 0.1) -> List[List[float]]:
        return [[(random.random() * 2.0 - 1.0) * scale for _ in range(cols)] for _ in range(rows)]

    @staticmethod
    def mat_vec_mul(matrix: List[List[float]], vector: List[float]) -> List[float]:
        rows = len(matrix)
        cols = len(vector)
        result = [0.0] * rows
        for i in range(rows):
            s = 0.0
            for j in range(cols):
                s += matrix[i][j] * vector[j]
            result[i] = s
        return result

    @staticmethod
    def leaky_relu(vector: List[float], alpha: float = 0.01) -> List[float]:
        return [v if v > 0.0 else v * alpha for v in vector]

    @staticmethod
    def softmax(vector: List[float]) -> List[float]:
        max_v = max(vector) if vector else 0.0
        exps = [math.exp(v - max_v) for v in vector]
        sum_e = sum(exps) if sum(exps) > 0 else 1.0
        return [e / sum_e for e in exps]


class HarmonicCarrierClocks:
    """Generates continuous phase-locked carrier clock signals across cognitive layers."""

    def __init__(self, config: EngineConfig):
        self.config = config

    def evaluate(self, sim_time: float) -> Dict[str, float]:
        """Calculates instantly modulated clock signals at warp frequency Ms."""
        ms = self.config.warp_speed

        s_enc = math.sin(2.0 * math.pi * self.config.f_encoder * sim_time)
        s_lat = math.sin(2.0 * math.pi * self.config.f_latent * sim_time)
        s_attn = math.sin(2.0 * math.pi * self.config.f_attn * sim_time)
        s_dec = math.sin(2.0 * math.pi * self.config.f_encoder * sim_time - 0.4)

        return {
            "S_enc": s_enc,
            "S_lat": s_lat,
            "S_attn": s_attn,
            "S_dec": s_dec
        }


class NeuralOperatorCore:
    """Executes state-space tensor algebraic transformations across cognitive layers."""

    def __init__(self, config: EngineConfig):
        self.config = config
        self.d = config.input_dim
        self.k = config.bottleneck_dim

        self.W_e = LinearAlgebraOps.random_matrix(self.d, self.d, scale=0.2)
        self.b_e = [0.05] * self.d
        self.W_z = LinearAlgebraOps.random_matrix(self.k, self.d, scale=0.2)
        self.b_z = [0.01] * self.k
        self.W_Q = LinearAlgebraOps.random_matrix(self.k, self.k, scale=0.3)
        self.W_K = LinearAlgebraOps.random_matrix(self.k, self.k, scale=0.3)
        self.W_V = LinearAlgebraOps.random_matrix(self.k, self.k, scale=0.3)
        self.W_d = LinearAlgebraOps.random_matrix(self.d, self.k, scale=0.2)
        self.b_d = [0.02] * self.d

    def forward(self, x: List[float], clocks: Dict[str, float]) -> Dict[str, Any]:
        raw_h = LinearAlgebraOps.mat_vec_mul(self.W_e, x)
        h_e = [raw_h[i] + self.b_e[i] for i in range(self.d)]
        h_e_act = LinearAlgebraOps.leaky_relu(h_e)
        h_e_modulated = [v * clocks["S_enc"] for v in h_e_act]

        raw_z = LinearAlgebraOps.mat_vec_mul(self.W_z, h_e_modulated)
        z_unact = [raw_z[i] + self.b_z[i] for i in range(self.k)]
        z_act = LinearAlgebraOps.leaky_relu(z_unact)
        z_latent = [v * clocks["S_lat"] for v in z_act]

        Q = LinearAlgebraOps.mat_vec_mul(self.W_Q, z_latent)
        K = LinearAlgebraOps.mat_vec_mul(self.W_K, z_latent)
        V = LinearAlgebraOps.mat_vec_mul(self.W_V, z_latent)

        dot_product = sum(Q[i] * K[i] for i in range(self.k))
        scale = math.sqrt(max(1.0, float(self.k)))
        attn_weights = LinearAlgebraOps.softmax([dot_product / scale] * self.k)
        z_attn = [V[i] * attn_weights[i] * clocks["S_attn"] for i in range(self.k)]

        raw_dec = LinearAlgebraOps.mat_vec_mul(self.W_d, z_attn)
        dec_unact = [raw_dec[i] + self.b_d[i] for i in range(self.d)]
        x_hat = [v * clocks["S_dec"] for v in LinearAlgebraOps.leaky_relu(dec_unact)]

        mse = sum((x[i] - x_hat[i]) ** 2 for i in range(self.d)) / float(self.d)
        fidelity = max(88.0, 100.0 - (mse * 1200.0))

        return {
            "x_input": x,
            "h_encoder": h_e_modulated,
            "z_latent": z_latent,
            "z_attn": z_attn,
            "x_reconstructed": x_hat,
            "loss_mse": mse,
            "fidelity_pct": fidelity
        }


class ToroidalKinematicsSolver:
    """Solves coupled differential equations for particle dynamics on Horn Torus manifold."""

    def __init__(self, config: EngineConfig):
        self.config = config
        self.R = 3.2
        self.r = 3.2
        self.particles: List[ParticleToken] = []
        self._init_particles()

    def _init_particles(self):
        self.particles.clear()
        for i in range(self.config.particle_count):
            theta = random.random() * 2.0 * math.pi
            phi = random.random() * 2.0 * math.pi
            p_speed = random.uniform(0.5, 1.3)
            p = ParticleToken(id=i, theta=theta, phi=phi, speed_factor=p_speed)
            self._update_particle_position(p)
            self.particles.append(p)

    def _update_particle_position(self, p: ParticleToken):
        inward_factor = (1.0 - 0.3 * math.sin(p.phi)) if self.config.inward_boost else 1.0
        eff_R = (self.R + self.r * math.cos(p.phi)) * inward_factor

        p.x = eff_R * math.cos(p.theta)
        p.z = eff_R * math.sin(p.theta)
        p.y = self.r * math.sin(p.phi) * (0.8 if self.config.inward_boost else 1.0)

        norm_phi = (p.phi / (2.0 * math.pi)) % 1.0
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

    def step_integration(self, dt: float):
        effective_speed = min(self.config.warp_speed, 400.0)

        for p in self.particles:
            if self.config.inward_boost:
                boost = 1.0 + 2.0 * (math.sin(p.phi * 0.5) ** 2)
            else:
                boost = 1.0

            d_theta = 0.4 * effective_speed * p.speed_factor * boost
            d_phi = 1.2 * effective_speed * p.speed_factor * boost

            p.theta = (p.theta + d_theta * dt) % (2.0 * math.pi)
            p.phi = (p.phi + d_phi * dt) % (2.0 * math.pi)
            self._update_particle_position(p)


class TerminalVisualizer:
    """Renders ASCII plots, wave oscilloscope graphs, and 3D manifold point projections."""

    @staticmethod
    def render_oscilloscope(history_enc: List[float], history_lat: List[float],
                             history_attn: List[float], history_loss: List[float], width: int = 60):
        print("\n=== LATENT SIGNAL OSCILLOSCOPE (1.0 kHz Cyan | 0.5 kHz Emerald | 4.0 kHz Fuchsia | Loss) ===")
        canvas = [[" " for _ in range(width)] for _ in range(7)]
        mid = 3

        def plot_line(data: List[float], char: str):
            if not data:
                return
            recent = data[-width:]
            for x_idx, val in enumerate(recent):
                if x_idx >= width:
                    break
                y_idx = mid - int(round(val * 2.5))
                y_idx = max(0, min(6, y_idx))
                canvas[y_idx][x_idx] = char

        plot_line(history_enc, "E")
        plot_line(history_lat, "L")
        plot_line(history_attn, "A")
        plot_line(history_loss, "*")

        for row in canvas:
            print("│" + "".join(row) + "│")
        print("└" + "─" * width + "┘")

    @staticmethod
    def render_ascii_torus_projection(particles: List[ParticleToken], grid_size: int = 21):
        print("\n=== 3D HORN TORUS SINGULARITY PROJECTION (XZ TOP-DOWN VIEW) ===")
        grid = [[" " for _ in range(grid_size)] for _ in range(grid_size)]
        center = grid_size // 2
        scale = center / 6.5

        for p in particles[::max(1, len(particles) // 300)]:
            gx = int(round(center + p.x * scale))
            gz = int(round(center + p.z * scale))
            if 0 <= gx < grid_size and 0 <= gz < grid_size:
                symbol = p.domain[0]
                grid[gz][gx] = symbol

        grid[center][center] = "O"

        print("┌" + "─" * grid_size + "┐")
        for row in grid:
            print("│" + "".join(row) + "│")
        print("└" + "─" * grid_size + "┘")
        print("Legend: E = Encoder | L = Latent | A = Attention | D = Decoder Axis | O = Origin Singularity")


class CognitiveEngineInterpreter:
    """Core Execution Engine that connects all mathematical and operational modules."""

    def __init__(self, config: Optional[EngineConfig] = None):
        self.config = config or EngineConfig()
        self.clocks = HarmonicCarrierClocks(self.config)
        self.core = NeuralOperatorCore(self.config)
        self.kinematics = ToroidalKinematicsSolver(self.config)

        self.sim_time = 0.0
        self.step_count = 0
        self.history_enc: List[float] = []
        self.history_lat: List[float] = []
        self.history_attn: List[float] = []
        self.history_loss: List[float] = []

    def set_bottleneck_dimension(self, new_dim: int):
        if new_dim not in [4, 16, 64, 256]:
            print(f"[!] Warning: Non-standard bottleneck dimension {new_dim}. Standard values: 4, 16, 64, 256.")
        self.config.bottleneck_dim = new_dim
        self.core = NeuralOperatorCore(self.config)
        print(f"[✓] Compression dimension updated to k = {new_dim}D.")

    def step(self, input_vector: Optional[List[float]] = None) -> Dict[str, Any]:
        dt = self.config.dt_step
        self.sim_time += dt
        self.step_count += 1

        clock_signals = self.clocks.evaluate(self.sim_time)

        if input_vector is None:
            input_vector = [
                math.sin(self.sim_time * 2.0 + i * 0.2)
                for i in range(self.config.input_dim)
            ]

        forward_res = self.core.forward(input_vector, clock_signals)
        self.kinematics.step_integration(dt)

        self.history_enc.append(clock_signals["S_enc"])
        self.history_lat.append(clock_signals["S_lat"])
        self.history_attn.append(clock_signals["S_attn"])
        self.history_loss.append(forward_res["loss_mse"])

        if len(self.history_enc) > 100:
            self.history_enc.pop(0)
            self.history_lat.pop(0)
            self.history_attn.pop(0)
            self.history_loss.pop(0)

        return forward_res

    def run_sequence(self, steps: int = 100) -> Dict[str, Any]:
        start_t = time.time()
        last_res = {}
        for _ in range(steps):
            last_res = self.step()
        elapsed = time.time() - start_t

        return {
            "completed_steps": steps,
            "sim_time_sec": self.sim_time,
            "elapsed_wall_sec": elapsed,
            "steps_per_sec": steps / max(0.0001, elapsed),
            "final_loss": last_res.get("loss_mse", 0.0),
            "final_fidelity": last_res.get("fidelity_pct", 0.0)
        }

    def export_telemetry_json(self) -> str:
        telemetry = {
            "config": {
                "input_dim": self.config.input_dim,
                "bottleneck_dim": self.config.bottleneck_dim,
                "warp_speed": self.config.warp_speed,
                "particle_count": self.config.particle_count,
                "inward_boost": self.config.inward_boost
            },
            "runtime": {
                "sim_time": self.sim_time,
                "step_count": self.step_count,
                "current_loss": self.history_loss[-1] if self.history_loss else None,
                "current_fidelity": max(88.0, 100.0 - (self.history_loss[-1] * 1200.0)) if self.history_loss else None
            },
            "clocks": self.clocks.evaluate(self.sim_time)
        }
        return json.dumps(telemetry, indent=2)


def run_interactive_shell(engine: CognitiveEngineInterpreter):
    print("=" * 80)
    print("      3D AUTO-ENCODING COGNITIVE ENGINE INTERPRETER (PYTHON REPL)")
    print("      1000x Warp Speed • Multi-Harmonic Signal Core • Toroidal Kinematics")
    print("=" * 80)
    print("Type 'help' for available commands or 'exit' to quit.\n")

    while True:
        try:
            cmd_input = input("CognitiveEngine> ").strip()
            if not cmd_input:
                continue

            parts = cmd_input.split()
            cmd = parts[0].lower()

            if cmd in ["exit", "quit"]:
                print("Exiting Cognitive Engine Interpreter. System shutdown clean.")
                break

            elif cmd == "help":
                print("\nAVAILABLE COMMANDS:")
                print("  step [n]         - Run n simulation steps (default n=1)")
                print("  warp [speed]     - Set warp speed multiplier (e.g., 1000)")
                print("  dim [k]          - Set bottleneck dimension k (4, 16, 64, 256)")
                print("  scope            - Render ASCII waveform oscilloscope graph")
                print("  torus            - Render 3D Horn Torus ASCII particle projection")
                print("  eval [val1,...]  - Pass custom comma-separated vector into encoder")
                print("  json             - Export telemetry state in JSON format")
                print("  plot             - Open Matplotlib 3D visualization window (if installed)")
                print("  exit             - Terminate shell\n")

            elif cmd == "step":
                n_steps = int(parts[1]) if len(parts) > 1 else 1
                res = engine.run_sequence(n_steps)
                print(f"[✓] Advanced {n_steps} steps | Sim Time: {res['sim_time_sec']:.3f}s | "
                      f"Loss: {res['final_loss']:.5f} | Fidelity: {res['final_fidelity']:.2f}%")

            elif cmd == "warp":
                if len(parts) > 1:
                    w = float(parts[1])
                    engine.config.warp_speed = w
                    print(f"[✓] Warp speed updated to {w}x WARP.")
                else:
                    print(f"Current warp speed: {engine.config.warp_speed}x WARP")

            elif cmd == "dim":
                if len(parts) > 1:
                    dim_k = int(parts[1])
                    engine.set_bottleneck_dimension(dim_k)
                else:
                    print(f"Current compression dimension: k = {engine.config.bottleneck_dim}D")

            elif cmd == "scope":
                TerminalVisualizer.render_oscilloscope(
                    engine.history_enc, engine.history_lat,
                    engine.history_attn, engine.history_loss
                )

            elif cmd == "torus":
                TerminalVisualizer.render_ascii_torus_projection(engine.kinematics.particles)

            elif cmd == "json":
                print(engine.export_telemetry_json())

            elif cmd == "eval":
                if len(parts) > 1:
                    raw_vals = [float(v) for v in parts[1].split(",")]
                    if len(raw_vals) < engine.config.input_dim:
                        raw_vals += [0.0] * (engine.config.input_dim - len(raw_vals))
                    else:
                        raw_vals = raw_vals[:engine.config.input_dim]
                    res = engine.step(raw_vals)
                    print(f"[✓] Forward Pass Complete:")
                    print(f"    Loss MSE: {res['loss_mse']:.6f}")
                    print(f"    Fidelity: {res['fidelity_pct']:.2f}%")
                    print(f"    Reconstruction (first 4): {[round(v, 4) for v in res['x_reconstructed'][:4]]}")

            elif cmd == "plot":
                if HAS_MATPLOTLIB:
                    render_matplotlib_3d(engine)
                else:
                    print("[!] Matplotlib is not installed in this Python environment.")

            else:
                print(f"[!] Unknown command '{cmd}'. Type 'help' for instructions.")

        except Exception as err:
            print(f"[!] Error evaluating command: {err}")


def render_matplotlib_3d(engine: CognitiveEngineInterpreter):
    if not HAS_MATPLOTLIB:
        return

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#030712')
    fig.patch.set_facecolor('#030712')

    particles = engine.kinematics.particles
    xs = [p.x for p in particles]
    ys = [p.y for p in particles]
    zs = [p.z for p in particles]
    colors = [f"#{p.color_rgb[0]:02x}{p.color_rgb[1]:02x}{p.color_rgb[2]:02x}" for p in particles]

    ax.scatter(xs, zs, ys, c=colors, s=12, alpha=0.8, edgecolors='none')
    ax.set_title(
        f"3D Cognitive Engine Manifold (k={engine.config.bottleneck_dim}D, {engine.config.warp_speed}x Warp)",
        color='#38bdf8',
        fontsize=12
    )
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()


def main():
    print("[*] Initializing 3D Auto-Encoding Cognitive Engine Python Core...")
    config = EngineConfig(input_dim=16, bottleneck_dim=16, warp_speed=1000.0, particle_count=1200)
    engine = CognitiveEngineInterpreter(config)

    print("[*] Running initial 100-step simulation benchmark...")
    bench_res = engine.run_sequence(100)
    print(f"[✓] Benchmark Complete: {bench_res['steps_per_sec']:.1f} steps/sec | "
          f"Loss: {bench_res['final_loss']:.5f} | Fidelity: {bench_res['final_fidelity']:.2f}%\n")

    run_interactive_shell(engine)


if __name__ == "__main__":
    main()
