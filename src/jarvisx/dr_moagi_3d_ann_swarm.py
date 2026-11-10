#!/usr/bin/env python3
"""Bounded self-optimising 3D micro-autoencoder swarm (reference implementation).

Virtual 10**6 x 10**6 x 10**6 addressing: materialises ONLY n_particles.
Each particle has its own trainable 7 -> hidden -> latent -> hidden -> 7 ANN.
The text->hash embedding is one-way; reconstructed 7-vectors are NOT decoded text.
A provably contractive *position* step is separate from ANN SGD/meta-pruning.
No exponential runtime gain or full-system fixed-point proof is asserted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from typing import Any, Dict, List, Optional

import numpy as np

VIRTUAL_SIDE = 1_000_000


def _digest_position_and_features(chunk: str) -> tuple[np.ndarray, np.ndarray]:
    """Decode SHA256 via uint32 integers, never as arbitrary IEEE floats."""
    digest = hashlib.sha256(chunk.encode("utf-8")).digest()
    ints = np.frombuffer(digest[:28], dtype="<u4").astype(np.float64)
    position = ((ints[:3] + 0.5) / 2**32).astype(np.float32)
    features = ((ints[3:7] + 0.5) / 2**31 - 1).astype(np.float32)
    return position, features


class NeuralParticle:
    """Seven-input per-particle MLP with actual SGD and a latent gating mask."""

    def __init__(self, particle_id: int, position: np.ndarray, features: np.ndarray,
                 rng: np.random.Generator, latent_dim: int = 16, hidden: int = 32):
        self.id = int(particle_id)
        self.pos = np.asarray(position, dtype=np.float32).copy()
        self.state = np.zeros(7, dtype=np.float32)
        self.state[3:] = features
        self.latent_dim = latent_dim
        self.W_enc1 = (rng.standard_normal((7, hidden)) * 0.10).astype(np.float32)
        self.b_enc1 = np.zeros(hidden, dtype=np.float32)
        self.W_enc2 = (rng.standard_normal((hidden, latent_dim)) * 0.10).astype(np.float32)
        self.b_enc2 = np.zeros(latent_dim, dtype=np.float32)
        self.W_dec1 = (rng.standard_normal((latent_dim, hidden)) * 0.10).astype(np.float32)
        self.b_dec1 = np.zeros(hidden, dtype=np.float32)
        self.W_dec2 = (rng.standard_normal((hidden, 7)) * 0.10).astype(np.float32)
        self.b_dec2 = np.zeros(7, dtype=np.float32)
        self.mask = np.ones(latent_dim, dtype=np.float32)
        self.latent = np.zeros(latent_dim, dtype=np.float32)
        self.raw_latent = self.latent.copy()
        self.last_error = math.inf

    def _forward(self, x: np.ndarray):
        a = np.tanh(x @ self.W_enc1 + self.b_enc1)
        raw = np.tanh(a @ self.W_enc2 + self.b_enc2)
        z = raw * self.mask
        b = np.tanh(z @ self.W_dec1 + self.b_dec1)
        y = b @ self.W_dec2 + self.b_dec2
        return a, raw, z, b, y

    def encode(self, features: np.ndarray) -> np.ndarray:
        x = np.concatenate((self.pos, np.asarray(features, dtype=np.float32)))
        _, raw, self.latent, _, _ = self._forward(x)
        self.raw_latent = raw
        return self.latent.copy()

    def decode(self) -> np.ndarray:
        h = np.tanh(self.latent @ self.W_dec1 + self.b_dec1)
        return h @ self.W_dec2 + self.b_dec2

    def train_step(self, learning_rate: float) -> float:
        """Full manual backpropagation with a per-tensor gradient norm cap."""
        x = np.concatenate((self.pos, self.state[3:]))
        a, raw, z, b, y = self._forward(x)
        grad_y = (2.0 / len(x)) * (y - x)
        gd2 = np.outer(b, grad_y)
        gb2 = grad_y.copy()
        grad_b = (self.W_dec2 @ grad_y) * (1.0 - b * b)
        gd1 = np.outer(z, grad_b)
        gb1 = grad_b.copy()
        grad_z = ((self.W_dec1 @ grad_b) * self.mask) * (1.0 - raw * raw)
        ge2 = np.outer(a, grad_z)
        gbe2 = grad_z.copy()
        grad_a = (self.W_enc2 @ grad_z) * (1.0 - a * a)
        ge1 = np.outer(x, grad_a)
        gbe1 = grad_a.copy()
        weights_and_gradients = (
            (self.W_dec2, gd2), (self.b_dec2, gb2),
            (self.W_dec1, gd1), (self.b_dec1, gb1),
            (self.W_enc2, ge2), (self.b_enc2, gbe2),
            (self.W_enc1, ge1), (self.b_enc1, gbe1),
        )
        for w, grad in weights_and_gradients:
            norm = float(np.linalg.norm(grad))
            w -= np.float32(learning_rate / max(1.0, norm)) * grad
        _, self.raw_latent, self.latent, _, recon = self._forward(x)
        self.last_error = float(np.mean((recon - x) ** 2))
        self.state[:3] = recon[:3] - self.pos
        if not np.isfinite(self.last_error):
            raise FloatingPointError("non-finite autoencoder reconstruction loss")
        return self.last_error


class SelfOptimisingSwarm3DANN:
    """Sparse virtual 3D swarm with stable centroid contraction and bounded meta-gates."""

    def __init__(self, n_particles: int = 128, latent_dim: int = 16,
                 max_iterations: int = 800, contract_strength: float = 0.45,
                 min_particles: int = 32, prune_threshold: float = 0.02,
                 meta_every: int = 8, seed: int = 42, hidden: int = 32,
                 learning_rate: float = 0.045):
        if not (1 <= min_particles <= n_particles <= 4096):
            raise ValueError("require 1 <= min_particles <= n_particles <= 4096")
        if latent_dim <= 0 or hidden <= 0 or max_iterations <= 0 or meta_every <= 0:
            raise ValueError("dimensions, iterations and meta_every must be positive")
        if not (0 < contract_strength < 1) or not (0 < learning_rate <= 0.5):
            raise ValueError("require 0 < contraction < 1 and 0 < learning_rate <= 0.5")
        if not (0 <= prune_threshold < 10):
            raise ValueError("invalid prune threshold")
        self.n_particles = n_particles
        self.latent_dim = latent_dim
        self.hidden = hidden
        self.max_iterations = max_iterations
        self.initial_contract_strength = float(contract_strength)
        self.contract_strength = float(contract_strength)
        self.min_particles = min_particles
        self.prune_threshold = float(prune_threshold)
        self.meta_every = meta_every
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.learning_rate = float(learning_rate)
        self.particles: List[NeuralParticle] = []
        self.history_error: List[float] = []
        self.particle_energy: Optional[np.ndarray] = None
        self.fixed_point_positions: Optional[np.ndarray] = None
        self.prune_history: List[Dict[str, int]] = []

    def embed(self, text: str) -> None:
        """One-way stable embedding into the 10^18-cell *address domain*."""
        text = text.strip() or "empty"
        self.rng = np.random.default_rng(self.seed)
        self.particles.clear()
        stride = max(1, len(text) // self.n_particles)
        for i in range(self.n_particles):
            start = (i * stride) % len(text)
            chunk = (text + text)[start : start + 24]
            pos, feats = _digest_position_and_features(f"{i}|{chunk}")
            self.particles.append(NeuralParticle(i, pos, feats, self.rng,
                                                 self.latent_dim, self.hidden))

    def virtual_coordinates(self) -> np.ndarray:
        if not self.particles:
            return np.empty((0, 3), dtype=np.int64)
        return np.minimum(
            np.floor(self.get_fixed_point_positions() * VIRTUAL_SIDE).astype(np.int64),
            VIRTUAL_SIDE - 1,
        )

    def encode_swarm(self) -> np.ndarray:
        if not self.particles:
            return np.empty((0, self.latent_dim), dtype=np.float32)
        return np.stack([p.encode(p.state[3:]) for p in self.particles])

    def contractive_step(self) -> float:
        if not self.particles:
            return 0.0
        positions = np.stack([p.pos for p in self.particles]).astype(np.float64)
        centroid = positions.mean(axis=0)
        # For a fixed particle set, pairwise distances contract by 1-g.
        # Pruning changes the set and thus the centroid.
        gain = min(0.85, max(0.08, self.contract_strength)) * 0.15
        losses = []
        for p in self.particles:
            p.pos = np.clip(centroid + (1.0 - gain) * (p.pos - centroid), 0, 1).astype(np.float32)
            losses.append(p.train_step(self.learning_rate))
        return float(np.mean(losses))

    def compute_particle_energy(self) -> np.ndarray:
        self.particle_energy = np.asarray([
            np.linalg.norm(p.latent) + np.linalg.norm(p.state[:3])
            for p in self.particles
        ], dtype=np.float32)
        return self.particle_energy.copy()

    def meta_contract(self) -> int:
        """Adaptive contraction and heuristic pruning; no quality gain guaranteed."""
        energies = self.compute_particle_energy()
        mean_energy = float(np.mean(energies)) if len(energies) else 0.0
        self.contract_strength *= (0.96 + 0.04 * math.tanh(mean_energy))
        self.contract_strength = float(np.clip(self.contract_strength, 0.08, 0.85))
        old_count = len(self.particles)
        if old_count > self.min_particles:
            cutoff = max(float(np.quantile(energies, 0.25)), self.prune_threshold)
            survivors = [p for p, e in zip(self.particles, energies) if e > cutoff]
            if len(survivors) < self.min_particles:
                order = sorted(range(old_count), key=lambda i: (-float(energies[i]), self.particles[i].id))
                keep = set(order[:self.min_particles])
                survivors = [p for i, p in enumerate(self.particles) if i in keep]
            self.particles = survivors
        for p in self.particles:
            magnitudes = np.abs(p.raw_latent)
            strongest = float(magnitudes.max())
            if strongest > 1e-7:
                p.mask = (magnitudes >= 0.15 * strongest).astype(np.float32)
                p.latent = p.raw_latent * p.mask
        removed = old_count - len(self.particles)
        self.prune_history.append({"before": old_count, "after": len(self.particles)})
        return removed

    def evolve(self, text: str) -> Dict[str, Any]:
        t0 = time.perf_counter()
        self.contract_strength = self.initial_contract_strength
        self.history_error.clear()
        self.prune_history.clear()
        self.fixed_point_positions = None
        self.embed(text)
        previous = None
        previous_loss = None
        stable_count = 0
        converged_at = None
        max_motion = math.inf
        for k in range(self.max_iterations):
            loss = self.contractive_step()
            self.history_error.append(loss)
            removed = 0
            if (k + 1) % self.meta_every == 0:
                removed = self.meta_contract()
            positions = {p.id: p.pos.copy() for p in self.particles}
            if previous is not None and not removed:
                max_motion = max(float(np.linalg.norm(pos - previous[pid]))
                                 for pid, pos in positions.items())
                if max_motion < 1e-5 and loss < 0.015 and previous_loss is not None \
                        and abs(previous_loss - loss) < 1e-6:
                    stable_count += 1
                else:
                    stable_count = 0
                if stable_count >= 3:
                    converged_at = k
                    break
            else:
                stable_count = 0
            previous = positions
            previous_loss = loss
        self.fixed_point_positions = np.stack([p.pos for p in self.particles])
        assert np.isfinite(self.fixed_point_positions).all()
        return {
            "virtual_volume": "1000000^3 symbolic cells (10^18 unallocated)",
            "initial_particles": self.n_particles,
            "final_particles": len(self.particles),
            "iterations": len(self.history_error),
            "converged_at": converged_at,
            "final_recon_error": float(self.history_error[-1]),
            "final_contract_strength": round(self.contract_strength, 6),
            "max_position_step": max_motion if math.isfinite(max_motion) else None,
            "prune_events": self.prune_history,
            "time_sec": round(time.perf_counter() - t0, 4),
            "status": "numerically_stationary" if converged_at is not None else "max_iter",
            "notes": "Loss measures reconstruction of 7-vector states, not original text. "
                     "Stationarity is a tolerance test, not proof of a full ANN fixed point.",
        }

    def summarize(self, text: str) -> str:
        return json.dumps(self.evolve(text), indent=2)

    def get_fixed_point_positions(self) -> Optional[np.ndarray]:
        if not self.particles:
            return None
        return np.stack([p.pos for p in self.particles])

    def get_swarm_latents(self) -> Optional[np.ndarray]:
        if not self.particles:
            return None
        return np.stack([p.latent for p in self.particles])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--particles", type=int, default=96)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--minimum", type=int, default=28)
    parser.add_argument("--text", default="def contractive_map(x, g=0.45): return x*(1-g)")
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    swarm = SelfOptimisingSwarm3DANN(n_particles=args.particles,
                                     max_iterations=args.iterations,
                                     min_particles=args.minimum)
    result = swarm.evolve(args.text)
    print(json.dumps(result, indent=2))
    if args.out:
        np.savez_compressed(args.out, positions=swarm.get_fixed_point_positions(),
                            addresses=swarm.virtual_coordinates(),
                            latents=swarm.get_swarm_latents(),
                            loss=np.asarray(swarm.history_error))


if __name__ == "__main__":
    main()
