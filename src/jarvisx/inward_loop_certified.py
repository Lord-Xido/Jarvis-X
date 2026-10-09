"""Deterministic analytic torus codec with a replayable SHA-256 witness chain.

This is an analytic Fourier embedding, NOT a trained neural network. The
hash chain is tamper-evident when independently anchored; it is not a signature.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Any

import numpy as np

R_MAJOR = 5.2
R_TUBE = 0.85
N = 8192
Z_DIM = 16
D_MODES = 4
SCHEMA = "jarvisx.inward-loop-certificate.v1"
TAU = 2.0 * math.pi


def torus(u: np.ndarray, v: np.ndarray, R: float = R_MAJOR, r: float = R_TUBE) -> np.ndarray:
    """Cartesian torus whose revolution axis is the Y axis."""
    u, v = np.broadcast_arrays(np.asarray(u, dtype=np.float64), np.asarray(v, dtype=np.float64))
    x = (R + r * np.cos(v)) * np.cos(u)
    y = r * np.sin(v)
    z = (R + r * np.cos(v)) * np.sin(u)
    return np.stack((x, y, z), axis=-1)


class FourierEncoder:
    """Four harmonics per circle: [cos ku, sin ku, cos lv, sin lv]."""

    def __init__(self) -> None:
        self.km = np.arange(1, D_MODES + 1, dtype=np.float64)
        self.lm = np.arange(1, D_MODES + 1, dtype=np.float64)

    def uv_from_xyz(self, P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        P = np.asarray(P, dtype=np.float64)
        if P.ndim != 2 or P.shape[1] != 3 or len(P) == 0:
            raise ValueError("P must have shape (N,3), with N >= 1")
        if not np.isfinite(P).all():
            raise ValueError("P contains non-finite coordinates")
        x, y, z = P.T
        radial = np.hypot(x, z)
        if np.any(radial < 1e-12):
            raise ValueError("Projection undefined on torus revolution axis")
        return np.arctan2(z, x), np.arctan2(y, radial - R_MAJOR)

    def __call__(self, P: np.ndarray) -> np.ndarray:
        u, v = self.uv_from_xyz(P)
        Z = np.concatenate(
            (
                np.cos(u[:, None] * self.km),
                np.sin(u[:, None] * self.km),
                np.cos(v[:, None] * self.lm),
                np.sin(v[:, None] * self.lm),
            ),
            axis=1,
        )
        assert Z.shape == (len(P), Z_DIM)
        return Z


class FourierDecoder:
    """Read first harmonic pairs to recover both torus angular coordinates."""

    def __call__(self, Z: np.ndarray) -> np.ndarray:
        Z = np.asarray(Z, dtype=np.float64)
        if Z.ndim != 2 or Z.shape[1] != Z_DIM or len(Z) == 0:
            raise ValueError("Z must have shape (N,16), with N >= 1")
        if not np.isfinite(Z).all():
            raise ValueError("Z contains non-finite values")
        if np.any(np.hypot(Z[:, 0], Z[:, 4]) < 1e-12) or np.any(
            np.hypot(Z[:, 8], Z[:, 12]) < 1e-12
        ):
            raise ValueError("First-harmonic angle undefined at zero pair")
        return torus(np.arctan2(Z[:, 4], Z[:, 0]), np.arctan2(Z[:, 12], Z[:, 8]))


class Field:
    """Deterministic angular shear transport (not a learned dynamics model)."""

    def __init__(self, seed: int = 0, n: int = N) -> None:
        if n < 1:
            raise ValueError("n must be >= 1")
        rng = np.random.default_rng(seed)
        self.u = rng.uniform(0.0, TAU, n)
        self.v = rng.uniform(0.0, TAU, n)
        self.wu = 0.01
        self.wv = 0.013
        self.kappa = 0.35
        self.t = 0.0

    def step(self, dt: float = 1.0 / 60.0) -> None:
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be a finite positive number")
        self.v = (self.v + (self.wv + self.kappa * np.sin(self.u)) * dt) % TAU
        self.u = (self.u + self.wu * dt) % TAU
        self.t += dt

    @property
    def P(self) -> np.ndarray:
        return torus(self.u, self.v)


def canonical_bytes(A: np.ndarray) -> bytes:
    """Canonical C-order little-endian float64 bytes, including array shape."""
    arr = np.ascontiguousarray(np.asarray(A, dtype="<f8"))
    return struct.pack("<I", arr.ndim) + struct.pack(f"<{arr.ndim}Q", *arr.shape) + arr.tobytes()


def _digest(*parts: bytes) -> bytes:
    h = hashlib.sha256()
    for item in parts:
        h.update(struct.pack("<Q", len(item)))
        h.update(item)
    return h.digest()


class Certificate:
    """Replayable content commitment, not an independently authenticated proof."""

    def __init__(self, seed: int, n: int, dt: float) -> None:
        self.chain = _digest(
            SCHEMA.encode(),
            json.dumps(
                {"seed": seed, "particles": n, "dt": dt, "R": R_MAJOR, "r": R_TUBE, "modes": D_MODES},
                sort_keys=True, separators=(",", ":"),
            ).encode(),
        )
        self.n = 0

    def advance(self, P: np.ndarray, Z: np.ndarray, X: np.ndarray, residual: np.ndarray, J: np.ndarray) -> str:
        self.n += 1
        self.chain = _digest(
            self.chain,
            struct.pack("<Q", self.n),
            canonical_bytes(P), canonical_bytes(Z), canonical_bytes(X),
            canonical_bytes(residual), canonical_bytes(J),
        )
        return self.chain.hex()


def projection_jacobian(
    P0: np.ndarray, enc: FourierEncoder, dec: FourierDecoder, eps: float = 1e-5
) -> np.ndarray:
    """Central-difference Jacobian evaluated on the torus surface, not at cloud centroid."""
    P0 = np.asarray(P0, dtype=np.float64)
    if P0.shape != (3,) or eps <= 0:
        raise ValueError("P0 must be one 3D point and eps > 0")
    basis = np.eye(3) * eps
    plus = dec(enc(P0[None, :] + basis))
    minus = dec(enc(P0[None, :] - basis))
    return (plus - minus).T / (2.0 * eps)


class InwardLoop:
    def __init__(self, seed: int = 0, n: int = N, dt: float = 1.0 / 60.0) -> None:
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be positive and finite")
        self.seed, self.n, self.dt = int(seed), int(n), float(dt)
        self.enc, self.dec = FourierEncoder(), FourierDecoder()
        self.field = Field(seed=self.seed, n=self.n)
        self.cert = Certificate(seed=self.seed, n=self.n, dt=self.dt)
        self.frame = 0

    def step(self) -> dict[str, float | int | str]:
        P = self.field.P
        Z = self.enc(P)
        X = self.dec(Z)
        r = np.linalg.norm(X - P, axis=1)
        X2 = self.dec(self.enc(X))
        J = projection_jacobian(P[0], self.enc, self.dec)
        eigvals = np.linalg.eigvals(J)
        eigvals = np.sort(eigvals.real)[::-1]
        eig_error = float(np.max(np.abs(eigvals - np.array((1.0, 1.0, 0.0)))))
        projector_error = float(np.max(np.abs(J @ J - J)))
        u, v = self.enc.uv_from_xyz(P[:1])
        normal = np.array((np.cos(u[0]) * np.cos(v[0]), np.sin(v[0]), np.sin(u[0]) * np.cos(v[0])))
        tangent_error = float(np.max(np.abs(J - (np.eye(3) - np.outer(normal, normal)))))
        root = self.cert.advance(P, Z, X, r, J)
        self.frame += 1
        result: dict[str, float | int | str] = {
            "frame": self.frame,
            "time": self.field.t,
            "cert_root": root,
            "cert_len": self.cert.n,
            "residual_max": float(np.max(r)),
            "residual_mean": float(np.mean(r)),
            "residual_witness_index": int(np.argmax(r)),
            "idempotence": float(np.max(np.linalg.norm(X2 - X, axis=1))),
            "eig_error": eig_error,
            "projector_error": projector_error,
            "tangent_error": tangent_error,
        }
        self.field.step(self.dt)
        return result


def run(frames: int = 600, n: int = N, seed: int = 0, dt: float = 1 / 60) -> dict[str, Any]:
    if frames < 1:
        raise ValueError("frames must be >= 1")
    loop = InwardLoop(seed=seed, n=n, dt=dt)
    records = [loop.step() for _ in range(frames)]
    return {
        "schema": SCHEMA,
        "configuration": {"seed": seed, "particles": n, "frames": frames, "dt": dt},
        "manifold": {"major_radius": R_MAJOR, "tube_radius": R_TUBE, "intrinsic_dim": 2},
        "latent_dim": Z_DIM,
        "metrics": records,
        "final_root": loop.cert.chain.hex(),
        "method": "analytic Fourier torus codec with finite-difference tangent audit",
        "note": "Hash chain commits to computed arrays; independent anchoring/signing is required for provenance.",
    }


def verify_manifest(payload: dict[str, Any]) -> bool:
    """Deterministically recompute every frame and compare data and certificate roots."""
    if payload.get("schema") != SCHEMA:
        return False
    try:
        c = payload["configuration"]
        regenerated = run(frames=c["frames"], n=c["particles"], seed=c["seed"], dt=c["dt"])
        return regenerated == payload
    except (ValueError, TypeError, KeyError, OverflowError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Measured 3D torus codec with a replayable witness chain")
    parser.add_argument("--frames", type=int, default=600)
    parser.add_argument("--particles", type=int, default=N)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dt", type=float, default=1.0 / 60)
    parser.add_argument("--json", type=Path, default=None, help="Export complete per-frame certificate manifest")
    parser.add_argument("--verify", type=Path, default=None, help="Recompute and verify an existing manifest")
    args = parser.parse_args()
    if args.verify:
        payload = json.loads(args.verify.read_text(encoding="utf-8"))
        passed = verify_manifest(payload)
        print(f"{'PASS' if passed else 'FAIL'}: certificate replay {args.verify}")
        if not passed:
            raise SystemExit(1)
        return
    result = run(frames=args.frames, n=args.particles, seed=args.seed, dt=args.dt)
    for m in result["metrics"]:
        if m["frame"] == 1 or m["frame"] % 60 == 0 or m["frame"] == args.frames:
            print(
                f"frame={m['frame']:5d} cert={m['cert_root'][:16]} "
                f"res_max={m['residual_max']:.3e} idem={m['idempotence']:.3e} "
                f"eig_err={m['eig_error']:.3e}"
            )
    print("final_root:", result["final_root"])
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print("manifest:", args.json)


if __name__ == "__main__":
    main()
