from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(slots=True)
class TrainMetrics:
    loss: float
    reconstruction_error: float
    latent_l1: float


class TinyAutoencoder:
    """Small NumPy autoencoder suitable for VM demonstrations and SDK testing."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        *,
        seed: int = 7,
        learning_rate: float = 0.01,
    ) -> None:
        if input_dim <= 0 or latent_dim <= 0:
            raise ValueError("dimensions must be positive")
        if latent_dim >= input_dim:
            raise ValueError("latent_dim should be smaller than input_dim")
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.learning_rate = float(learning_rate)
        rng = np.random.default_rng(seed)
        scale_e = np.sqrt(2.0 / input_dim)
        scale_d = np.sqrt(2.0 / latent_dim)
        self.w_enc = rng.normal(0.0, scale_e, (input_dim, latent_dim)).astype(np.float32)
        self.b_enc = np.zeros(latent_dim, dtype=np.float32)
        self.w_dec = rng.normal(0.0, scale_d, (latent_dim, input_dim)).astype(np.float32)
        self.b_dec = np.zeros(input_dim, dtype=np.float32)

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(x, 0.0)

    @staticmethod
    def _relu_grad(x: np.ndarray) -> np.ndarray:
        return (x > 0.0).astype(np.float32)

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        x = np.clip(x, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-x))

    def _as_batch(self, x: np.ndarray | list[float]) -> np.ndarray:
        array = np.asarray(x, dtype=np.float32)
        if array.ndim == 1:
            array = array[None, :]
        if array.ndim != 2 or array.shape[1] != self.input_dim:
            raise ValueError(f"expected shape (batch, {self.input_dim}), got {array.shape}")
        return array

    def encode(self, x: np.ndarray | list[float]) -> np.ndarray:
        batch = self._as_batch(x)
        return self._relu(batch @ self.w_enc + self.b_enc)

    def decode(self, z: np.ndarray | list[float]) -> np.ndarray:
        latent = np.asarray(z, dtype=np.float32)
        if latent.ndim == 1:
            latent = latent[None, :]
        if latent.ndim != 2 or latent.shape[1] != self.latent_dim:
            raise ValueError(f"expected latent shape (batch, {self.latent_dim})")
        return self._sigmoid(latent @ self.w_dec + self.b_dec)

    def reconstruct(self, x: np.ndarray | list[float]) -> np.ndarray:
        return self.decode(self.encode(x))

    def train_step(self, x: np.ndarray | list[float]) -> TrainMetrics:
        batch = self._as_batch(x)
        batch_size = batch.shape[0]

        pre_z = batch @ self.w_enc + self.b_enc
        z = self._relu(pre_z)
        pre_y = z @ self.w_dec + self.b_dec
        y = self._sigmoid(pre_y)

        error = y - batch
        loss = float(np.mean(error**2))

        d_y = (2.0 / (batch_size * self.input_dim)) * error
        d_pre_y = d_y * y * (1.0 - y)
        d_w_dec = z.T @ d_pre_y
        d_b_dec = d_pre_y.sum(axis=0)
        d_z = d_pre_y @ self.w_dec.T
        d_pre_z = d_z * self._relu_grad(pre_z)
        d_w_enc = batch.T @ d_pre_z
        d_b_enc = d_pre_z.sum(axis=0)

        clip = 5.0
        for grad in (d_w_enc, d_b_enc, d_w_dec, d_b_dec):
            np.clip(grad, -clip, clip, out=grad)

        lr = self.learning_rate
        self.w_enc -= lr * d_w_enc
        self.b_enc -= lr * d_b_enc
        self.w_dec -= lr * d_w_dec
        self.b_dec -= lr * d_b_dec

        return TrainMetrics(
            loss=loss,
            reconstruction_error=float(np.mean(np.abs(error))),
            latent_l1=float(np.mean(np.abs(z))),
        )

    def state_dict(self) -> dict[str, Any]:
        """Return an isolated model snapshot suitable for transactional staging."""

        return {
            "input_dim": self.input_dim,
            "latent_dim": self.latent_dim,
            "learning_rate": self.learning_rate,
            "w_enc": self.w_enc.copy(),
            "b_enc": self.b_enc.copy(),
            "w_dec": self.w_dec.copy(),
            "b_dec": self.b_dec.copy(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if int(state["input_dim"]) != self.input_dim:
            raise ValueError("input_dim mismatch")
        if int(state["latent_dim"]) != self.latent_dim:
            raise ValueError("latent_dim mismatch")
        self.learning_rate = float(state["learning_rate"])
        self.w_enc = np.asarray(state["w_enc"], dtype=np.float32).copy()
        self.b_enc = np.asarray(state["b_enc"], dtype=np.float32).copy()
        self.w_dec = np.asarray(state["w_dec"], dtype=np.float32).copy()
        self.b_dec = np.asarray(state["b_dec"], dtype=np.float32).copy()
        self._validate_state()

    def clone(self) -> "TinyAutoencoder":
        clone = TinyAutoencoder(
            self.input_dim,
            self.latent_dim,
            seed=0,
            learning_rate=self.learning_rate,
        )
        clone.load_state_dict(self.state_dict())
        return clone

    def _validate_state(self) -> None:
        expected = {
            "w_enc": (self.input_dim, self.latent_dim),
            "b_enc": (self.latent_dim,),
            "w_dec": (self.latent_dim, self.input_dim),
            "b_dec": (self.input_dim,),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if value.shape != shape:
                raise ValueError(f"{name} shape mismatch: expected {shape}, got {value.shape}")
            if not np.all(np.isfinite(value)):
                raise FloatingPointError(f"{name} contains non-finite values")

    def save(self, path: str | Path) -> None:
        np.savez_compressed(path, **self.state_dict())

    @classmethod
    def load(cls, path: str | Path) -> "TinyAutoencoder":
        with np.load(path) as data:
            model = cls(
                int(data["input_dim"]),
                int(data["latent_dim"]),
                learning_rate=float(data["learning_rate"]),
            )
            model.load_state_dict({name: data[name] for name in data.files})
        return model
