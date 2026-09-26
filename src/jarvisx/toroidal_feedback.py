from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import importlib
import math
from typing import Any, cast

np: Any = importlib.import_module("numpy")
Array = Any


TAU = 2.0 * math.pi


def _wrap_angle(values: Array) -> Array:
    return np.mod(values, TAU)


@dataclass(frozen=True)
class TorusConfig:
    """Configuration for the closed 3D toroidal feedback machine."""

    major_radius: float = 18.0
    minor_radius: float = 6.0
    lambda_permeation: float = 0.9997
    window: int = 32
    top_k: int = 4
    dt: float = 1.0

    def __post_init__(self) -> None:
        if self.major_radius <= 0.0:
            raise ValueError("major_radius must be positive")
        if self.minor_radius <= 0.0:
            raise ValueError("minor_radius must be positive")
        if not 0.0 < self.lambda_permeation < 1.0:
            raise ValueError("lambda_permeation must lie strictly in (0, 1)")
        if self.window < 2:
            raise ValueError("window must be >= 2")
        if self.top_k < 1:
            raise ValueError("top_k must be >= 1")
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")


@dataclass(frozen=True)
class TorusState:
    u: Array
    v: Array
    sigma: float

    def __post_init__(self) -> None:
        u = np.asarray(self.u, dtype=np.float64)
        v = np.asarray(self.v, dtype=np.float64)
        if u.ndim != 1 or v.ndim != 1:
            raise ValueError("u and v must be one-dimensional arrays")
        if u.shape != v.shape:
            raise ValueError("u and v must have identical shapes")
        if u.size == 0:
            raise ValueError("state batch cannot be empty")
        if not np.isfinite(u).all() or not np.isfinite(v).all():
            raise ValueError("u and v must be finite")
        if not 0.0 < float(self.sigma) <= 1.0:
            raise ValueError("sigma must lie in (0, 1]")

        object.__setattr__(self, "u", _wrap_angle(u))
        object.__setattr__(self, "v", _wrap_angle(v))
        object.__setattr__(self, "sigma", float(self.sigma))

    @property
    def batch(self) -> int:
        return int(self.u.size)


@dataclass(frozen=True)
class SpectralModel:
    energy: float
    winding: int
    phase_u: float
    phase_v: float
    singular_values: Array


@dataclass(frozen=True)
class StepTelemetry:
    step: int
    sigma: float
    energy: float
    winding: int
    phase_u: float
    phase_v: float
    core_distance: float
    mean_speed: float


@dataclass(frozen=True)
class RunResult:
    trajectory: Array
    telemetry: tuple[StepTelemetry, ...]
    final_state: TorusState


class ToroidalFeedbackEngine:
    """
    Operational 3D dynamical machine on T^2 x (0, 1].

    Recent wrapped angular history is represented as
    [cos(u), sin(u), cos(v), sin(v)], compressed by SVD, analysed by FFT,
    converted into a feedback vector field, and integrated with a midpoint
    (RK2) step while sigma follows its exact exponential permeation law.
    """

    def __init__(self, state: TorusState, config: TorusConfig | None = None) -> None:
        self.config = config or TorusConfig()
        self.state = state
        self._step = 0
        encoded = self.encode_angles(state.u, state.v)
        self._history = np.repeat(
            encoded[:, np.newaxis, :],
            self.config.window,
            axis=1,
        )

    @staticmethod
    def encode_angles(u: Array, v: Array) -> Array:
        """Encode e^(iu) and e^(iv) as four real channels."""
        return np.stack(
            (np.cos(u), np.sin(u), np.cos(v), np.sin(v)),
            axis=-1,
        )

    def history_matrix(self) -> Array:
        """Return the batch x (4 * window) self-representation matrix."""
        return self._history.reshape(self.state.batch, 4 * self.config.window).copy()

    def embed(self, state: TorusState | None = None) -> Array:
        """Embed (u, v, sigma) into Euclidean 3-space."""
        s = self.state if state is None else state
        r = s.sigma * self.config.minor_radius
        rho = self.config.major_radius + r * np.cos(s.v)
        x = rho * np.cos(s.u)
        y = r * np.sin(s.v)
        z = rho * np.sin(s.u)
        return np.stack((x, y, z), axis=-1)

    def spectral_model(self) -> SpectralModel:
        """Compute SVD energy and dominant non-DC temporal winding."""
        matrix = self.history_matrix()
        _, singular_values, vh = np.linalg.svd(matrix, full_matrices=False)

        total = float(np.dot(singular_values, singular_values))
        if total <= np.finfo(np.float64).eps:
            energy = 0.0
        else:
            k = min(self.config.top_k, singular_values.size)
            energy = float(np.dot(singular_values[:k], singular_values[:k]) / total)

        dominant = vh[0].reshape(self.config.window, 4)
        u_mode = dominant[:, 0] + 1j * dominant[:, 1]
        v_mode = dominant[:, 2] + 1j * dominant[:, 3]

        u_fft = np.fft.fft(u_mode)
        v_fft = np.fft.fft(v_mode)
        max_positive = self.config.window // 2

        if max_positive < 1:
            winding = 1
        else:
            power = (
                np.abs(u_fft[1 : max_positive + 1]) ** 2
                + np.abs(v_fft[1 : max_positive + 1]) ** 2
            )
            if power.size == 0 or float(np.max(power)) <= np.finfo(np.float64).eps:
                winding = 1
            else:
                winding = int(np.argmax(power) + 1)

        phase_u = float(np.angle(u_fft[winding]))
        phase_v = float(np.angle(v_fft[winding]))

        return SpectralModel(
            energy=float(np.clip(energy, 0.0, 1.0)),
            winding=winding,
            phase_u=phase_u,
            phase_v=phase_v,
            singular_values=singular_values.copy(),
        )

    @staticmethod
    def _field(
        u: Array,
        v: Array,
        model: SpectralModel,
    ) -> tuple[Array, Array]:
        q = model.winding
        f = model.energy * np.cos(q * u + model.phase_u) * np.cos(v)
        g = model.energy * np.sin(q * v + model.phase_v) * np.cos(u)
        return f, g

    def angular_velocity(
        self,
        u: Array,
        v: Array,
        sigma: float,
        model: SpectralModel,
    ) -> tuple[Array, Array]:
        f, g = self._field(u, v, model)
        return sigma * f, sigma * g

    def _append_history(self, state: TorusState) -> None:
        self._history = np.roll(self._history, shift=-1, axis=1)
        self._history[:, -1, :] = self.encode_angles(state.u, state.v)

    def core_distance(self, state: TorusState | None = None) -> float:
        s = self.state if state is None else state
        return self.config.minor_radius * s.sigma

    def step(self) -> StepTelemetry:
        """Advance one closed-loop cycle."""
        cfg = self.config
        current = self.state
        model = self.spectral_model()

        du1, dv1 = self.angular_velocity(current.u, current.v, current.sigma, model)

        sigma_mid = current.sigma * (cfg.lambda_permeation ** (0.5 * cfg.dt))
        u_mid = _wrap_angle(current.u + 0.5 * cfg.dt * du1)
        v_mid = _wrap_angle(current.v + 0.5 * cfg.dt * dv1)

        du2, dv2 = self.angular_velocity(u_mid, v_mid, sigma_mid, model)

        sigma_new = current.sigma * (cfg.lambda_permeation ** cfg.dt)
        next_state = TorusState(
            u=_wrap_angle(current.u + cfg.dt * du2),
            v=_wrap_angle(current.v + cfg.dt * dv2),
            sigma=sigma_new,
        )

        self.state = next_state
        self._step += 1
        self._append_history(next_state)

        mean_speed = float(np.mean(np.sqrt(du2 * du2 + dv2 * dv2)))
        return StepTelemetry(
            step=self._step,
            sigma=next_state.sigma,
            energy=model.energy,
            winding=model.winding,
            phase_u=model.phase_u,
            phase_v=model.phase_v,
            core_distance=self.core_distance(next_state),
            mean_speed=mean_speed,
        )

    def run(self, steps: int) -> RunResult:
        if steps < 0:
            raise ValueError("steps must be non-negative")

        positions = [self.embed()]
        telemetry: list[StepTelemetry] = []
        for _ in range(steps):
            telemetry.append(self.step())
            positions.append(self.embed())

        return RunResult(
            trajectory=np.stack(positions, axis=0),
            telemetry=tuple(telemetry),
            final_state=self.state,
        )


def make_state(
    u: float | Iterable[float] = 1.4,
    v: float | Iterable[float] = 4.8,
    sigma: float = 1.0,
    *,
    batch: int = 64,
) -> TorusState:
    """Create a deterministic batch from scalar or iterable torus coordinates."""
    if batch < 1:
        raise ValueError("batch must be >= 1")

    def _coerce(value: float | Iterable[float], name: str) -> Array:
        if not isinstance(value, Iterable):
            return np.full(batch, float(cast(Any, value)), dtype=np.float64)
        arr = np.asarray(tuple(value), dtype=np.float64)
        if arr.ndim != 1:
            raise ValueError(f"{name} must be scalar or one-dimensional")
        return arr

    u_arr = _coerce(u, "u")
    v_arr = _coerce(v, "v")
    if u_arr.shape != v_arr.shape:
        raise ValueError("u and v iterable shapes must match")
    return TorusState(u_arr, v_arr, sigma)


def demo(steps: int = 32) -> RunResult:
    engine = ToroidalFeedbackEngine(make_state())
    return engine.run(steps)


def main() -> int:
    result = demo()
    last = result.telemetry[-1] if result.telemetry else None
    if last is not None:
        print(
            f"steps={last.step} sigma={last.sigma:.8f} "
            f"energy={last.energy:.6f} q={last.winding} "
            f"core_distance={last.core_distance:.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
