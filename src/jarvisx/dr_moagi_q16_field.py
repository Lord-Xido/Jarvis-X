"""Q16.16 execution substrate for the Dr. Moagi 3D field equation.

The module implements the fixed-point and integrity mechanics stated by the
Dr. Moagi equation as bounded, deterministic software:

    Xi -> Psi gate -> Phi encode/convolve -> A -> SHA3 ledger
       -> Lambda projection -> upshift -> Theta decode -> V_out

It also exposes a typed sparse-field step for the continuous-time-inspired
master recurrence. Computational state and cryptographic ledger state are
kept separate by design.

The source equation names eta as a "quantum fluctuation". Here eta is an
explicit seeded stochastic numerical term; this software does not claim to
model or access a physical quantum process.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import random
from typing import Mapping, Sequence

Coordinate3D = tuple[int, int, int]
SparseRawField = dict[Coordinate3D, int]

INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1
UINT32_MASK = (1 << 32) - 1
Q_FRAC_BITS = 16
Q_SCALE = 1 << Q_FRAC_BITS
Q_OUTPUT_MAX_RAW = (1 << 16) - 1


def _require_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def sat_i32(value: int) -> int:
    """Clamp an integer to the signed 32-bit domain."""

    value = _require_int("value", value)
    return INT32_MIN if value < INT32_MIN else INT32_MAX if value > INT32_MAX else value


def q_from_float(value: float) -> int:
    """Quantize a finite real value to signed Q16.16 raw storage."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("value must be numeric")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("value must be finite")
    return sat_i32(int(round(value * Q_SCALE)))


def q_to_float(raw: int) -> float:
    raw = sat_i32(raw)
    return raw / Q_SCALE


def q_add(a: int, b: int) -> int:
    return sat_i32(sat_i32(a) + sat_i32(b))


def q_sub(a: int, b: int) -> int:
    return sat_i32(sat_i32(a) - sat_i32(b))


def q_mul(a: int, b: int) -> int:
    """Q16.16 multiply with a 64-bit-style intermediate and 16-bit rescale."""

    product = sat_i32(a) * sat_i32(b)
    if product >= 0:
        scaled = product >> Q_FRAC_BITS
    else:
        scaled = -((-product) >> Q_FRAC_BITS)
    return sat_i32(scaled)


def q_shift_left(raw: int, bits: int) -> int:
    _require_int("bits", bits)
    if bits < 0:
        raise ValueError("bits must be non-negative")
    return sat_i32(sat_i32(raw) << bits)


def bitwise_gate(raw: int, mask: int) -> int:
    """Apply the stated Psi bitwise-AND gate to a signed 32-bit Q payload."""

    raw = sat_i32(raw)
    mask = _require_int("mask", mask) & UINT32_MASK
    gated_u32 = (raw & UINT32_MASK) & mask
    return gated_u32 - (1 << 32) if gated_u32 & (1 << 31) else gated_u32


def project(raw: int, lower: int, upper: int) -> int:
    raw = sat_i32(raw)
    lower = sat_i32(lower)
    upper = sat_i32(upper)
    if lower > upper:
        raise ValueError("lower bound exceeds upper bound")
    return lower if raw < lower else upper if raw > upper else raw


@dataclass(frozen=True)
class Q16Interval:
    lower: int = INT32_MIN
    upper: int = INT32_MAX

    def __post_init__(self) -> None:
        if self.lower < INT32_MIN or self.upper > INT32_MAX:
            raise ValueError("interval must fit signed 32-bit storage")
        if self.lower > self.upper:
            raise ValueError("lower bound exceeds upper bound")

    def apply(self, raw: int) -> int:
        return project(raw, self.lower, self.upper)


class Sha3Ledger:
    """Append-only SHA3-256 chain over serialized operational records."""

    GENESIS = bytes(32)

    def __init__(self) -> None:
        self.head = self.GENESIS
        self.entries: list[dict[str, object]] = []

    @staticmethod
    def _record_bytes(record: Mapping[str, object]) -> bytes:
        return json.dumps(
            dict(record), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")

    def bind(self, record: Mapping[str, object]) -> str:
        payload = self.head + self._record_bytes(record)
        digest = hashlib.sha3_256(payload).digest()
        self.entries.append(
            {
                "prev": self.head.hex(),
                "record": dict(record),
                "hash": digest.hex(),
            }
        )
        self.head = digest
        return digest.hex()

    def verify(self) -> bool:
        previous = self.GENESIS
        for envelope in self.entries:
            if envelope.get("prev") != previous.hex():
                return False
            record = envelope.get("record")
            if not isinstance(record, Mapping):
                return False
            digest = hashlib.sha3_256(previous + self._record_bytes(record)).digest()
            if envelope.get("hash") != digest.hex():
                return False
            previous = digest
        return previous == self.head


@dataclass(frozen=True)
class CellTrace:
    coordinate: Coordinate3D
    gated_values: tuple[int, ...]
    convolution_raw: int
    activation_raw: int
    ledger_hash: str
    safe_activation_raw: int
    upshift_raw: int
    decoded_raw: int
    output_raw: int


@dataclass(frozen=True)
class FieldStepReport:
    tick: int
    output: SparseRawField
    mean_abs_delta: float
    max_abs_raw: int
    ledger_hash: str


@dataclass(frozen=True)
class DrMoagiQ16Config:
    side: int = 64
    lambda_inverse_raw: int = field(default_factory=lambda: q_from_float(1.0))
    gamma_gain_raw: int = field(default_factory=lambda: q_from_float(0.0))
    eta_amplitude_raw: int = field(default_factory=lambda: q_from_float(0.0))
    seed: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.side, bool) or not isinstance(self.side, int) or self.side <= 0:
            raise ValueError("side must be a positive integer")
        for name in ("lambda_inverse_raw", "gamma_gain_raw", "eta_amplitude_raw"):
            value = getattr(self, name)
            if value < INT32_MIN or value > INT32_MAX:
                raise ValueError(f"{name} must fit signed 32-bit storage")
        _require_int("seed", self.seed)


class DrMoagiQ16FieldRuntime:
    """Sparse 3D Q16.16 runtime for the master equation and codec bus."""

    def __init__(self, config: DrMoagiQ16Config | None = None) -> None:
        self.config = config or DrMoagiQ16Config()
        self.ledger = Sha3Ledger()
        self.rng = random.Random(self.config.seed)
        self.state: SparseRawField = {}
        self.previous_state: SparseRawField = {}
        self.tick = 0

    def _coord(self, coordinate: Coordinate3D) -> Coordinate3D:
        if not isinstance(coordinate, tuple) or len(coordinate) != 3:
            raise TypeError("coordinate must be a 3-tuple")
        out: list[int] = []
        for axis in coordinate:
            _require_int("coordinate axis", axis)
            if not 0 <= axis < self.config.side:
                raise ValueError("coordinate outside configured 3D lattice")
            out.append(axis)
        return out[0], out[1], out[2]

    def load(self, values: Mapping[Coordinate3D, int]) -> None:
        parsed: SparseRawField = {}
        for coordinate, raw in values.items():
            coord = self._coord(coordinate)
            parsed[coord] = sat_i32(raw)
        self.previous_state = dict(parsed)
        self.state = parsed
        self.tick = 0

    def load_float(self, values: Mapping[Coordinate3D, float]) -> None:
        self.load({coord: q_from_float(value) for coord, value in values.items()})

    @staticmethod
    def encode_decode_cell(
        *,
        coordinate: Coordinate3D,
        values: Sequence[int],
        psi_masks: Sequence[int],
        phi_weights: Sequence[int],
        theta_weights: Sequence[int],
        constraint: Q16Interval,
        ledger: Sha3Ledger,
        tick: int = 0,
    ) -> CellTrace:
        """Execute the stated discrete update law for one logical cell.

        G_k = V_k & Psi_k
        C = sum_k Q16mul(G_k, W_phi_k)
        A = clamp(C, 0, INT32_MAX)
        H_t = SHA3(H_{t-1} || serialize(A,t,coord))
        A_safe = project(A, Lambda_min, Lambda_max)
        U = sat(A_safe << 2)
        D = sum_j Q16mul(U, W_theta_j)
        V_out = clamp(D, 0, 2^16-1) [raw integer domain]
        """

        if not (len(values) == len(psi_masks) == len(phi_weights)):
            raise ValueError("values, psi_masks and phi_weights must have equal length")
        if not theta_weights:
            raise ValueError("theta_weights must not be empty")

        gated = tuple(bitwise_gate(v, m) for v, m in zip(values, psi_masks))
        convolution = 0
        for g, w in zip(gated, phi_weights):
            convolution = q_add(convolution, q_mul(g, w))
        activation = project(convolution, 0, INT32_MAX)
        ledger_hash = ledger.bind(
            {
                "tick": int(tick),
                "coordinate": list(coordinate),
                "activation_raw": activation,
            }
        )
        safe = constraint.apply(activation)
        upshift = q_shift_left(safe, 2)
        decoded = 0
        for weight in theta_weights:
            decoded = q_add(decoded, q_mul(upshift, weight))
        output = project(decoded, 0, Q_OUTPUT_MAX_RAW)
        return CellTrace(
            coordinate=coordinate,
            gated_values=gated,
            convolution_raw=convolution,
            activation_raw=activation,
            ledger_hash=ledger_hash,
            safe_activation_raw=safe,
            upshift_raw=upshift,
            decoded_raw=decoded,
            output_raw=output,
        )

    @staticmethod
    def _sample(field: Mapping[Coordinate3D, int], coordinate: Coordinate3D) -> int:
        return sat_i32(field.get(coordinate, 0))

    def convolve(
        self,
        field: Mapping[Coordinate3D, int],
        kernel: Mapping[Coordinate3D, int],
        coordinate: Coordinate3D,
    ) -> int:
        total = 0
        x, y, z = coordinate
        for (dx, dy, dz), weight in kernel.items():
            target = (x + dx, y + dy, z + dz)
            if not all(0 <= target[i] < self.config.side for i in range(3)):
                continue
            total = q_add(total, q_mul(self._sample(field, target), weight))
        return total

    def step_field(
        self,
        *,
        psi_raw: Mapping[Coordinate3D, int] | None = None,
        phi_kernel: Mapping[Coordinate3D, int] | None = None,
        adaptive_gradient_raw: Mapping[Coordinate3D, int] | None = None,
        constraints: Mapping[Coordinate3D, Q16Interval] | None = None,
    ) -> FieldStepReport:
        """Advance a sparse discretization of the master recurrence by one tick.

        Xi_{t+1} = Pi_Lambda[Xi_t + Psi + Phi*Xi_t
                             - Lambda^-1 * grad_E
                             + Omega(Xi_{t-1})
                             + Gamma(Xi_t-Xi_{t-1}) + eta]

        Omega is represented computationally as the previous field contribution;
        the cryptographic Omega ledger is maintained separately by SHA3-256.
        """

        psi_raw = psi_raw or {}
        phi_kernel = phi_kernel or {(0, 0, 0): 0}
        adaptive_gradient_raw = adaptive_gradient_raw or {}
        constraints = constraints or {}

        support = set(self.state) | set(self.previous_state) | set(psi_raw) | set(adaptive_gradient_raw)
        next_state: SparseRawField = {}
        sum_abs_delta = 0
        max_abs_raw = 0

        for coord in sorted(support):
            current = self._sample(self.state, coord)
            previous = self._sample(self.previous_state, coord)
            intent = sat_i32(psi_raw.get(coord, 0))
            phi = self.convolve(self.state, phi_kernel, coord)
            adaptive = q_mul(
                self.config.lambda_inverse_raw,
                sat_i32(adaptive_gradient_raw.get(coord, 0)),
            )
            torsion = q_mul(self.config.gamma_gain_raw, q_sub(current, previous))
            eta = 0
            if self.config.eta_amplitude_raw:
                eta_unit = q_from_float(self.rng.uniform(-1.0, 1.0))
                eta = q_mul(self.config.eta_amplitude_raw, eta_unit)

            candidate = current
            candidate = q_add(candidate, intent)
            candidate = q_add(candidate, phi)
            candidate = q_sub(candidate, adaptive)
            candidate = q_add(candidate, previous)
            candidate = q_add(candidate, torsion)
            candidate = q_add(candidate, eta)
            candidate = constraints.get(coord, Q16Interval()).apply(candidate)

            if candidate != 0:
                next_state[coord] = candidate
            delta = abs(candidate - current)
            sum_abs_delta += delta
            max_abs_raw = max(max_abs_raw, abs(candidate))

        self.previous_state = dict(self.state)
        self.state = next_state
        self.tick += 1
        ledger_hash = self.ledger.bind(
            {
                "tick": self.tick,
                "cells": [
                    [coord[0], coord[1], coord[2], raw]
                    for coord, raw in sorted(next_state.items())
                ],
            }
        )
        mean_abs_delta = (sum_abs_delta / max(1, len(support))) / Q_SCALE
        return FieldStepReport(
            tick=self.tick,
            output=dict(next_state),
            mean_abs_delta=mean_abs_delta,
            max_abs_raw=max_abs_raw,
            ledger_hash=ledger_hash,
        )


def temporal_compression_law(v_clock: float) -> dict[str, object]:
    """Return an exact symbolic/log-domain representation of the stated law.

    v_clock^infinity = exp(10^(6^(10^6)) * v_clock)

    Materializing the coefficient or exponential is intentionally avoided: the
    coefficient is astronomically beyond finite machine representation. For
    v_clock > 0 the extended-real value diverges to +infinity; for v_clock == 0
    it is 1; for v_clock < 0 it tends to 0.
    """

    if isinstance(v_clock, bool) or not isinstance(v_clock, (int, float)):
        raise TypeError("v_clock must be numeric")
    v_clock = float(v_clock)
    if not math.isfinite(v_clock):
        raise ValueError("v_clock must be finite")
    limit = "+infinity" if v_clock > 0 else "1" if v_clock == 0 else "0"
    return {
        "law": "exp(10^(6^(10^6)) * v_clock)",
        "log_value": "10^(6^(10^6)) * v_clock",
        "v_clock": v_clock,
        "extended_real_limit": limit,
        "materialized": False,
    }
