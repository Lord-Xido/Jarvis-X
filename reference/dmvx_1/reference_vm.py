#!/usr/bin/env python3
"""Deterministic reference VM for the perfected DM-vOmegaXi+ firmware."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from enum import IntEnum
from typing import Iterable, Sequence


class Status(IntEnum):
    OK = 0
    READY = 1
    COMMITTED = 2
    REJECTED = 3
    DRIFT = 0x101
    VERIFY_FAILED = 0x102
    BOUNDS_FAILED = 0x103
    BUDGET_FAILED = 0x104
    NONFINITE = 0x105
    AUTH_FAILED = 0x106


@dataclass(frozen=True)
class Config:
    max_retries: int = 4
    max_elements: int = 4096
    free_energy_limit: float = 0.25
    reconstruction_tolerance: float = 0.03125
    latent_bound: float = 1.0
    modulation_gain: float = 0.90
    beta: float = 0.05
    quantization_scale: int = 32767


@dataclass
class Receipt:
    tx_id: int
    outcome: str
    retries: int
    free_energy: float
    reconstruction_distance: float
    input_digest: str
    candidate_digest: str
    committed_digest: str
    status: int


@dataclass
class TransactionResult:
    status: Status
    output: list[float] | None
    receipt: Receipt


class DMVOmegaXiVM:
    """A bounded propose/validate/commit state machine.

    This VM is intentionally small. It demonstrates the firmware invariants rather
    than pretending to allocate an unbounded manifold or execute physical neural
    hardware instructions.
    """

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config()
        self.theta = 1.0
        self.omega_committed: list[int] = []
        self.receipts: list[Receipt] = []
        self.faults: list[dict[str, object]] = []
        self.tx_counter = 0

    @staticmethod
    def _digest(values: Sequence[float] | Sequence[int]) -> str:
        payload = json.dumps(list(values), separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _finite(values: Iterable[float]) -> bool:
        return all(math.isfinite(v) for v in values)

    def _encode(self, payload: Sequence[float]) -> list[float]:
        return [math.tanh(self.theta * value) for value in payload]

    def _free_energy(self, latent: Sequence[float]) -> float:
        if not latent:
            return 0.0
        quadratic = sum(value * value for value in latent) / len(latent)
        sparse = sum(abs(value) for value in latent) / len(latent)
        return quadratic + self.config.beta * sparse

    def _quantize(self, latent: Sequence[float]) -> list[int]:
        scale = self.config.quantization_scale
        bound = self.config.latent_bound
        return [
            int(round(max(-bound, min(bound, value)) * scale))
            for value in latent
        ]

    def _decode(self, quantized: Sequence[int]) -> list[float]:
        scale = self.config.quantization_scale
        epsilon = 1.0 / scale
        decoded: list[float] = []
        for raw in quantized:
            z = raw / scale
            z = max(-1.0 + epsilon, min(1.0 - epsilon, z))
            decoded.append(math.atanh(z) / self.theta)
        return decoded

    @staticmethod
    def _distance(lhs: Sequence[float], rhs: Sequence[float]) -> float:
        if len(lhs) != len(rhs):
            return math.inf
        if not lhs:
            return 0.0
        return sum(abs(a - b) for a, b in zip(lhs, rhs)) / len(lhs)

    def _policy_valid(self, payload: Sequence[float], candidate: Sequence[int], authorized: bool) -> bool:
        if not authorized:
            return False
        if len(payload) > self.config.max_elements:
            return False
        limit = self.config.quantization_scale
        return all(-limit <= value <= limit for value in candidate)

    def _record_fault(self, tx_id: int, status: Status, detail: str) -> None:
        self.faults.append({"tx_id": tx_id, "status": int(status), "detail": detail})

    def process(self, payload: Sequence[float], *, authorized: bool = True) -> TransactionResult:
        self.tx_counter += 1
        tx_id = self.tx_counter
        input_values = [float(value) for value in payload]
        input_digest = self._digest(input_values)
        committed_before = list(self.omega_committed)

        status = Status.OK
        retries = 0
        free_energy = math.inf
        distance = math.inf
        candidate: list[int] = []
        output: list[float] | None = None

        if len(input_values) > self.config.max_elements:
            status = Status.BUDGET_FAILED
        elif not self._finite(input_values):
            status = Status.NONFINITE
        elif not authorized:
            status = Status.AUTH_FAILED
        else:
            while retries <= self.config.max_retries:
                latent = self._encode(input_values)
                if not self._finite(latent):
                    status = Status.NONFINITE
                    break

                free_energy = self._free_energy(latent)
                if free_energy > self.config.free_energy_limit:
                    status = Status.DRIFT
                else:
                    candidate = self._quantize(latent)
                    output = self._decode(candidate)
                    distance = self._distance(input_values, output)

                    if not self._finite(output):
                        status = Status.NONFINITE
                    elif distance > self.config.reconstruction_tolerance:
                        status = Status.VERIFY_FAILED
                    elif not self._policy_valid(input_values, candidate, authorized):
                        status = Status.BOUNDS_FAILED
                    else:
                        status = Status.OK
                        break

                if retries == self.config.max_retries:
                    break
                retries += 1
                self.theta *= self.config.modulation_gain

        if status == Status.OK:
            self.omega_committed = list(candidate)
            outcome = "committed"
            final_status = Status.COMMITTED
        else:
            self.omega_committed = committed_before
            candidate = []
            output = None
            outcome = "rejected"
            final_status = Status.REJECTED
            self._record_fault(tx_id, status, status.name)

        receipt = Receipt(
            tx_id=tx_id,
            outcome=outcome,
            retries=retries,
            free_energy=free_energy,
            reconstruction_distance=distance,
            input_digest=input_digest,
            candidate_digest=self._digest(candidate),
            committed_digest=self._digest(self.omega_committed),
            status=int(status),
        )
        self.receipts.append(receipt)
        return TransactionResult(status=final_status, output=output, receipt=receipt)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "values",
        nargs="*",
        type=float,
        default=[0.10, -0.20, 0.30, -0.40],
        help="Input vector values.",
    )
    parser.add_argument("--unauthorized", action="store_true", help="Force authorization rejection.")
    args = parser.parse_args()

    vm = DMVOmegaXiVM()
    result = vm.process(args.values, authorized=not args.unauthorized)
    print(json.dumps({
        "status": result.status.name,
        "output": result.output,
        "receipt": asdict(result.receipt),
        "faults": vm.faults,
    }, indent=2, sort_keys=True))
    return 0 if result.status == Status.COMMITTED else 2


if __name__ == "__main__":
    raise SystemExit(main())
