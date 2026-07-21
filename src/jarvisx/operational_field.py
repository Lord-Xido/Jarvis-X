"""Operational tetration field with committed sparse latent state."""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import replace
from typing import Dict, Mapping, Optional, Tuple

from .tetration_field import (
    BrickState,
    FieldStepMetrics,
    Observation,
    TetrationAddress,
    TetrationFieldAutomaton,
)


class OperationalTetrationFieldAutomaton(TetrationFieldAutomaton):
    """Extend the brick field so B, Z and Omega commit as one transaction.

    The base field calculates latent states while processing each frontier brick.
    This operational wrapper reconstructs the committed latent state for every
    retained brick, validates it, binds it into the journal, and restores the
    prior field transaction if latent sealing fails.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.latent_repository: Dict[TetrationAddress, Tuple[float, ...]] = {}

    def _evolved_latent(self, state: BrickState) -> Tuple[float, ...]:
        latent = self.network.encode(state.values)
        conditioned = self.network.condition_with_omega(latent, state.omega)
        expert, _ = self.network.route(conditioned)
        if expert != state.expert_index:
            raise ValueError("committed expert index does not match latent router")
        return tuple(
            math.tanh(
                conditioned[index]
                + self.network._dot(self.network.experts[expert][index], conditioned)
                + self.network.expert_bias[expert][index]
            )
            for index in range(self.network.latent_dim)
        )

    def _build_latent_repository(
        self, states: Mapping[TetrationAddress, BrickState]
    ) -> Dict[TetrationAddress, Tuple[float, ...]]:
        repository = {
            address: self._evolved_latent(state)
            for address, state in sorted(states.items())
        }
        for latent in repository.values():
            if len(latent) != self.network.latent_dim:
                raise ValueError("latent state has the wrong dimension")
            if not all(math.isfinite(value) for value in latent):
                raise ValueError("latent repository contains a non-finite value")
        return repository

    def _seal_latents(
        self, repository: Mapping[TetrationAddress, Tuple[float, ...]]
    ) -> str:
        digest = hashlib.sha256(bytes.fromhex(self.journal_hash))
        digest.update(b"JARVIS-X-SPARSE-Z-V1")
        for address in sorted(repository):
            latent = repository[address]
            digest.update(address.canonical_bytes())
            digest.update(struct.pack(">I", len(latent)))
            for value in latent:
                digest.update(struct.pack(">d", value))
        return digest.hexdigest()

    def step(
        self,
        injections: Optional[Mapping[TetrationAddress, Observation]] = None,
    ) -> FieldStepMetrics:
        previous_directory = self.directory
        previous_cycle = self.cycle
        previous_hash = self.journal_hash
        previous_metrics = self.last_metrics
        previous_latents = dict(self.latent_repository)

        metrics = super().step(injections)
        if not metrics.committed:
            return metrics

        try:
            candidate_latents = self._build_latent_repository(self.directory.to_dict())
            sealed_hash = self._seal_latents(candidate_latents)
        except (TypeError, ValueError, OverflowError) as exc:
            self.directory = previous_directory
            self.cycle = previous_cycle
            self.journal_hash = previous_hash
            self.last_metrics = previous_metrics
            self.latent_repository = previous_latents
            self.last_metrics = self._metrics(
                committed=False,
                frontier=metrics.frontier_bricks,
                mse=metrics.reconstruction_mse,
                energy=metrics.relative_energy,
                reason="latent commit failed: {}".format(exc),
            )
            return self.last_metrics

        self.latent_repository = candidate_latents
        self.journal_hash = sealed_hash
        self.last_metrics = replace(metrics, journal_hash=sealed_hash)
        return self.last_metrics

    def latent_state(
        self, address: TetrationAddress
    ) -> Optional[Tuple[float, ...]]:
        return self.latent_repository.get(address)

    def snapshot(self) -> Dict[str, object]:
        state = super().snapshot()
        state.update(
            {
                "latent_repository_entries": len(self.latent_repository),
                "physical_state": ["active_frontier", "B", "Z", "Omega"],
                "memory_model": "O(M_t * (192 + d + 192)) for B, Z and Omega",
                "journal_hash": self.journal_hash,
                "last_metrics": self.last_metrics.to_dict(),
            }
        )
        return state
