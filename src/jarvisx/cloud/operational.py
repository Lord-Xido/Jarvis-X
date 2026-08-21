"""Operational service layer combining virtual routing with durable state."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path

from .extent import HierarchicalAddress, SymbolicParameterExtent
from .multimodal import MediaEnvelope
from .persistence import JobRecord, SQLiteStateStore
from .routing import ShardCoordinate, SpatialShardRouter
from .topology import PlacementDecision, TopologyScheduler


@dataclass
class OperationalHyperCloud:
    extent: SymbolicParameterExtent = field(default_factory=SymbolicParameterExtent)
    router: SpatialShardRouter = field(default_factory=SpatialShardRouter)
    state: SQLiteStateStore | None = None
    scheduler: TopologyScheduler = field(default_factory=TopologyScheduler)

    def __post_init__(self) -> None:
        if self.state is None:
            self.state = SQLiteStateStore(":memory:", ".jarvisx-hypercloud-objects")

    @classmethod
    def from_environment(cls) -> "OperationalHyperCloud":
        database = os.getenv("JARVISX_STATE_DB", "/tmp/jarvisx-hypercloud/state.db")
        object_root = os.getenv("JARVISX_OBJECT_ROOT", "/tmp/jarvisx-hypercloud/objects")
        shape_text = os.getenv("JARVISX_LATTICE_SHAPE", "16,16,16")
        parts = tuple(int(part.strip()) for part in shape_text.split(","))
        if len(parts) != 3 or any(part <= 0 for part in parts):
            raise RuntimeError("JARVISX_LATTICE_SHAPE must contain three positive integers")
        Path(database).parent.mkdir(parents=True, exist_ok=True)
        Path(object_root).mkdir(parents=True, exist_ok=True)
        return cls(
            router=SpatialShardRouter(shape=parts),
            state=SQLiteStateStore(database, object_root),
        )

    def route(
        self,
        *,
        namespace: str,
        modality: str,
        address: HierarchicalAddress,
    ) -> ShardCoordinate:
        if not namespace:
            raise ValueError("namespace must not be empty")
        address.validate(radix=self.extent.address_radix)
        return self.router.route(namespace=namespace, modality=modality, address=address)

    def _address_for_key(self, key: str) -> HierarchicalAddress:
        digest = sha256(key.encode("utf-8")).digest()
        digits = tuple(
            int.from_bytes(digest[offset : offset + 4], "big") % self.extent.address_radix
            for offset in range(0, 24, 4)
        )
        address = HierarchicalAddress(digits)
        address.validate(radix=self.extent.address_radix)
        return address

    def _job_target(self, *, namespace: str, modality: str, key: str) -> ShardCoordinate:
        return self.route(
            namespace=namespace,
            modality=modality,
            address=self._address_for_key(key),
        )

    def set_parameter(self, namespace: str, address: HierarchicalAddress, value: float) -> None:
        address.validate(radix=self.extent.address_radix)
        assert self.state is not None
        self.state.set_parameter(namespace, address.canonical(), value)

    def get_parameter(self, namespace: str, address: HierarchicalAddress) -> float | None:
        address.validate(radix=self.extent.address_radix)
        assert self.state is not None
        return self.state.get_parameter(namespace, address.canonical())

    def ingest(self, namespace: str, media: MediaEnvelope) -> dict[str, str | int | float]:
        if not namespace:
            raise ValueError("namespace must not be empty")
        assert self.state is not None
        return self.state.put_media(namespace, media)

    def enqueue_codec(self, namespace: str, digest: str) -> JobRecord:
        assert self.state is not None
        record = self.state.media_record(namespace, digest)
        if record is None:
            raise KeyError(digest)
        modality = str(record["kind"])
        target = self._job_target(namespace=namespace, modality=modality, key=digest)
        return self.state.create_job(
            namespace,
            "codec_roundtrip",
            {"media_sha256": digest},
            target=target,
        )

    def enqueue_chat(self, namespace: str, prompt: str, system: str | None = None) -> JobRecord:
        if not prompt.strip():
            raise ValueError("prompt must not be empty")
        assert self.state is not None
        key = f"{system or ''}\x1f{prompt}"
        target = self._job_target(namespace=namespace, modality="text", key=key)
        return self.state.create_job(
            namespace,
            "chat",
            {"prompt": prompt, "system": system},
            target=target,
        )

    def job(self, job_id: str) -> JobRecord | None:
        assert self.state is not None
        return self.state.get_job(job_id)

    def placement_preview(self, job_id: str, *, worker_ttl_seconds: float = 30.0) -> PlacementDecision | None:
        assert self.state is not None
        job = self.state.get_job(job_id)
        if job is None or job.target is None:
            return None
        capability = "chat" if job.operation == "chat" else "codec"
        return self.scheduler.choose(
            target=job.target,
            workers=self.state.active_workers(ttl_seconds=worker_ttl_seconds),
            capability=capability,
        )

    def describe(self, *, backend_name: str | None = None) -> dict[str, object]:
        assert self.state is not None
        external_model_backend = bool(os.getenv("JARVISX_MODEL_BASE_URL", "").strip())
        worker_ttl = float(os.getenv("JARVISX_WORKER_TTL_SECONDS", "30"))
        workers = self.state.active_workers(ttl_seconds=worker_ttl)
        return {
            "runtime": "jarvisx-3d-hypercloud",
            "status": "operational-permeated",
            "virtual_parameter_extent": self.extent.metadata(),
            "deployment_lattice": {
                "shape": self.router.shape,
                "shards": self.router.shard_count,
                "active_workers": len(workers),
            },
            "materialized_parameters": self.state.parameter_count(),
            "ingested_media_objects": self.state.media_count(),
            "jobs": self.state.job_counts(),
            "chat_backend": backend_name,
            "claims": {
                "dense_parameter_allocation": False,
                "durable_local_state": True,
                "persistent_job_queue": True,
                "leased_worker_execution": True,
                "abandoned_job_recovery": True,
                "topology_aware_3d_placement": True,
                "lossless_multimodal_codec": True,
                "openai_compatible_model_adapter": True,
                "external_model_backend_configured": external_model_backend,
                "distributed_accelerator_backend": False,
                "deterministic_3d_routing": True,
            },
        }
