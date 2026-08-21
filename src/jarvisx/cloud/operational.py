"""Operational service layer combining virtual routing with durable state."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from .extent import HierarchicalAddress, SymbolicParameterExtent
from .multimodal import MediaEnvelope
from .persistence import JobRecord, SQLiteStateStore
from .routing import ShardCoordinate, SpatialShardRouter


@dataclass
class OperationalHyperCloud:
    extent: SymbolicParameterExtent = field(default_factory=SymbolicParameterExtent)
    router: SpatialShardRouter = field(default_factory=SpatialShardRouter)
    state: SQLiteStateStore | None = None

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
        if self.state.media_record(digest) is None:
            raise KeyError(digest)
        return self.state.create_job(namespace, "codec_roundtrip", {"media_sha256": digest})

    def enqueue_chat(self, namespace: str, prompt: str, system: str | None = None) -> JobRecord:
        if not prompt.strip():
            raise ValueError("prompt must not be empty")
        assert self.state is not None
        return self.state.create_job(
            namespace,
            "chat",
            {"prompt": prompt, "system": system},
        )

    def job(self, job_id: str) -> JobRecord | None:
        assert self.state is not None
        return self.state.get_job(job_id)

    def describe(self, *, backend_name: str | None = None) -> dict[str, object]:
        assert self.state is not None
        external_model_backend = bool(os.getenv("JARVISX_MODEL_BASE_URL", "").strip())
        return {
            "runtime": "jarvisx-3d-hypercloud",
            "status": "operational-reference",
            "virtual_parameter_extent": self.extent.metadata(),
            "deployment_lattice": {
                "shape": self.router.shape,
                "shards": self.router.shard_count,
            },
            "materialized_parameters": self.state.parameter_count(),
            "ingested_media_objects": self.state.media_count(),
            "jobs": self.state.job_counts(),
            "chat_backend": backend_name,
            "claims": {
                "dense_parameter_allocation": False,
                "durable_local_state": True,
                "persistent_job_queue": True,
                "lossless_multimodal_codec": True,
                "openai_compatible_model_adapter": True,
                "external_model_backend_configured": external_model_backend,
                "distributed_accelerator_backend": False,
                "deterministic_3d_routing": True,
            },
        }
