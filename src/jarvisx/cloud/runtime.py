"""Sparse multimodal HyperCloud runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock

from .extent import HierarchicalAddress, SymbolicParameterExtent
from .multimodal import MediaEnvelope
from .routing import ShardCoordinate, SpatialShardRouter


@dataclass
class SparseParameterStore:
    """Thread-safe materialized subset of a symbolic parameter space."""

    extent: SymbolicParameterExtent = field(default_factory=SymbolicParameterExtent)
    _values: dict[tuple[str, str], float] = field(default_factory=dict, init=False, repr=False)
    _lock: RLock = field(default_factory=RLock, init=False, repr=False)

    def set(self, namespace: str, address: HierarchicalAddress, value: float) -> None:
        address.validate(radix=self.extent.address_radix)
        if not namespace:
            raise ValueError("namespace must not be empty")
        key = (namespace, address.canonical())
        with self._lock:
            self._values[key] = float(value)

    def get(self, namespace: str, address: HierarchicalAddress) -> float | None:
        address.validate(radix=self.extent.address_radix)
        key = (namespace, address.canonical())
        with self._lock:
            return self._values.get(key)

    @property
    def materialized_parameters(self) -> int:
        with self._lock:
            return len(self._values)


@dataclass
class HyperCloudRuntime:
    """Reference control-plane runtime for sparse 3D multimodal execution.

    It provides deterministic routing and bounded state materialization. Neural
    kernels, distributed object stores, accelerator execution and model-specific
    encoders/decoders attach behind this interface in later integration tracks.
    """

    extent: SymbolicParameterExtent = field(default_factory=SymbolicParameterExtent)
    router: SpatialShardRouter = field(default_factory=SpatialShardRouter)
    store: SparseParameterStore = field(init=False)
    _media: dict[str, dict[str, str | int]] = field(default_factory=dict, init=False, repr=False)
    _media_lock: RLock = field(default_factory=RLock, init=False, repr=False)

    def __post_init__(self) -> None:
        self.store = SparseParameterStore(extent=self.extent)

    def route(
        self,
        *,
        namespace: str,
        modality: str,
        address: HierarchicalAddress,
    ) -> ShardCoordinate:
        address.validate(radix=self.extent.address_radix)
        return self.router.route(namespace=namespace, modality=modality, address=address)

    def ingest(self, *, namespace: str, media: MediaEnvelope) -> dict[str, str | int]:
        if not namespace:
            raise ValueError("namespace must not be empty")
        record = {"namespace": namespace, **media.descriptor()}
        with self._media_lock:
            self._media[media.digest] = record
        return record

    def media_record(self, digest: str) -> dict[str, str | int] | None:
        with self._media_lock:
            record = self._media.get(digest)
            return dict(record) if record is not None else None

    def describe(self) -> dict[str, object]:
        with self._media_lock:
            media_objects = len(self._media)
        return {
            "runtime": "jarvisx-3d-hypercloud",
            "status": "reference-control-plane",
            "virtual_parameter_extent": self.extent.metadata(),
            "deployment_lattice": {
                "shape": self.router.shape,
                "shards": self.router.shard_count,
            },
            "materialized_parameters": self.store.materialized_parameters,
            "ingested_media_objects": media_objects,
            "claims": {
                "dense_parameter_allocation": False,
                "distributed_accelerator_backend": False,
                "multimodal_envelope_support": True,
                "deterministic_3d_routing": True,
            },
        }
