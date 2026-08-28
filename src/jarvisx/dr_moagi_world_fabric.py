"""Sparse worldwide 3D data/compute fabric for the Dr Moagi research stack.

The astronomically large proposed lattice is treated as a *virtual, generative
namespace*. Only finite symbolic octree paths that are actually used are
materialized.

Three identities remain independent:

* logical address: a finite 3-bit-octant path in a virtual 3D namespace;
* content identity: a SHA-256 digest over canonical bytes;
* physical placement: a mutable region selected by a bounded locality planner.

Authoritative bytes and learned/derived representations are also separated.
The exact plane is immutable and self-verifying; the latent plane is derived
and replaceable without changing source truth.

This is a bounded reference runtime, not a claim of globally distributed
production storage. Network consensus, durable object storage, geographic
replication and learned embeddings remain explicit adapter boundaries.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from enum import Enum
from typing import Callable, Mapping, Sequence


VectorEncoder = Callable[[bytes], Sequence[float]]


class ConsistencyClass(str, Enum):
    """Consistency contract attached to one logical cell."""

    IMMUTABLE = "c0-immutable"
    LOCAL_MUTABLE = "c1-local-mutable"
    GLOBAL_AUTHORITATIVE = "c2-global-authoritative"


@dataclass(frozen=True, order=True)
class SymbolicAddress3D:
    """Finite octree path inside an effectively unbounded virtual 3D space.

    Each path element is one octant selector in ``[0, 7]``. No gigantic
    absolute integer coordinate is required. Arbitrary finite depth is
    supported by deriving additional octants from SHAKE-256 output.
    """

    path: tuple[int, ...]

    def __post_init__(self) -> None:
        for octant in self.path:
            if isinstance(octant, bool) or not isinstance(octant, int):
                raise TypeError("octree path elements must be integers")
            if not 0 <= octant <= 7:
                raise ValueError("octree path elements must be in [0, 7]")

    @property
    def depth(self) -> int:
        return len(self.path)

    def child(self, octant: int) -> "SymbolicAddress3D":
        return SymbolicAddress3D(self.path + (octant,))

    def parent(self) -> "SymbolicAddress3D | None":
        if not self.path:
            return None
        return SymbolicAddress3D(self.path[:-1])

    def is_ancestor_of(self, other: "SymbolicAddress3D") -> bool:
        return other.path[: self.depth] == self.path

    def to_text(self) -> str:
        return "dm3:/" + "/".join(str(value) for value in self.path)

    @classmethod
    def from_text(cls, value: str) -> "SymbolicAddress3D":
        if not value.startswith("dm3:/"):
            raise ValueError("symbolic address must start with 'dm3:/'")
        suffix = value[len("dm3:/") :]
        if not suffix:
            return cls(())
        return cls(tuple(int(part) for part in suffix.split("/")))

    @classmethod
    def from_cid(cls, cid: str, *, depth: int = 24) -> "SymbolicAddress3D":
        if isinstance(depth, bool) or not isinstance(depth, int) or depth <= 0:
            raise ValueError("depth must be a positive integer")
        prefix = "sha256:"
        if not cid.startswith(prefix):
            raise ValueError("CID must use the sha256: scheme")
        try:
            digest = bytes.fromhex(cid[len(prefix) :])
        except ValueError as exc:
            raise ValueError("CID contains invalid hexadecimal digest") from exc
        if len(digest) != hashlib.sha256().digest_size:
            raise ValueError("CID must contain a full SHA-256 digest")

        byte_count = math.ceil((depth * 3) / 8)
        stream = hashlib.shake_256(digest).digest(byte_count)
        path: list[int] = []
        accumulator = 0
        bits = 0
        for byte in stream:
            accumulator = (accumulator << 8) | byte
            bits += 8
            while bits >= 3 and len(path) < depth:
                bits -= 3
                path.append((accumulator >> bits) & 0b111)
                accumulator &= (1 << bits) - 1 if bits else 0
            if len(path) == depth:
                break
        return cls(tuple(path))


@dataclass(frozen=True)
class ProvenanceRecord:
    """Immutable observation describing where content was encountered."""

    source: str
    observed_at_ns: int
    source_uri: str | None = None
    media_type: str | None = None
    license: str | None = None

    def __post_init__(self) -> None:
        if not self.source:
            raise ValueError("source must be non-empty")
        if isinstance(self.observed_at_ns, bool) or not isinstance(self.observed_at_ns, int):
            raise TypeError("observed_at_ns must be an integer")
        if self.observed_at_ns < 0:
            raise ValueError("observed_at_ns must be non-negative")

    @classmethod
    def now(
        cls,
        source: str,
        *,
        source_uri: str | None = None,
        media_type: str | None = None,
        license: str | None = None,
    ) -> "ProvenanceRecord":
        return cls(
            source=source,
            observed_at_ns=time.time_ns(),
            source_uri=source_uri,
            media_type=media_type,
            license=license,
        )


@dataclass(frozen=True)
class RegionSpec:
    """One physical-placement candidate in an abstract latency geometry."""

    region_id: str
    x: float
    y: float
    z: float
    capacity_cells: int = 1_000_000

    def __post_init__(self) -> None:
        if not self.region_id:
            raise ValueError("region_id must be non-empty")
        if isinstance(self.capacity_cells, bool) or not isinstance(self.capacity_cells, int):
            raise TypeError("capacity_cells must be an integer")
        if self.capacity_cells <= 0:
            raise ValueError("capacity_cells must be positive")
        if not all(math.isfinite(float(value)) for value in (self.x, self.y, self.z)):
            raise ValueError("region coordinates must be finite")

    def distance_to(self, other: "RegionSpec") -> float:
        return math.sqrt(
            (self.x - other.x) ** 2
            + (self.y - other.y) ** 2
            + (self.z - other.z) ** 2
        )


@dataclass(frozen=True)
class WorldCell:
    """Metadata twin joining exact truth, derived representation and placement."""

    address: SymbolicAddress3D
    cid: str
    latent: tuple[float, ...]
    provenance: tuple[ProvenanceRecord, ...]
    consistency: ConsistencyClass
    region_id: str
    byte_length: int
    version: int = 1

    @property
    def provenance_count(self) -> int:
        return len(self.provenance)


class ExactContentStore:
    """Thread-safe immutable in-memory reference object plane."""

    def __init__(self) -> None:
        self._objects: dict[str, bytes] = {}
        self._lock = threading.RLock()

    @staticmethod
    def cid_for(data: bytes) -> str:
        if not isinstance(data, bytes):
            raise TypeError("exact content must be bytes")
        return "sha256:" + hashlib.sha256(data).hexdigest()

    def put(self, data: bytes) -> str:
        cid = self.cid_for(data)
        with self._lock:
            existing = self._objects.get(cid)
            if existing is not None and existing != data:
                raise RuntimeError("cryptographic identity collision detected")
            self._objects.setdefault(cid, data)
        return cid

    def get(self, cid: str, *, verify: bool = True) -> bytes:
        with self._lock:
            data = self._objects[cid]
        if verify and self.cid_for(data) != cid:
            raise RuntimeError("content-integrity verification failed")
        return data

    def verify(self, cid: str) -> bool:
        try:
            return self.cid_for(self.get(cid, verify=False)) == cid
        except KeyError:
            return False

    @property
    def object_count(self) -> int:
        with self._lock:
            return len(self._objects)

    @property
    def resident_bytes(self) -> int:
        with self._lock:
            return sum(len(value) for value in self._objects.values())


class ByteSketchEncoder:
    """Dependency-free deterministic content sketch.

    This is intentionally *not* advertised as a learned semantic embedding. It
    provides a replaceable latent-plane contract and deterministic test fixture.
    Production experimentation should inject a workload-validated multimodal
    embedding model through ``WorldFabric(encoder=...)``.
    """

    def __init__(self, dimensions: int = 32) -> None:
        if isinstance(dimensions, bool) or not isinstance(dimensions, int) or dimensions <= 0:
            raise ValueError("dimensions must be a positive integer")
        self.dimensions = dimensions

    def __call__(self, data: bytes) -> tuple[float, ...]:
        if not isinstance(data, bytes):
            raise TypeError("encoder input must be bytes")
        vector = [0.0] * self.dimensions
        if not data:
            return tuple(vector)
        for index, byte in enumerate(data):
            vector[byte % self.dimensions] += 1.0
            if index:
                mixed = ((data[index - 1] << 8) | byte) % self.dimensions
                vector[mixed] += 0.25
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return tuple(vector)
        return tuple(value / norm for value in vector)


class HashChainMetadataLedger:
    """Append-only canonical JSON hash chain for metadata mutations."""

    def __init__(self) -> None:
        self._records: list[dict[str, object]] = []
        self._head = "0" * 64
        self._lock = threading.RLock()

    @property
    def head(self) -> str:
        with self._lock:
            return self._head

    @property
    def length(self) -> int:
        with self._lock:
            return len(self._records)

    def append(self, event: Mapping[str, object]) -> str:
        with self._lock:
            body = {
                "index": len(self._records),
                "previous": self._head,
                "event": dict(event),
            }
            canonical = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
            digest = hashlib.sha256(canonical).hexdigest()
            record = {**body, "hash": digest}
            self._records.append(record)
            self._head = digest
            return digest

    def verify(self) -> bool:
        with self._lock:
            previous = "0" * 64
            for index, record in enumerate(self._records):
                body = {
                    "index": index,
                    "previous": previous,
                    "event": record["event"],
                }
                canonical = json.dumps(
                    body, sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
                digest = hashlib.sha256(canonical).hexdigest()
                if record.get("previous") != previous or record.get("hash") != digest:
                    return False
                previous = digest
            return previous == self._head


class AdaptivePlacementPlanner:
    """Bounded deterministic physical placement and self-folding locality planner."""

    def __init__(self, regions: Sequence[RegionSpec], *, load_weight: float = 0.25) -> None:
        if not regions:
            raise ValueError("at least one region is required")
        if load_weight < 0.0 or not math.isfinite(load_weight):
            raise ValueError("load_weight must be finite and non-negative")
        self.regions = {region.region_id: region for region in regions}
        if len(self.regions) != len(regions):
            raise ValueError("region IDs must be unique")
        self.load_weight = load_weight
        self._placement: dict[SymbolicAddress3D, str] = {}
        self._loads = {region.region_id: 0 for region in regions}
        self._interactions: dict[tuple[SymbolicAddress3D, SymbolicAddress3D], float] = {}
        self._lock = threading.RLock()

    def region_for(self, address: SymbolicAddress3D) -> str:
        with self._lock:
            return self._placement[address]

    def assign(self, address: SymbolicAddress3D) -> str:
        with self._lock:
            existing = self._placement.get(address)
            if existing is not None:
                return existing
            candidates = [
                region
                for region in self.regions.values()
                if self._loads[region.region_id] < region.capacity_cells
            ]
            if not candidates:
                raise RuntimeError("all placement regions are at capacity")
            region = min(
                candidates,
                key=lambda item: (
                    self._loads[item.region_id] / item.capacity_cells,
                    item.region_id,
                ),
            )
            self._placement[address] = region.region_id
            self._loads[region.region_id] += 1
            return region.region_id

    def release(self, address: SymbolicAddress3D) -> None:
        """Release an uncommitted reservation after a failed staged ingest."""
        with self._lock:
            region_id = self._placement.pop(address, None)
            if region_id is None:
                return
            self._loads[region_id] -= 1
            for edge in tuple(self._interactions):
                if address in edge:
                    del self._interactions[edge]

    def record_interaction(
        self,
        left: SymbolicAddress3D,
        right: SymbolicAddress3D,
        *,
        weight: float = 1.0,
    ) -> None:
        if left == right:
            return
        if weight <= 0.0 or not math.isfinite(weight):
            raise ValueError("interaction weight must be finite and positive")
        edge = (left, right) if left < right else (right, left)
        with self._lock:
            self._interactions[edge] = self._interactions.get(edge, 0.0) + weight

    def _communication_cost(self, address: SymbolicAddress3D, candidate_id: str) -> float:
        candidate = self.regions[candidate_id]
        cost = 0.0
        for (left, right), weight in self._interactions.items():
            if address == left:
                peer = right
            elif address == right:
                peer = left
            else:
                continue
            peer_region_id = self._placement.get(peer)
            if peer_region_id is None:
                continue
            cost += weight * candidate.distance_to(self.regions[peer_region_id])
        normalized_load = self._loads[candidate_id] / candidate.capacity_cells
        return cost + self.load_weight * normalized_load

    def rebalance(
        self,
        *,
        max_moves: int = 16,
        min_gain: float = 1.0e-9,
    ) -> tuple[tuple[str, str, str, float], ...]:
        if isinstance(max_moves, bool) or not isinstance(max_moves, int) or max_moves < 0:
            raise ValueError("max_moves must be a non-negative integer")
        if min_gain < 0.0 or not math.isfinite(min_gain):
            raise ValueError("min_gain must be finite and non-negative")
        moves: list[tuple[str, str, str, float]] = []
        with self._lock:
            weighted_addresses = sorted(
                self._placement,
                key=lambda address: (-self._incident_weight(address), address),
            )
            for address in weighted_addresses:
                if len(moves) >= max_moves:
                    break
                current_id = self._placement[address]
                current_cost = self._communication_cost(address, current_id)
                candidates = [
                    region_id
                    for region_id, region in self.regions.items()
                    if region_id == current_id
                    or self._loads[region_id] < region.capacity_cells
                ]
                best_id = min(
                    candidates,
                    key=lambda region_id: (self._communication_cost(address, region_id), region_id),
                )
                best_cost = self._communication_cost(address, best_id)
                gain = current_cost - best_cost
                if best_id == current_id or gain <= min_gain:
                    continue
                self._loads[current_id] -= 1
                self._loads[best_id] += 1
                self._placement[address] = best_id
                moves.append((address.to_text(), current_id, best_id, gain))
        return tuple(moves)

    def _incident_weight(self, address: SymbolicAddress3D) -> float:
        return sum(weight for edge, weight in self._interactions.items() if address in edge)

    def snapshot_loads(self) -> dict[str, int]:
        with self._lock:
            return dict(self._loads)


class BoundedScheduler:
    """Map many logical tasks onto a bounded pool of OS threads."""

    def __init__(self, *, workers: int = 4, max_in_flight: int | None = None) -> None:
        if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
            raise ValueError("workers must be a positive integer")
        limit = max_in_flight if max_in_flight is not None else workers * 4
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < workers:
            raise ValueError("max_in_flight must be an integer >= workers")
        self.workers = workers
        self.max_in_flight = limit
        self._executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="dm-world")
        self._slots = threading.BoundedSemaphore(limit)
        self._closed = False
        self._lock = threading.Lock()
        self.submitted = 0
        self.completed = 0

    def submit(
        self,
        function: Callable[..., object],
        /,
        *args: object,
        **kwargs: object,
    ) -> Future[object]:
        with self._lock:
            if self._closed:
                raise RuntimeError("scheduler is closed")
        if not self._slots.acquire(blocking=False):
            raise RuntimeError("scheduler in-flight limit reached")
        with self._lock:
            self.submitted += 1

        def run() -> object:
            try:
                return function(*args, **kwargs)
            finally:
                with self._lock:
                    self.completed += 1
                self._slots.release()

        try:
            return self._executor.submit(run)
        except BaseException:
            self._slots.release()
            raise

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._executor.shutdown(wait=True, cancel_futures=False)

    def __enter__(self) -> "BoundedScheduler":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


class WorldFabric:
    """Reference dual-plane sparse 3D worldwide data/compute fabric."""

    def __init__(
        self,
        *,
        regions: Sequence[RegionSpec] | None = None,
        encoder: VectorEncoder | None = None,
        address_depth: int = 24,
        workers: int = 4,
        max_in_flight: int | None = None,
    ) -> None:
        if isinstance(address_depth, bool) or not isinstance(address_depth, int) or address_depth <= 0:
            raise ValueError("address_depth must be a positive integer")
        default_regions = (
            RegionSpec("africa", 0.0, 0.0, 0.0),
            RegionSpec("europe", 1.0, 0.0, 0.0),
            RegionSpec("americas", 0.0, 1.0, 0.0),
        )
        self.address_depth = address_depth
        self.exact = ExactContentStore()
        self.encoder: VectorEncoder = encoder or ByteSketchEncoder()
        self.ledger = HashChainMetadataLedger()
        self.placement = AdaptivePlacementPlanner(tuple(regions or default_regions))
        self.scheduler = BoundedScheduler(workers=workers, max_in_flight=max_in_flight)
        self._cells: dict[SymbolicAddress3D, WorldCell] = {}
        self._address_by_cid: dict[str, SymbolicAddress3D] = {}
        self._lock = threading.RLock()

    def _unique_address(self, cid: str) -> SymbolicAddress3D:
        depth = self.address_depth
        while True:
            address = SymbolicAddress3D.from_cid(cid, depth=depth)
            existing = self._cells.get(address)
            if existing is None or existing.cid == cid:
                return address
            depth += 1

    def _validated_latent(self, data: bytes) -> tuple[float, ...]:
        latent = tuple(float(value) for value in self.encoder(data))
        if not latent or not all(math.isfinite(value) for value in latent):
            raise ValueError("encoder must return a non-empty finite vector")
        return latent

    def _attach_provenance(
        self,
        cell: WorldCell,
        provenance: ProvenanceRecord,
    ) -> WorldCell:
        if provenance in cell.provenance:
            return cell
        updated = replace(
            cell,
            provenance=cell.provenance + (provenance,),
            version=cell.version + 1,
        )
        self._cells[cell.address] = updated
        self.ledger.append(
            {
                "type": "provenance-link",
                "address": cell.address.to_text(),
                "cid": cell.cid,
                "provenance": asdict(provenance),
                "version": updated.version,
            }
        )
        return updated

    def ingest(
        self,
        data: bytes,
        *,
        provenance: ProvenanceRecord,
        consistency: ConsistencyClass = ConsistencyClass.IMMUTABLE,
    ) -> WorldCell:
        if not isinstance(data, bytes):
            raise TypeError("data must be bytes")
        if not isinstance(provenance, ProvenanceRecord):
            raise TypeError("provenance must be a ProvenanceRecord")
        if not isinstance(consistency, ConsistencyClass):
            raise TypeError("consistency must be a ConsistencyClass")

        cid = ExactContentStore.cid_for(data)
        with self._lock:
            existing_address = self._address_by_cid.get(cid)
            if existing_address is not None:
                return self._attach_provenance(self._cells[existing_address], provenance)

            # Stage all fallible derived work before exact bytes become resident.
            latent = self._validated_latent(data)
            address = self._unique_address(cid)
            region_id = self.placement.assign(address)
            try:
                stored_cid = self.exact.put(data)
                if stored_cid != cid:
                    raise RuntimeError("content identity changed during staged ingest")
                cell = WorldCell(
                    address=address,
                    cid=cid,
                    latent=latent,
                    provenance=(provenance,),
                    consistency=consistency,
                    region_id=region_id,
                    byte_length=len(data),
                )
                self._cells[address] = cell
                self._address_by_cid[cid] = address
                self.ledger.append(
                    {
                        "type": "ingest",
                        "address": address.to_text(),
                        "cid": cid,
                        "bytes": len(data),
                        "consistency": consistency.value,
                        "region": region_id,
                        "provenance": asdict(provenance),
                    }
                )
                return cell
            except BaseException:
                self.placement.release(address)
                self._cells.pop(address, None)
                self._address_by_cid.pop(cid, None)
                raise

    def submit_ingest(
        self,
        data: bytes,
        *,
        provenance: ProvenanceRecord,
        consistency: ConsistencyClass = ConsistencyClass.IMMUTABLE,
    ) -> Future[object]:
        return self.scheduler.submit(
            self.ingest,
            data,
            provenance=provenance,
            consistency=consistency,
        )

    def cell(self, address: SymbolicAddress3D) -> WorldCell:
        with self._lock:
            return self._cells[address]

    def cell_for_cid(self, cid: str) -> WorldCell:
        with self._lock:
            return self._cells[self._address_by_cid[cid]]

    def provenance_for_cid(self, cid: str) -> tuple[ProvenanceRecord, ...]:
        return self.cell_for_cid(cid).provenance

    def retrieve(self, address: SymbolicAddress3D) -> bytes:
        cell = self.cell(address)
        return self.exact.get(cell.cid, verify=True)

    @staticmethod
    def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
        if len(left) != len(right):
            raise ValueError("latent vectors have incompatible dimensions")
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)

    def nearest(self, query: bytes, *, k: int = 5) -> tuple[tuple[WorldCell, float], ...]:
        if not isinstance(query, bytes):
            raise TypeError("query must be bytes")
        if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer")
        latent = self._validated_latent(query)
        with self._lock:
            scored = [(cell, self._cosine(latent, cell.latent)) for cell in self._cells.values()]
        scored.sort(key=lambda item: (-item[1], item[0].address))
        return tuple(scored[:k])

    def record_interaction(
        self,
        left: SymbolicAddress3D,
        right: SymbolicAddress3D,
        *,
        weight: float = 1.0,
    ) -> None:
        with self._lock:
            if left not in self._cells or right not in self._cells:
                raise KeyError("interaction endpoints must be materialized cells")
        self.placement.record_interaction(left, right, weight=weight)

    def fold_placement(
        self,
        *,
        max_moves: int = 16,
    ) -> tuple[tuple[str, str, str, float], ...]:
        moves = self.placement.rebalance(max_moves=max_moves)
        if not moves:
            return moves
        with self._lock:
            for address_text, previous, current, gain in moves:
                address = SymbolicAddress3D.from_text(address_text)
                cell = self._cells[address]
                self._cells[address] = replace(
                    cell,
                    region_id=current,
                    version=cell.version + 1,
                )
                self.ledger.append(
                    {
                        "type": "placement-fold",
                        "address": address_text,
                        "cid": cell.cid,
                        "from": previous,
                        "to": current,
                        "gain": gain,
                    }
                )
        return moves

    def verify(self) -> bool:
        with self._lock:
            cells = tuple(self._cells.values())
            indexes_match = all(
                self._address_by_cid.get(cell.cid) == cell.address for cell in cells
            ) and len(self._address_by_cid) == len(cells)
        return (
            indexes_match
            and self.ledger.verify()
            and all(self.exact.verify(cell.cid) for cell in cells)
        )

    def stats(self) -> dict[str, object]:
        with self._lock:
            cells = len(self._cells)
            provenance_observations = sum(cell.provenance_count for cell in self._cells.values())
        return {
            "materialized_cells": cells,
            "exact_objects": self.exact.object_count,
            "resident_bytes": self.exact.resident_bytes,
            "provenance_observations": provenance_observations,
            "ledger_records": self.ledger.length,
            "ledger_head": self.ledger.head,
            "region_loads": self.placement.snapshot_loads(),
            "scheduler_workers": self.scheduler.workers,
            "scheduler_max_in_flight": self.scheduler.max_in_flight,
            "scheduler_submitted": self.scheduler.submitted,
            "scheduler_completed": self.scheduler.completed,
        }

    def close(self) -> None:
        self.scheduler.close()

    def __enter__(self) -> "WorldFabric":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()
