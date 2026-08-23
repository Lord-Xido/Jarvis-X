from __future__ import annotations

import pytest

from jarvisx.dr_moagi_world_fabric import (
    BoundedScheduler,
    ConsistencyClass,
    ExactContentStore,
    ProvenanceRecord,
    RegionSpec,
    SymbolicAddress3D,
    WorldFabric,
)


def test_symbolic_address_supports_arbitrary_finite_depth_without_absolute_coordinates() -> None:
    cid = ExactContentStore.cid_for(b"world-fabric-address")
    short = SymbolicAddress3D.from_cid(cid, depth=24)
    deep = SymbolicAddress3D.from_cid(cid, depth=512)

    assert short.depth == 24
    assert deep.depth == 512
    assert short.is_ancestor_of(deep)
    assert SymbolicAddress3D.from_text(short.to_text()) == short
    assert all(0 <= octant <= 7 for octant in deep.path)


def test_exact_plane_is_content_addressed_deduplicated_and_verified() -> None:
    store = ExactContentStore()
    payload = b"authoritative exact bytes"

    first = store.put(payload)
    second = store.put(payload)

    assert first == second
    assert store.object_count == 1
    assert store.resident_bytes == len(payload)
    assert store.get(first) == payload
    assert store.verify(first)
    assert not store.verify("sha256:" + "0" * 64)


def test_world_fabric_keeps_exact_truth_separate_from_latent_plane() -> None:
    provenance = ProvenanceRecord.now(
        "unit-test",
        source_uri="memory://fixture",
        media_type="text/plain",
    )
    with WorldFabric(workers=2) as fabric:
        cell = fabric.ingest(
            b"Dr Moagi worldwide sparse fabric",
            provenance=provenance,
            consistency=ConsistencyClass.GLOBAL_AUTHORITATIVE,
        )

        assert cell.cid.startswith("sha256:")
        assert len(cell.latent) == 32
        assert fabric.retrieve(cell.address) == b"Dr Moagi worldwide sparse fabric"
        assert fabric.verify()
        assert fabric.stats()["materialized_cells"] == 1


def test_ingest_deduplicates_content_without_duplicate_materialization() -> None:
    provenance = ProvenanceRecord.now("dedupe-test")
    with WorldFabric() as fabric:
        first = fabric.ingest(b"same", provenance=provenance)
        second = fabric.ingest(b"same", provenance=provenance)

        assert first == second
        assert fabric.stats()["materialized_cells"] == 1
        assert fabric.stats()["exact_objects"] == 1
        assert fabric.ledger.length == 1


def test_nearest_uses_replaceable_latent_index_but_returns_exact_cells() -> None:
    provenance = ProvenanceRecord.now("nearest-test")
    with WorldFabric() as fabric:
        alpha = fabric.ingest(b"alpha alpha alpha", provenance=provenance)
        fabric.ingest(b"zzzzzzzzzzzz", provenance=provenance)

        results = fabric.nearest(b"alpha alpha", k=1)

        assert len(results) == 1
        assert results[0][0].cid == alpha.cid
        assert -1.0 <= results[0][1] <= 1.0


def test_self_folding_placement_reduces_interaction_distance() -> None:
    regions = (
        RegionSpec("left", 0.0, 0.0, 0.0, capacity_cells=100),
        RegionSpec("right", 10.0, 0.0, 0.0, capacity_cells=100),
    )
    provenance = ProvenanceRecord.now("placement-test")
    with WorldFabric(regions=regions) as fabric:
        left = fabric.ingest(b"left-object", provenance=provenance)
        right = fabric.ingest(b"right-object", provenance=provenance)
        assert left.region_id != right.region_id

        fabric.record_interaction(left.address, right.address, weight=100.0)
        moves = fabric.fold_placement(max_moves=1)

        assert len(moves) == 1
        assert fabric.cell(left.address).region_id == fabric.cell(right.address).region_id
        assert fabric.verify()


def test_bounded_scheduler_executes_logical_tasks_on_fixed_worker_pool() -> None:
    with BoundedScheduler(workers=2, max_in_flight=4) as scheduler:
        future = scheduler.submit(lambda left, right: left + right, 20, 22)
        assert future.result(timeout=2.0) == 42

    assert scheduler.submitted == 1
    assert scheduler.completed == 1


def test_invalid_address_and_encoder_contracts_are_rejected() -> None:
    with pytest.raises(ValueError):
        SymbolicAddress3D((8,))

    def empty_encoder(_: bytes) -> tuple[float, ...]:
        return ()

    with WorldFabric(encoder=empty_encoder) as fabric:
        with pytest.raises(ValueError, match="non-empty finite vector"):
            fabric.ingest(b"payload", provenance=ProvenanceRecord.now("bad-encoder"))
