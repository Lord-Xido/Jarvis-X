from __future__ import annotations

import pytest

from jarvisx.cloud import (
    HierarchicalAddress,
    HyperCloudRuntime,
    MediaEnvelope,
    MediaKind,
    SpatialShardRouter,
    SymbolicParameterExtent,
)


def test_symbolic_extent_does_not_expand_parameter_count() -> None:
    extent = SymbolicParameterExtent()

    assert extent.expression == "1000000^(1000000^1000000)"
    assert extent.address_radix == 1_000_000
    assert extent.metadata()["allocation_semantics"] == "symbolic-sparse"


def test_hierarchical_address_rejects_digits_outside_radix() -> None:
    address = HierarchicalAddress((1, 2, 1_000_000))

    with pytest.raises(ValueError, match="outside radix"):
        address.validate(radix=1_000_000)


def test_3d_routing_is_deterministic_and_bounded() -> None:
    router = SpatialShardRouter(shape=(4, 5, 6))
    address = HierarchicalAddress((10, 20, 30, 40))

    first = router.route(namespace="model-a", modality="text", address=address)
    second = router.route(namespace="model-a", modality="text", address=address)

    assert first == second
    assert 0 <= first.x < 4
    assert 0 <= first.y < 5
    assert 0 <= first.z < 6


def test_parameter_store_materializes_only_touched_addresses() -> None:
    runtime = HyperCloudRuntime()
    address = HierarchicalAddress((7, 11, 13))

    assert runtime.store.materialized_parameters == 0
    assert runtime.store.get("tenant/model", address) is None

    runtime.store.set("tenant/model", address, 0.125)

    assert runtime.store.get("tenant/model", address) == pytest.approx(0.125)
    assert runtime.store.materialized_parameters == 1
    assert runtime.describe()["claims"]["dense_parameter_allocation"] is False


def test_multimodal_ingest_is_content_addressed() -> None:
    runtime = HyperCloudRuntime()
    media = MediaEnvelope(
        kind=MediaKind.IMAGE,
        payload=b"synthetic-image-bytes",
        content_type="image/example",
    )

    record = runtime.ingest(namespace="demo", media=media)

    assert record["kind"] == "image"
    assert record["sha256"] == media.digest
    assert runtime.media_record(media.digest) == record
    assert runtime.describe()["ingested_media_objects"] == 1


def test_modality_changes_spatial_route() -> None:
    runtime = HyperCloudRuntime(router=SpatialShardRouter(shape=(32, 32, 32)))
    address = HierarchicalAddress((101, 202, 303))

    text_route = runtime.route(namespace="demo", modality="text", address=address)
    video_route = runtime.route(namespace="demo", modality="video", address=address)

    assert text_route != video_route
