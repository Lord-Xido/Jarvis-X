import pytest

from jarvisx.mm3d_cosmogram import CosmogramConfig, MM3DCosmogram


def make_voxels(config, seed=0):
    return tuple(
        bytes([(seed + index) % 256]) * config.voxel_bytes
        for index in range(config.cell_count)
    )


def make_engine(boundary="bounded"):
    config = CosmogramConfig(side=2, boundary=boundary)
    codebook = {index: "atom:{}".format(index).encode() for index in range(8)}
    return MM3DCosmogram(codebook=codebook, config=config), config


def test_cycle_is_deterministic():
    engine_a, config = make_engine()
    engine_b, _ = make_engine()
    voxels = make_voxels(config)
    allow = (True,) * config.cell_count

    receipt_a = engine_a.step(voxels, allow)
    receipt_b = engine_b.step(voxels, allow)

    assert receipt_a == receipt_b
    assert engine_a.verify()
    assert engine_b.verify()


def test_policy_projection_replaces_forbidden_cells_with_neutral_index():
    engine, config = make_engine()
    voxels = make_voxels(config)
    allow = (False, True, True, True, True, True, True, True)

    receipt = engine.step(voxels, allow)

    assert receipt.masked[0] == config.neutral_index
    assert receipt.masked[1:] == receipt.encoded[1:]


def test_ledger_hash_chains_across_cycles():
    engine, config = make_engine()
    allow = (True,) * config.cell_count

    first = engine.step(make_voxels(config, seed=1), allow)
    second = engine.step(make_voxels(config, seed=2), allow)

    assert second.previous_chain_hash == first.chain_hash
    assert second.chain_hash != first.chain_hash
    assert engine.verify()


def test_periodic_and_bounded_topologies_are_distinct_contracts():
    bounded, config = make_engine("bounded")
    periodic, _ = make_engine("periodic")
    voxels = make_voxels(config)
    allow = (True,) * config.cell_count

    bounded_receipt = bounded.step(voxels, allow)
    periodic_receipt = periodic.step(voxels, allow)

    assert bounded_receipt.evolved != periodic_receipt.evolved
    assert bounded_receipt.chain_hash != periodic_receipt.chain_hash


def test_rejects_wrong_voxel_width():
    engine, config = make_engine()
    voxels = [b"x" * config.voxel_bytes for _ in range(config.cell_count)]
    voxels[0] = b"short"

    with pytest.raises(ValueError, match="exactly"):
        engine.step(tuple(voxels), (True,) * config.cell_count)
