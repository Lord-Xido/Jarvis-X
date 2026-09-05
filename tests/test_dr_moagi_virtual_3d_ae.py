from jarvisx.dr_moagi_virtual_3d_ae import Codec, Config, DrMoagiVirtual3DAE, hf, stream


def test_coordinate_materialization_is_deterministic():
    assert stream((3, 4, 5), 128, 1337) == stream((3, 4, 5), 128, 1337)
    assert stream((3, 4, 5), 128, 1337) != stream((3, 4, 6), 128, 1337)


def test_binary_codec_is_latent_cycle_consistent():
    codec = Codec(64, 8)
    source = 0xA5A55A5AF00F0FF0
    latent = codec.encode(source)
    reconstructed = codec.decode(latent)
    assert codec.encode(reconstructed) == latent
    assert 0.0 <= hf(source, reconstructed, 64) <= 1.0


def test_active_tile_is_sparse_and_bounded():
    engine = DrMoagiVirtual3DAE(Config(tile=4, bits=64, latent=8))
    engine.materialize()
    assert engine.active_streams == 4**3
    assert len(engine.state) == 4**3


def test_inward_loop_reaches_reference_fixed_point():
    engine = DrMoagiVirtual3DAE(Config(tile=4, bits=64, latent=8, passes=5, alpha=.35, beta=.65, epsilon=0.0))
    history = engine.run()
    assert len(history) >= 2
    assert history[-1].reality_gap == 0.0
    assert history[-1].changed_bits == 0


def test_geometry_contains_outer_and_latent_positions():
    engine = DrMoagiVirtual3DAE(Config(tile=4, bits=64, latent=8))
    sample = engine.geometry(1)[0]
    assert sample["virtual_coord"] == (0, 0, 0)
    assert len(sample["input_position_3d"]) == 3
    assert len(sample["latent_position_3d"]) == 3
