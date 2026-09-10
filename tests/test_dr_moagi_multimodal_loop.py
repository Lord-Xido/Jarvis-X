import math

from jarvisx.dr_moagi_multimodal_loop import (
    Autoencoder3D,
    DrMoagiMultimodal3DLoop,
    Modality,
    Payload3DAdapter,
    SeptillionAddressSpace,
    VirtualSeptillionClock,
)


def test_septillion_address_deinterleaves_24_digits_into_3d():
    assert SeptillionAddressSpace.address(0) == (0, 0, 0)
    assert SeptillionAddressSpace.address(1) == (1, 0, 0)
    assert SeptillionAddressSpace.address(10) == (0, 1, 0)
    assert SeptillionAddressSpace.address(100) == (0, 0, 1)
    x, y, z = SeptillionAddressSpace.address(10**24 - 1)
    assert (x, y, z) == (99_999_999, 99_999_999, 99_999_999)


def test_virtual_clock_uses_integer_nanosecond_accounting():
    ticks = iter((1_000_000_000, 1_000_000_123))
    clock = VirtualSeptillionClock(now_ns=lambda: next(ticks))
    assert clock.virtual_ops() == 123 * 10**15


def test_transactional_autoencoder_never_commits_a_worse_step():
    volume = Payload3DAdapter.ingest(b"3D transactional test", Modality.TEXT, 8)
    model = Autoencoder3D(8, seed=7)
    before = model.loss(volume)
    model.train_transactional(volume)
    after = model.loss(volume)
    assert math.isfinite(after)
    assert after <= before + 1.0e-12


def test_full_loop_generates_every_modality(tmp_path):
    engine = DrMoagiMultimodal3DLoop(edge=8, temporal_depth=3, seed=11)
    engine.set_payload(Modality.TEXT, "Jarvis X multimodal loop")
    engine.set_payload(Modality.AUDIO, bytes(range(64)))
    metrics = engine.run(cycles=2, deterministic_virtual_stride=10**20)

    assert metrics.cycle == 2
    assert metrics.virtual_ops == 10**20
    assert metrics.virtual_index == 10**20
    assert math.isfinite(metrics.aggregate_reconstruction_mse)
    assert math.isfinite(metrics.aggregate_cycle_mse)
    assert set(metrics.modalities) == {m.value for m in Modality}
    assert abs(sum(m.fusion_weight for m in metrics.modalities.values()) - 1.0) < 1.0e-6

    for modality in Modality:
        payload = engine.generated_payload(modality)
        assert payload

    engine.export(tmp_path)
    assert (tmp_path / "metrics.json").is_file()
    assert (tmp_path / "fused-latent.obj").is_file()
    assert (tmp_path / "generated-text.raw").is_file()
