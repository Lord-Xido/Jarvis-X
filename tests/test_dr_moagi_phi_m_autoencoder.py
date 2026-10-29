"""Opt-in PyTorch Phi_M 3D autoencoder smoke/regression tests."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from jarvisx.dr_moagi_phi_m_autoencoder import (
    TrainConfig, FieldGenerator, PhiMAutoEncoder, train,
    encode_field, decode_code, load_model,
)


def test_3d_field_generation_is_seeded_and_bounded():
    left = FieldGenerator(8, 1, 123).sample(2)
    right = FieldGenerator(8, 1, 123).sample(2)
    assert left.shape == (2, 8, 8, 8, 3)
    np.testing.assert_array_equal(left, right)
    assert np.isfinite(left).all()


@pytest.mark.parametrize("grid", [8, 16])
def test_encoder_decoder_shapes_and_backprop(grid):
    model = PhiMAutoEncoder(grid=grid, latent=8)
    x = torch.randn(2, 3, grid, grid, grid)
    y, z = model(x)
    assert y.shape == x.shape
    assert z.shape == (2, 8)
    torch.nn.functional.mse_loss(y, x).backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_bounded_train_checkpoint_and_io(tmp_path):
    cfg = TrainConfig(grid=8, latent=8, train_samples=6,
                      val_samples=3, steps=5, batch=2, eval_every=2)
    output = train(cfg, tmp_path)
    assert output["updates"] == 5
    assert np.isfinite(output["validation_mse"])
    model = load_model(tmp_path / "model.pt")
    field = np.load(tmp_path / "demo_field.npy", allow_pickle=False)
    code = encode_field(model, field)
    restored = decode_code(model, code)
    assert code.shape == (8,)
    assert restored.shape == field.shape
    assert np.isfinite(restored).all()
    with pytest.raises(ValueError):
        encode_field(model, field[:5])
    with pytest.raises(ValueError):
        decode_code(model, np.full((8,), np.nan))


@pytest.mark.parametrize("invalid", [
    {"grid": 12}, {"grid": 32}, {"depth": 0}, {"steps": 0},
    {"batch": 9, "train_samples": 8}, {"lr": -1},
    {"device": "tpu"}, {"val_samples": 0},
])
def test_invalid_train_config_rejected(invalid):
    with pytest.raises(ValueError):
        TrainConfig(**invalid)
