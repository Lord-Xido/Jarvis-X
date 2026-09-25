"""Smoke tests for the optional trainable MM3D backend."""
from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("numpy")
pytest.importorskip("PIL")

from jarvisx.mm3d_engine import MM3DConfig, MM3DEngine, demo_batch, laplacian3d


def tiny_config() -> MM3DConfig:
    return MM3DConfig(
        latent_channels=16,
        grid=2,
        sequence_length=16,
        image_size=16,
        audio_samples=64,
        video_frames=2,
        video_size=8,
        refinement_depth=1,
        attention_heads=4,
    )


def test_laplacian_preserves_constant_field() -> None:
    field = torch.ones(1, 2, 3, 3, 3)
    assert torch.allclose(laplacian3d(field), torch.zeros_like(field))


def test_mm3d_forward_and_backward() -> None:
    cfg = tiny_config()
    engine = MM3DEngine(cfg)
    batch = demo_batch(cfg, torch.device("cpu"))

    environment = torch.randn(1, cfg.environment_channels, 3, 3, 3)
    outputs = engine(stateful=False, environment=environment, **batch)
    assert tuple(outputs["latent"].shape) == (1, 16, 2, 2, 2)
    assert tuple(outputs["image"].shape) == (1, 3, 16, 16)
    assert tuple(outputs["audio"].shape) == (1, 1, 64)
    assert tuple(outputs["video"].shape) == (1, 3, 2, 8, 8)
    assert tuple(outputs["text_logits"].shape) == (1, 16, 256)
    assert tuple(outputs["code_logits"].shape) == (1, 16, 256)
    assert tuple(outputs["action_logits"].shape) == (1, cfg.action_dim)
    assert tuple(outputs["action_probs"].shape) == (1, cfg.action_dim)
    assert tuple(outputs["rendered_3d"].shape) == (1, 3, 16, 16)
    assert torch.allclose(
        outputs["action_probs"].sum(dim=1), torch.ones(1), atol=1.0e-6
    )

    weights = outputs["modality_weights"]
    assert weights.shape[1] == 6
    assert torch.allclose(weights.sum(dim=1), torch.ones(1), atol=1.0e-6)

    loss, metrics = engine.loss(outputs, **batch)
    assert torch.isfinite(loss)
    assert metrics["total"] > 0.0
    loss.backward()
    assert any(parameter.grad is not None for parameter in engine.parameters())
