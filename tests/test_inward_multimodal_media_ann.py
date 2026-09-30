import numpy as np

from jarvisx.inward_multimodal_media_ann import (
    InwardMultimodalMediaANN,
    MediaANNConfig,
    Modality,
    TinyAutoencoder,
    bounded_optimize,
    consensus_latent,
    generate_pcm_audio,
    generate_rgb_frame,
    virtual_xyz,
)


def _inputs():
    x = np.linspace(-1.0, 1.0, 16)
    image = np.outer(np.sin(2 * x), np.cos(3 * x)).astype(np.float32)
    audio = np.sin(2 * np.pi * 220 * np.arange(2000) / 8000).astype(np.float32)
    return {
        Modality.TEXT: ["multimodal 3D test"],
        Modality.IMAGE: [image],
        Modality.AUDIO: [audio],
        Modality.VIDEO: [{"frames": 8, "motion": "orbit"}],
        Modality.GEOMETRY: [{"primitive": "torus"}],
        Modality.CODE: ["def f(x): return x * x"],
        Modality.DATA: [{"scene": "test"}],
    }


def _config():
    return MediaANNConfig(
        feature_dim=16,
        latent_dim=8,
        hidden_dim=20,
        train_epochs=60,
        inner_steps=8,
        learning_rate=0.04,
    )


def test_tiny_autoencoder_training_reduces_loss():
    rng = np.random.default_rng(7)
    x = rng.normal(0.0, 0.25, (8, 16))
    ae = TinyAutoencoder(16, 8, 20, seed=7)
    history = ae.train(x, epochs=80, learning_rate=0.04)
    assert history[-1] < history[0]


def test_inward_runtime_reduces_fixed_point_residual():
    result = InwardMultimodalMediaANN(_config()).run(
        _inputs(),
        "produce a coherent verified multimodal representation",
    )
    assert result.training_loss[-1] < result.training_loss[0]
    assert result.metrics[-1].fixed_point_residual < result.metrics[0].fixed_point_residual
    assert 0.0 < result.score <= 1.0


def test_media_surfaces_share_one_3d_consensus_latent():
    result = InwardMultimodalMediaANN(_config()).run(_inputs(), "shared media state")
    latent = consensus_latent(result)
    frame = generate_rgb_frame(latent, phase=0.25, size=48)
    pcm = generate_pcm_audio(latent, duration_seconds=0.05, sample_rate=8000)
    xyz = virtual_xyz(latent)

    assert frame.shape == (48, 48, 3)
    assert frame.dtype == np.uint8
    assert pcm.dtype == np.int16
    assert pcm.shape == (400,)
    assert all(0 <= value < 8192 for value in xyz)


def test_bounded_optimizer_does_not_regress_fixed_point_integrity():
    config = _config()
    baseline = InwardMultimodalMediaANN(config).run(_inputs(), "optimize safely")
    _, optimized, audit = bounded_optimize(
        _inputs(),
        "optimize safely",
        initial=config,
        generations=2,
        population=4,
        seed=11,
    )
    assert audit
    assert (
        optimized.metrics[-1].fixed_point_residual
        <= baseline.metrics[-1].fixed_point_residual + 1e-12
    )
