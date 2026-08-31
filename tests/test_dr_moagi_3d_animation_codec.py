from __future__ import annotations

import hashlib

import numpy as np

from jarvisx import dr_moagi_3d_animation_codec as bridge


def test_reference_source_is_exactly_1000_lines() -> None:
    status = bridge.verify_reference()
    assert status["fragments"] == 5
    assert status["lines"] == 1000
    assert status["syntax"] == "ok"
    assert status["sha256"] == bridge.EXPECTED_SHA256


def test_materialized_source_matches_reference(tmp_path) -> None:
    target = bridge.materialize(tmp_path / "dr_moagi_3d_animation_autoencoder.py")
    payload = target.read_bytes()
    assert len(target.read_text(encoding="utf-8").splitlines()) == 1000
    assert hashlib.sha256(payload).hexdigest() == bridge.EXPECTED_SHA256


def test_end_to_end_cube_encode_decode(tmp_path) -> None:
    engine = bridge.load_reference_engine()
    clip = engine.generate_cube_animation(frame_count=6, fps=30.0)
    config = engine.CodecConfig(
        latent_dim=4,
        hidden_dim=12,
        epochs=6,
        batch_size=3,
        learning_rate=0.01,
        quant_bits=8,
        keyframe_interval=2,
        seed=1337,
    )
    codec = engine.AnimationAutoCodec(config)
    fit_metrics = codec.fit(clip, verbose=False)
    packet = codec.encode(clip)
    decoded = codec.decode(packet)

    assert decoded.frame_count == clip.frame_count
    assert decoded.vertex_count == clip.vertex_count
    assert decoded.vertex_tensor().shape == clip.vertex_tensor().shape
    assert np.isfinite(fit_metrics["rmse"])
    assert packet.latent_codes.shape == (clip.frame_count, config.latent_dim)

    packet_path = tmp_path / "packet.npz"
    engine.save_packet(packet, packet_path)
    restored = engine.load_packet(packet_path)
    decoded_again = codec.decode(restored)
    assert decoded_again.vertex_tensor().shape == clip.vertex_tensor().shape
