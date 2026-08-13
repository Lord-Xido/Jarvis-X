from __future__ import annotations

import math
import tempfile
from pathlib import Path

from jarvisx.mmvm import LambdaPolicy, MMVMCodec, MMVMConfig, MMVMKernel, SparseVoxelMemory


def test_virtual_memory_contract_is_exact_one_million_decimal_gb() -> None:
    config = MMVMConfig()
    assert config.side == 100_000
    assert config.virtual_cells == 10**15
    assert config.virtual_bytes == 10**15
    assert config.virtual_decimal_gb == 1_000_000


def test_3d_address_roundtrip_covers_last_logical_byte() -> None:
    config = MMVMConfig()
    with tempfile.TemporaryDirectory() as tmp:
        memory = SparseVoxelMemory(Path(tmp) / "mmvm.sqlite3", config)
        index = config.virtual_cells - 1
        address = memory.index_to_address(index)
        assert (address.x, address.y, address.z) == (99_999, 99_999, 99_999)
        assert memory.coordinates_to_index(address.x, address.y, address.z) == index
        memory.close()


def test_codec_is_lossless_and_latent_is_bounded() -> None:
    config = MMVMConfig()
    codec = MMVMCodec(config)
    payload = bytes(range(256)) * 4 + b"Jarvis-X MMVM"
    packet = codec.encode(payload)
    refined, xi_dot = codec.refine(packet.latent, omega=0.4, cycle=7)
    assert codec.decode(packet) == payload
    assert len(packet.latent) == config.latent_dim
    assert len(refined) == config.latent_dim
    assert all(math.isfinite(value) and -1.0 <= value <= 1.0 for value in refined)
    assert xi_dot >= 0.0


def test_lambda_rejects_checksum_mismatch() -> None:
    config = MMVMConfig()
    codec = MMVMCodec(config)
    policy = LambdaPolicy(config)
    packet = codec.encode(b"alpha")
    decision = policy.validate(packet, packet.latent, b"beta")
    assert not decision.accepted
    assert decision.reason == "checksum mismatch"


def test_kernel_runs_transaction_to_sparse_commit_and_artifact() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        kernel = MMVMKernel(Path(tmp) / "mmvm.sqlite3")
        task = kernel.submit(
            b"auto encode and decode the system",
            modality="text",
            target="image",
        )
        metrics = kernel.run_next()
        assert metrics is not None
        assert metrics.state == "committed"
        assert metrics.lambda_accepted
        assert metrics.reconstruction_error == 0.0
        assert metrics.memory_address is not None
        assert task.object_id is not None
        assert task.artifact_id is not None
        stored = kernel.memory.fetch_object(task.object_id)
        artifact = kernel.memory.fetch_artifact(task.artifact_id)
        assert stored is not None
        assert artifact is not None
        assert artifact.media_type == "image/svg+xml"
        assert artifact.payload.startswith(b"<svg")
        assert kernel.status()["memory"]["resident_objects"] == 1
        kernel.memory.close()


def test_same_payload_different_modalities_do_not_alias_object_identity() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        kernel = MMVMKernel(Path(tmp) / "mmvm.sqlite3")
        one = kernel.submit(b"same payload", modality="text")
        two = kernel.submit(b"same payload", modality="audio")
        kernel.run_until_idle()
        assert one.object_id is not None and two.object_id is not None
        assert one.object_id != two.object_id
        first = kernel.memory.fetch_object(one.object_id)
        second = kernel.memory.fetch_object(two.object_id)
        assert first is not None and second is not None
        assert first["address"]["index"] != second["address"]["index"]
        kernel.memory.close()


def test_audio_and_3d_generators_are_materialized() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        kernel = MMVMKernel(Path(tmp) / "mmvm.sqlite3")
        audio = kernel.submit(b"latent pulse", modality="text", target="audio")
        voxels = kernel.submit(b"voxel memory", modality="text", target="3d")
        kernel.run_until_idle()
        audio_artifact = kernel.memory.fetch_artifact(audio.artifact_id or "")
        voxel_artifact = kernel.memory.fetch_artifact(voxels.artifact_id or "")
        assert audio_artifact is not None and audio_artifact.payload[:4] == b"RIFF"
        assert voxel_artifact is not None and b'"voxels"' in voxel_artifact.payload
        kernel.memory.close()
