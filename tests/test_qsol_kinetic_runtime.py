import numpy as np
import pytest

from jarvisx.qsol_kinetic_runtime import (
    DMADescriptor,
    KineticConfig,
    Opcode,
    PackedInstruction,
    QSOLKineticRuntime,
)


def sparse_volume() -> np.ndarray:
    volume = np.zeros((8, 8, 8), dtype=np.uint8)
    volume[2, 3, 4] = 255
    volume[2, 3, 5] = 240
    volume[3, 3, 4] = 250
    volume[3, 4, 4] = 245
    return volume


def test_instruction_word_round_trip_and_spatial_payload() -> None:
    payload = PackedInstruction.spatial_payload(32, 64, 16, 1023)
    instruction = PackedInstruction(
        opcode=int(Opcode.DMA_BURST_ST),
        flags=0x0D,
        regdst=1,
        payload=payload,
    )

    word = instruction.pack()
    restored = PackedInstruction.unpack(word)

    assert word == 0x100D0108040043FF
    assert restored == instruction
    assert restored.unpack_spatial_payload(restored.payload) == (32, 64, 16, 1023)


def test_dma_uses_true_uint8_byte_accounting() -> None:
    runtime = QSOLKineticRuntime(KineticConfig(latent_dim=16, max_cycles=1))
    volume = sparse_volume()
    descriptor = DMADescriptor(channel=3, origin=(2, 3, 4), shape=(2, 2, 2))

    result = runtime.run(volume, descriptor=descriptor)

    assert result.bytes_transferred == 8
    assert result.reconstruction.shape == (2, 2, 2)
    assert result.reconstruction.dtype == np.uint8


def test_source_anchored_kinetic_loop_converges_without_trivial_latent_collapse() -> None:
    runtime = QSOLKineticRuntime(
        KineticConfig(
            latent_dim=64,
            max_cycles=160,
            latent_tolerance=1.0e-3,
        )
    )

    result = runtime.run(sparse_volume())

    assert result.converged
    assert result.cycles <= 160
    assert np.linalg.norm(result.latent) > 1.0
    assert result.receipts[-1].latent_rms <= 1.0e-3
    assert result.receipts[-1].authoritative_mse < result.receipts[0].candidate_mse
    assert result.reconstruction.shape == (8, 8, 8)
    assert result.bytes_transferred == 512


def test_committed_reconstruction_error_is_monotone_nonincreasing() -> None:
    result = QSOLKineticRuntime(
        KineticConfig(latent_dim=64, max_cycles=120)
    ).run(sparse_volume())

    authoritative = [receipt.authoritative_mse for receipt in result.receipts]

    assert authoritative == sorted(authoritative, reverse=True)
    assert any(not receipt.committed for receipt in result.receipts)


def test_invalid_dma_region_fails_closed() -> None:
    runtime = QSOLKineticRuntime(KineticConfig(latent_dim=8, max_cycles=1))

    with pytest.raises(ValueError, match="exceeds source volume bounds"):
        runtime.run(
            sparse_volume(),
            descriptor=DMADescriptor(channel=0, origin=(7, 7, 7), shape=(2, 2, 2)),
        )


def test_non_uint8_voxel_storage_is_rejected() -> None:
    runtime = QSOLKineticRuntime(KineticConfig(latent_dim=8, max_cycles=1))

    with pytest.raises(TypeError, match="uint8"):
        runtime.run(np.zeros((8, 8, 8), dtype=np.float64))
