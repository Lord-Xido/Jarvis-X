from __future__ import annotations

from pathlib import Path

import pytest

from jarvisx.dr_moagi_firmware import (
    FIRMWARE_LAYOUT,
    IMAGE_SIZE,
    FirmwareBuilder,
    FirmwareImage,
    SparseMetricCodec3D,
    build_reference_riscv_elf,
    generate_aes256_key,
    generate_ed25519_keypair,
    identity_metric_for_state,
    inspect_riscv_elf,
    relax_metric_field,
    validate_layout,
)
from jarvisx.dr_moagi_os import demo_field


def test_firmware_layout_is_exact_contiguous_one_gib() -> None:
    validate_layout()
    assert FIRMWARE_LAYOUT[0].offset == 0
    assert FIRMWARE_LAYOUT[-1].end == IMAGE_SIZE == 1_073_741_824
    assert [region.capacity >> 20 for region in FIRMWARE_LAYOUT] == [1, 383, 320, 192, 128]


def test_reference_kernel_is_valid_riscv_elf64() -> None:
    payload = build_reference_riscv_elf()
    info = inspect_riscv_elf(payload)
    assert payload[:4] == b"\x7fELF"
    assert info["machine"] == 243
    assert info["entry"] == 0x80000000
    assert info["program_headers"] == 1


def test_sparse_metric_roundtrip_and_spd_relaxation() -> None:
    field = {
        (1, 1, 1): (1.2, 1.1, 1.0, 0.05, 0.02, 0.01),
        (2, 1, 1): (1.1, 1.0, 1.3, 0.02, 0.01, 0.03),
    }
    codec = SparseMetricCodec3D()
    payload = codec.encode(field, side=8)
    side, decoded = codec.decode(payload)
    assert side == 8
    assert set(decoded) == set(field)
    relaxed = relax_metric_field(decoded, side=8, alpha=0.1)
    assert set(relaxed) == set(field)
    codec.encode(relaxed, side=8)  # validation succeeds

    with pytest.raises(ValueError, match="positive definite"):
        codec.encode({(1, 1, 1): (-1, 1, 1, 0, 0, 0)}, side=8)


def test_signed_encrypted_image_verifies_and_is_logically_one_gib(tmp_path: Path) -> None:
    private, public = generate_ed25519_keypair()
    aes = generate_aes256_key()
    state = demo_field(8)
    image_path = tmp_path / "dr-moagi.img"
    build = FirmwareBuilder().build(
        image_path,
        state=state,
        side=8,
        signing_private_key=private,
        encryption_key=aes,
    )
    assert build["image_size"] == IMAGE_SIZE
    assert image_path.stat().st_size == IMAGE_SIZE

    image = FirmwareImage(image_path)
    report = image.verify(public_key=public, encryption_key=aes)
    assert report.signed is True
    assert report.encrypted is True
    assert report.signature_valid is True
    assert report.qsol_cells == len(state)
    assert report.metric_cells == len(state)
    assert report.kernel_machine == 243


def test_wrong_crypto_keys_fail_closed(tmp_path: Path) -> None:
    private, public = generate_ed25519_keypair()
    _, wrong_public = generate_ed25519_keypair()
    aes = generate_aes256_key()
    wrong_aes = generate_aes256_key()
    image_path = tmp_path / "secure.img"
    FirmwareBuilder().build(
        image_path,
        state=demo_field(8),
        side=8,
        signing_private_key=private,
        encryption_key=aes,
    )
    image = FirmwareImage(image_path)
    with pytest.raises(ValueError, match="trust anchor"):
        image.verify(public_key=wrong_public, encryption_key=aes)
    with pytest.raises(ValueError, match="authentication failed"):
        image.verify(public_key=public, encryption_key=wrong_aes)


def test_section_tampering_is_detected_before_boot(tmp_path: Path) -> None:
    state = demo_field(8)
    image_path = tmp_path / "tampered.img"
    FirmwareBuilder().build(image_path, state=state, side=8)
    qsol = next(region for region in FIRMWARE_LAYOUT if region.name == "qsol")
    with image_path.open("r+b") as handle:
        handle.seek(qsol.offset)
        first = handle.read(1)
        handle.seek(qsol.offset)
        handle.write(bytes([first[0] ^ 0x01]))
    with pytest.raises(ValueError, match="section checksum mismatch"):
        FirmwareImage(image_path).verify()


def test_verified_boot_runs_existing_autonomic_system_and_metric_transaction(tmp_path: Path) -> None:
    state = demo_field(8)
    image_path = tmp_path / "boot.img"
    FirmwareBuilder().build(
        image_path,
        state=state,
        side=8,
        metric=identity_metric_for_state(state),
    )
    session = FirmwareImage(image_path).boot(max_active_cells=512)
    report = session.run(1)
    assert report.verification.qsol_cells == len(state)
    assert len(report.autonomic.state_reports) == 1
    assert len(session.trace_head) == 64
    assert session.export_state()
