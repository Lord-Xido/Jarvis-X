from __future__ import annotations

import json
import random
from hashlib import sha256
from pathlib import Path

import pytest

import jarvisx.universal_bitcode as ub
from jarvisx.universal_bitcode_cli import main as cli_main


def _rebuild(
    container: bytes,
    manifest: dict[str, object],
    *,
    canonical: bool = True,
    payload: bytes | None = None,
) -> bytes:
    fields = list(ub.HEADER.unpack_from(container))
    old_manifest_size = fields[3]
    old_payload = container[ub.HEADER_SIZE + old_manifest_size :]
    active_payload = old_payload if payload is None else payload
    if canonical:
        manifest_bytes = json.dumps(
            manifest,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    else:
        manifest_bytes = json.dumps(manifest, ensure_ascii=False, indent=2).encode()
    fields[3] = len(manifest_bytes)
    fields[4] = manifest["raw_size"]
    fields[5] = len(active_payload)
    fields[6] = bytes.fromhex(str(manifest["raw_sha256"]))
    fields[7] = sha256(manifest_bytes).digest()
    fields[8] = sha256(active_payload).digest()
    return ub.HEADER.pack(*fields) + manifest_bytes + active_payload


def _manifest(container: bytes) -> dict[str, object]:
    manifest_size = ub.HEADER.unpack_from(container)[3]
    start = ub.HEADER_SIZE
    value = json.loads(container[start : start + manifest_size])
    assert isinstance(value, dict)
    return value


@pytest.mark.parametrize(
    ("source_name", "data", "kind", "format_name", "evidence"),
    [
        ("paper.bin", b"%PDF-1.7\n", ub.MediaKind.DOCUMENT, "pdf", "signature"),
        ("pixel.bin", b"\x89PNG\r\n\x1a\nrest", ub.MediaKind.IMAGE, "png", "signature"),
        ("pixel.bin", b"\xff\xd8\xffrest", ub.MediaKind.IMAGE, "jpeg", "signature"),
        ("pixel.bin", b"GIF89arest", ub.MediaKind.IMAGE, "gif", "signature"),
        ("pixel.bin", b"RIFF\x04\x00\x00\x00WEBP", ub.MediaKind.IMAGE, "webp", "signature"),
        ("sound.bin", b"RIFF\x04\x00\x00\x00WAVE", ub.MediaKind.AUDIO, "wav", "signature"),
        ("sound.bin", b"fLaCrest", ub.MediaKind.AUDIO, "flac", "signature"),
        ("sound.bin", b"ID3rest", ub.MediaKind.AUDIO, "mp3", "signature"),
        ("movie.bin", b"\x00\x00\x00\x18ftypisom", ub.MediaKind.VIDEO, "mp4", "signature"),
        ("movie.bin", b"\x1aE\xdf\xa3rest", ub.MediaKind.VIDEO, "webm", "signature"),
        ("scene.bin", b"glTF\x02\x00\x00\x00", ub.MediaKind.SCENE_3D, "glb", "signature"),
        ("model.bin", b"GGUF\x03\x00\x00\x00", ub.MediaKind.MODEL, "gguf", "signature"),
        ("program.bin", b"\x7fELF" + b"\x00" * 20, ub.MediaKind.CODE, "elf", "signature"),
        ("program.bin", b"MZ" + b"\x00" * 20, ub.MediaKind.CODE, "pe", "signature"),
        ("archive.bin", b"\x1f\x8brest", ub.MediaKind.ARCHIVE, "gzip", "signature"),
        ("notes.md", b"hello", ub.MediaKind.TEXT, "markdown", "extension"),
        ("mesh.obj", b"v 0 0 0\n", ub.MediaKind.SCENE_3D, "obj", "extension"),
        ("state", b'{"b":2,"a":1}', ub.MediaKind.DOCUMENT, "json", "utf8+json"),
        ("state", b"plain UTF-8", ub.MediaKind.TEXT, "text", "utf8"),
        ("state", b"\x00\xff\x10", ub.MediaKind.BINARY, "raw", "fallback"),
        ("state", b"\x80\x81", ub.MediaKind.BINARY, "raw", "fallback"),
        ("state", b"{not-json", ub.MediaKind.TEXT, "text", "utf8"),
    ],
)
def test_detect_contract(
    source_name: str,
    data: bytes,
    kind: ub.MediaKind,
    format_name: str,
    evidence: str,
) -> None:
    contract = ub.detect_contract(data, source_name=source_name)
    assert contract.media_kind is kind
    assert contract.format_name == format_name
    assert contract.metadata["detected_by"] == evidence


def test_zip_extension_refines_container_contract() -> None:
    contract = ub.detect_contract(b"PK\x03\x04payload", source_name="report.docx")
    assert contract.media_kind is ub.MediaKind.DOCUMENT
    assert contract.format_name == "docx"
    assert contract.metadata["detected_by"] == "signature+extension"


def test_signature_wins_and_records_conflicting_extension() -> None:
    contract = ub.detect_contract(b"%PDF-1.7\n", source_name="wrong.png")
    assert contract.format_name == "pdf"
    assert contract.metadata["extension_hint"] == "png"


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"Jarvis-X" * 20_000,
        bytes(range(256)) * 17,
        random.Random(1337).randbytes(32_769),
    ],
)
def test_round_trip_is_exact_and_deterministic(data: bytes) -> None:
    runtime = ub.UniversalBitcodeRuntime()
    contract = ub.RepresentationContract(
        media_kind=ub.MediaKind.BINARY,
        media_type="application/octet-stream",
        format_name="raw",
        source_name="fixture.bin",
        metadata={"z": 2, "a": [1, True, None]},
    )
    left = runtime.encode(data, contract=contract, chunk_size=1024)
    right = runtime.encode(data, contract=contract, chunk_size=1024)
    decoded = runtime.decode(left)
    assert left == right
    assert decoded.data == data
    assert decoded.contract == contract
    assert decoded.container_sha256 == sha256(left).hexdigest()
    assert runtime.verify(left).valid


def test_encoder_selects_compression_per_chunk() -> None:
    compressible = b"A" * 1024
    incompressible = random.Random(7).randbytes(1024)
    runtime = ub.UniversalBitcodeRuntime()
    container = runtime.encode(compressible + incompressible, chunk_size=1024)
    manifest = runtime.inspect(container)
    assert [chunk.codec for chunk in manifest.chunks] == ["zlib", "identity"]
    assert runtime.decode(container).data == compressible + incompressible


def test_empty_artifact_has_defined_merkle_root() -> None:
    runtime = ub.UniversalBitcodeRuntime()
    container = runtime.encode(b"")
    report = runtime.verify(container)
    assert report.chunk_count == 0
    assert report.merkle_root_sha256 == sha256(b"").hexdigest()
    assert report.payload_ratio == 1.0


def test_close_loop_reaches_byte_exact_fixed_point() -> None:
    runtime = ub.UniversalBitcodeRuntime()
    receipt = runtime.close_loop(
        b"bits -> structure -> bits", source_name="axiom.txt", chunk_size=7
    )
    assert receipt.reality_gap_bytes == 0
    assert receipt.fixed_point
    assert receipt.input_sha256 == receipt.reconstructed_sha256
    assert receipt.container_sha256 == receipt.reencoded_sha256
    assert receipt.as_dict()["media_kind"] == "text"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"media_type": "invalid"},
        {"format_name": "bad format"},
        {"source_name": "bad\nname"},
        {"schema": ""},
        {"schema": "bad\x00schema"},
        {"media_kind": "telepathy"},
        {"metadata": {"nan": float("nan")}},
    ],
)
def test_contract_rejects_ambiguous_or_noncanonical_values(kwargs: dict[str, object]) -> None:
    values: dict[str, object] = {
        "media_kind": ub.MediaKind.BINARY,
        "media_type": "application/octet-stream",
        "format_name": "raw",
    }
    values.update(kwargs)
    with pytest.raises(ub.BitcodeFormatError):
        ub.RepresentationContract(**values)  # type: ignore[arg-type]


def test_contract_parser_is_version_strict() -> None:
    contract = ub.detect_contract(b"text")
    value = contract.as_dict()
    value["unexpected"] = True
    with pytest.raises(ub.BitcodeFormatError, match="fields"):
        ub.RepresentationContract.from_dict(value)
    value = contract.as_dict()
    value["media_kind"] = "telepathy"
    with pytest.raises(ub.BitcodeFormatError, match="unsupported"):
        ub.RepresentationContract.from_dict(value)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_input_bytes": 0},
        {"max_manifest_bytes": True},
        {"max_chunks": -1},
    ],
)
def test_budget_requires_positive_integer_fields(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        ub.BitcodeBudget(**kwargs)  # type: ignore[arg-type]


def test_versioned_data_parsers_reject_wrong_shapes_and_types() -> None:
    with pytest.raises(ub.BitcodeFormatError, match="contract must"):
        ub.RepresentationContract.from_dict(None)
    contract = ub.detect_contract(b"x").as_dict()
    contract["metadata"] = []
    with pytest.raises(ub.BitcodeFormatError, match="metadata"):
        ub.RepresentationContract.from_dict(contract)
    contract = ub.detect_contract(b"x").as_dict()
    contract["schema"] = 3
    with pytest.raises(ub.BitcodeFormatError, match="schema"):
        ub.RepresentationContract.from_dict(contract)
    with pytest.raises(ub.BitcodeFormatError, match="chunk descriptor"):
        ub.ChunkDescriptor.from_dict(None)

    runtime = ub.UniversalBitcodeRuntime()
    manifest = runtime.inspect(runtime.encode(b"parser")).as_dict()
    chunks = manifest["chunks"]
    assert isinstance(chunks, list) and isinstance(chunks[0], dict)
    chunk = dict(chunks[0])
    chunk["codec"] = "brotli"
    with pytest.raises(ub.BitcodeFormatError, match="unsupported chunk codec"):
        ub.ChunkDescriptor.from_dict(chunk)
    chunk = dict(chunks[0])
    chunk["index"] = True
    with pytest.raises(ub.BitcodeFormatError, match="integer"):
        ub.ChunkDescriptor.from_dict(chunk)
    chunk = dict(chunks[0])
    chunk["raw_sha256"] = "BAD"
    with pytest.raises(ub.BitcodeFormatError, match="SHA-256"):
        ub.ChunkDescriptor.from_dict(chunk)
    chunk = dict(chunks[0])
    chunk["extra"] = 1
    with pytest.raises(ub.BitcodeFormatError, match="chunk fields"):
        ub.ChunkDescriptor.from_dict(chunk)

    with pytest.raises(ub.BitcodeFormatError, match="manifest must"):
        ub.BitcodeManifest.from_dict(None)
    wrong = dict(manifest)
    wrong["extra"] = 1
    with pytest.raises(ub.BitcodeFormatError, match="manifest fields"):
        ub.BitcodeManifest.from_dict(wrong)
    wrong = dict(manifest)
    wrong["version"] = 2
    with pytest.raises(ub.BitcodeFormatError, match="schema or version"):
        ub.BitcodeManifest.from_dict(wrong)
    wrong = dict(manifest)
    wrong["chunks"] = {}
    with pytest.raises(ub.BitcodeFormatError, match="chunks must"):
        ub.BitcodeManifest.from_dict(wrong)


@pytest.mark.parametrize("mutation", ["magic", "version", "flags"])
def test_header_contract_is_strict(mutation: str) -> None:
    runtime = ub.UniversalBitcodeRuntime()
    container = runtime.encode(b"header")
    fields = list(ub.HEADER.unpack_from(container))
    if mutation == "magic":
        fields[0] = b"BROKEN!!"
    elif mutation == "version":
        fields[1] = 2
    else:
        fields[2] = 1
    broken = ub.HEADER.pack(*fields) + container[ub.HEADER_SIZE :]
    with pytest.raises(ub.BitcodeFormatError):
        runtime.inspect(broken)


def test_container_rejects_truncation_and_trailing_bytes() -> None:
    runtime = ub.UniversalBitcodeRuntime()
    container = runtime.encode(b"bounded envelope")
    with pytest.raises(ub.BitcodeFormatError, match="shorter"):
        runtime.inspect(container[:10])
    with pytest.raises(ub.BitcodeFormatError, match="length"):
        runtime.inspect(container[:-1])
    with pytest.raises(ub.BitcodeFormatError, match="length"):
        runtime.inspect(container + b"x")


def test_payload_corruption_is_rejected_before_decompression() -> None:
    runtime = ub.UniversalBitcodeRuntime()
    container = bytearray(runtime.encode(b"A" * 4096, chunk_size=4096))
    container[-1] ^= 1
    with pytest.raises(ub.IntegrityError, match="payload digest"):
        runtime.decode(container)


def test_manifest_corruption_is_rejected() -> None:
    runtime = ub.UniversalBitcodeRuntime()
    container = bytearray(runtime.encode(b"manifest"))
    container[ub.HEADER_SIZE + 10] ^= 1
    with pytest.raises(ub.IntegrityError, match="manifest digest"):
        runtime.inspect(container)


def test_invalid_utf8_manifest_is_rejected_after_digest_validation() -> None:
    runtime = ub.UniversalBitcodeRuntime()
    container = runtime.encode(b"manifest")
    fields = list(ub.HEADER.unpack_from(container))
    old_size = fields[3]
    payload = container[ub.HEADER_SIZE + old_size :]
    manifest_bytes = b"\xff\xfe"
    fields[3] = len(manifest_bytes)
    fields[7] = sha256(manifest_bytes).digest()
    broken = ub.HEADER.pack(*fields) + manifest_bytes + payload
    with pytest.raises(ub.BitcodeFormatError, match="UTF-8 JSON"):
        runtime.inspect(broken)


def test_over_nested_manifest_is_reported_as_a_format_error() -> None:
    runtime = ub.UniversalBitcodeRuntime()
    container = runtime.encode(b"")
    fields = list(ub.HEADER.unpack_from(container))
    old_size = fields[3]
    payload = container[ub.HEADER_SIZE + old_size :]
    depth = 100_000
    manifest_bytes = b"[" * depth + b"0" + b"]" * depth
    fields[3] = len(manifest_bytes)
    fields[7] = sha256(manifest_bytes).digest()
    broken = ub.HEADER.pack(*fields) + manifest_bytes + payload
    with pytest.raises(ub.BitcodeFormatError, match="UTF-8 JSON"):
        runtime.inspect(broken)


def test_manifest_and_header_identities_must_agree() -> None:
    runtime = ub.UniversalBitcodeRuntime()
    container = runtime.encode(b"identities")
    fields = list(ub.HEADER.unpack_from(container))
    fields[6] = b"\x00" * 32
    wrong_raw_header = ub.HEADER.pack(*fields) + container[ub.HEADER_SIZE :]
    with pytest.raises(ub.IntegrityError, match="raw identity"):
        runtime.inspect(wrong_raw_header)

    manifest = _manifest(container)
    manifest["payload_sha256"] = "0" * 64
    wrong_payload_manifest = _rebuild(container, manifest)
    with pytest.raises(ub.IntegrityError, match="payload identity"):
        runtime.inspect(wrong_payload_manifest)


def test_noncanonical_manifest_is_rejected_even_with_valid_digest() -> None:
    runtime = ub.UniversalBitcodeRuntime()
    container = runtime.encode(b"canonical")
    broken = _rebuild(container, _manifest(container), canonical=False)
    with pytest.raises(ub.BitcodeFormatError, match="canonical"):
        runtime.inspect(broken)


def test_chunk_layout_is_strict() -> None:
    runtime = ub.UniversalBitcodeRuntime()
    container = runtime.encode(b"layout" * 100, chunk_size=128)
    manifest = _manifest(container)
    chunks = manifest["chunks"]
    assert isinstance(chunks, list) and isinstance(chunks[1], dict)
    chunks[1]["raw_offset"] = 999
    broken = _rebuild(container, manifest)
    with pytest.raises(ub.BitcodeFormatError, match="offsets"):
        runtime.inspect(broken)


def test_raw_chunk_digest_mismatch_is_rejected() -> None:
    runtime = ub.UniversalBitcodeRuntime()
    container = runtime.encode(b"digest" * 100)
    manifest = _manifest(container)
    chunks = manifest["chunks"]
    assert isinstance(chunks, list) and isinstance(chunks[0], dict)
    chunks[0]["raw_sha256"] = "0" * 64
    broken = _rebuild(container, manifest)
    with pytest.raises(ub.IntegrityError, match="raw chunk"):
        runtime.decode(broken)


def test_encoded_chunk_digest_mismatch_is_rejected() -> None:
    runtime = ub.UniversalBitcodeRuntime()
    container = runtime.encode(b"encoded digest")
    manifest = _manifest(container)
    chunks = manifest["chunks"]
    assert isinstance(chunks, list) and isinstance(chunks[0], dict)
    chunks[0]["encoded_sha256"] = "0" * 64
    broken = _rebuild(container, manifest)
    with pytest.raises(ub.IntegrityError, match="encoded chunk"):
        runtime.decode(broken)


def test_inspect_does_not_execute_codec_but_decode_rejects_bad_zlib() -> None:
    runtime = ub.UniversalBitcodeRuntime()
    container = runtime.encode(b"Z" * 4096, chunk_size=4096)
    manifest = _manifest(container)
    chunks = manifest["chunks"]
    assert isinstance(chunks, list) and isinstance(chunks[0], dict)
    payload = b"not-a-deflate-stream"
    chunks[0]["encoded_size"] = len(payload)
    chunks[0]["encoded_sha256"] = sha256(payload).hexdigest()
    manifest["encoded_size"] = len(payload)
    manifest["payload_sha256"] = sha256(payload).hexdigest()
    broken = _rebuild(container, manifest, payload=payload)
    assert runtime.inspect(broken).encoded_size == len(payload)
    with pytest.raises(ub.IntegrityError, match="could not be decoded"):
        runtime.decode(broken)


def test_decompressor_enforces_declared_raw_size() -> None:
    runtime = ub.UniversalBitcodeRuntime()
    container = runtime.encode(b"A" * 1024, chunk_size=1024)
    manifest = _manifest(container)
    chunks = manifest["chunks"]
    assert isinstance(chunks, list) and isinstance(chunks[0], dict)
    chunks[0]["raw_size"] = 1
    manifest["raw_size"] = 1
    manifest["raw_sha256"] = sha256(b"A").hexdigest()
    broken = _rebuild(container, manifest)
    with pytest.raises(ub.IntegrityError, match="bounded size"):
        runtime.decode(broken)


def test_merkle_root_mismatch_is_rejected_after_reconstruction() -> None:
    runtime = ub.UniversalBitcodeRuntime()
    container = runtime.encode(b"merkle")
    manifest = _manifest(container)
    manifest["merkle_root_sha256"] = "f" * 64
    broken = _rebuild(container, manifest)
    with pytest.raises(ub.IntegrityError, match="Merkle"):
        runtime.decode(broken)


def test_additional_chunk_layout_failures_are_rejected() -> None:
    runtime = ub.UniversalBitcodeRuntime()
    container = runtime.encode(b"L" * 128, chunk_size=64)

    missing = _manifest(container)
    missing_chunks = missing["chunks"]
    assert isinstance(missing_chunks, list)
    missing_chunks.pop()
    with pytest.raises(ub.BitcodeFormatError, match="chunk count"):
        runtime.inspect(_rebuild(container, missing))

    wrong_index = _manifest(container)
    index_chunks = wrong_index["chunks"]
    assert isinstance(index_chunks, list) and isinstance(index_chunks[0], dict)
    index_chunks[0]["index"] = 4
    with pytest.raises(ub.BitcodeFormatError, match="indexes"):
        runtime.inspect(_rebuild(container, wrong_index))

    oversized_raw = _manifest(container)
    raw_chunks = oversized_raw["chunks"]
    assert isinstance(raw_chunks, list) and isinstance(raw_chunks[0], dict)
    raw_chunks[0]["raw_size"] = 65
    with pytest.raises(ub.BitcodeFormatError, match="raw size"):
        runtime.inspect(_rebuild(container, oversized_raw))

    not_closed = _manifest(container)
    closing_chunks = not_closed["chunks"]
    assert isinstance(closing_chunks, list) and isinstance(closing_chunks[-1], dict)
    closing_chunks[-1]["raw_size"] = 63
    with pytest.raises(ub.BitcodeFormatError, match="does not close"):
        runtime.inspect(_rebuild(container, not_closed))


def test_parser_applies_manifest_and_chunk_count_budgets() -> None:
    container = ub.UniversalBitcodeRuntime().encode(b"B" * 128, chunk_size=64)
    with pytest.raises(ub.ResourceLimitError, match="manifest size"):
        ub.UniversalBitcodeRuntime(budget=ub.BitcodeBudget(max_manifest_bytes=32)).inspect(
            container
        )
    with pytest.raises(ub.ResourceLimitError, match="chunk count"):
        ub.UniversalBitcodeRuntime(budget=ub.BitcodeBudget(max_chunks=1)).inspect(container)


@pytest.mark.parametrize("chunk_size", [0, -1, True])
def test_chunk_size_requires_a_positive_integer(chunk_size: object) -> None:
    with pytest.raises(ub.BitcodeFormatError, match="positive integer"):
        ub.UniversalBitcodeRuntime().encode(b"x", chunk_size=chunk_size)  # type: ignore[arg-type]


def test_runtime_budgets_reject_before_expansion() -> None:
    small = ub.UniversalBitcodeRuntime(
        budget=ub.BitcodeBudget(
            max_input_bytes=8,
            max_manifest_bytes=1024,
            max_metadata_bytes=128,
            max_source_name_bytes=4,
            max_chunks=1,
            max_chunk_size=4,
        )
    )
    with pytest.raises(ub.ResourceLimitError, match="input size"):
        small.encode(b"123456789", chunk_size=4)
    with pytest.raises(ub.ResourceLimitError, match="chunk count"):
        small.encode(b"12345", chunk_size=4)
    with pytest.raises(ub.ResourceLimitError, match="chunk_size"):
        small.encode(b"1", chunk_size=5)
    with pytest.raises(ub.ResourceLimitError, match="source_name"):
        small.encode(b"1", source_name="12345", chunk_size=4)
    contract = ub.RepresentationContract(
        ub.MediaKind.BINARY,
        "application/octet-stream",
        "raw",
        metadata={"long": "x" * 256},
    )
    with pytest.raises(ub.ResourceLimitError, match="metadata"):
        small.encode(b"1", contract=contract, chunk_size=4)


def test_parse_obeys_tighter_budget_than_encoder() -> None:
    container = ub.UniversalBitcodeRuntime().encode(b"123456789")
    runtime = ub.UniversalBitcodeRuntime(budget=ub.BitcodeBudget(max_input_bytes=4))
    with pytest.raises(ub.ResourceLimitError, match="byte budget"):
        runtime.inspect(container)


def test_manifest_size_budget_is_enforced() -> None:
    runtime = ub.UniversalBitcodeRuntime(budget=ub.BitcodeBudget(max_manifest_bytes=32))
    with pytest.raises(ub.ResourceLimitError, match="manifest size"):
        runtime.encode(b"")


def test_cli_cycle_inspect_verify_and_decode(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "scene.gltf"
    container = tmp_path / "scene.jxbi"
    restored = tmp_path / "scene.restored.gltf"
    source.write_bytes(b'{"asset":{"version":"2.0"}}')

    assert cli_main(["cycle", str(source), str(container), "--chunk-size", "8"]) == 0
    cycle_output = json.loads(capsys.readouterr().out)
    assert cycle_output["fixed_point"] is True
    assert cycle_output["reality_gap_bytes"] == 0

    assert cli_main(["inspect", str(container)]) == 0
    inspected = json.loads(capsys.readouterr().out)
    assert inspected["contract"]["format_name"] == "gltf"

    assert cli_main(["verify", str(container)]) == 0
    verified = json.loads(capsys.readouterr().out)
    assert verified["valid"] is True

    assert cli_main(["decode", str(container), str(restored)]) == 0
    capsys.readouterr()
    assert restored.read_bytes() == source.read_bytes()


def test_cli_decode_decompresses_the_container_once(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.txt"
    container = tmp_path / "source.jxbi"
    restored = tmp_path / "source.restored.txt"
    source.write_text("bounded decode", encoding="utf-8")
    assert cli_main(["encode", str(source), str(container)]) == 0
    capsys.readouterr()

    decode_calls = 0
    original_decode = ub.UniversalBitcodeRuntime.decode

    def counted_decode(
        runtime: ub.UniversalBitcodeRuntime, data: bytes | bytearray | memoryview
    ) -> ub.DecodedArtifact:
        nonlocal decode_calls
        decode_calls += 1
        return original_decode(runtime, data)

    monkeypatch.setattr(ub.UniversalBitcodeRuntime, "decode", counted_decode)
    assert cli_main(["decode", str(container), str(restored)]) == 0
    capsys.readouterr()
    assert decode_calls == 1
    assert restored.read_bytes() == source.read_bytes()


def test_cli_refuses_implicit_overwrite_and_same_path(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "input.txt"
    output = tmp_path / "output.jxbi"
    source.write_text("bounded", encoding="utf-8")
    output.write_bytes(b"existing")
    assert cli_main(["encode", str(source), str(output)]) == 2
    assert "output exists" in capsys.readouterr().err
    assert output.read_bytes() == b"existing"
    assert cli_main(["encode", str(source), str(source), "--force"]) == 2
    assert "must be different" in capsys.readouterr().err


def test_cli_accepts_explicit_contract_and_atomic_force(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "artifact"
    output = tmp_path / "nested" / "artifact.jxbi"
    source.write_bytes(b"custom")
    args = [
        "encode",
        str(source),
        str(output),
        "--media-kind",
        "model",
        "--media-type",
        "application/x-example-model",
        "--format-name",
        "example-model",
        "--schema",
        "example/v1",
        "--metadata",
        '{"axis":3}',
    ]
    assert cli_main(args) == 0
    capsys.readouterr()
    first = output.read_bytes()
    assert cli_main(args + ["--force"]) == 0
    capsys.readouterr()
    assert output.read_bytes() == first
    contract = ub.UniversalBitcodeRuntime().inspect(first).contract
    assert contract.media_kind is ub.MediaKind.MODEL
    assert contract.metadata["axis"] == 3


def test_cli_reports_bad_metadata(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = tmp_path / "input"
    output = tmp_path / "output"
    source.write_bytes(b"x")
    assert cli_main(["encode", str(source), str(output), "--metadata", "[]"]) == 2
    assert "JSON object" in capsys.readouterr().err


def test_rom_forge_provenance_manifest_is_complete() -> None:
    path = Path(__file__).parents[1] / "reference" / "rom_forge_legacy" / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["schema"] == "jarvisx.rom-forge-provenance"
    assert manifest["live_code_copied"] is False
    entries = manifest["entries"]
    assert [entry["index"] for entry in entries] == list(range(1, 17))
    assert len({entry["source_name"] for entry in entries}) == 16
    assert all(ub._DIGEST_RE.fullmatch(entry["sha256"]) for entry in entries)
    assert all(entry["bytes"] >= 0 for entry in entries)
