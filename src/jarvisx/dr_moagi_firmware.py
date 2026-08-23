"""Verified 1 GiB firmware-container runtime for the Dr Moagi architecture.

This module turns the earlier conceptual byte map into a concrete, deterministic
container format with an exact 1 GiB *logical* image size.  The image is sparse
on filesystems that support sparse files: only the used payload bytes are
written; the large reserved regions remain holes.

The container deliberately separates immutable executable material from mutable
state.  It embeds a valid minimal RV64 ELF monitor stub (or a caller-provided
RISC-V ELF), exact DMOS2 sparse state, a sparse symmetric-positive-definite 3D
metric packet, and a hash-chain trace anchor.  The manifest may be authenticated
with Ed25519 and the state/metric payloads may be encrypted with AES-256-GCM.

This is a reference firmware/runtime layer.  It does not claim that the Python
host is bare metal or that a generic GPU executes RISC-V instructions directly.
Board-specific startup code and accelerator binaries can replace the reference
ELF payload while retaining the verified container contract.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence, cast

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from .dr_moagi_field_runtime import Coordinate, SparseField
from .dr_moagi_frontier import morton3_decode, morton3_encode
from .dr_moagi_meta_optimizer import SelfOptimizing3DSystem
from .dr_moagi_os import DrMoagiOSConfig, DrMoagiOSKernel
from .dr_moagi_os_store import SparseStateCodec3D
from .dr_moagi_system_evolution import AutonomicRunReport, SelfEvolving3DArchitecture

MIB = 1 << 20
IMAGE_SIZE = 1 << 30  # exactly 1 GiB / 1,073,741,824 bytes
HEADER_SIZE = 512
MANIFEST_OFFSET = HEADER_SIZE

_CONTAINER_MAGIC = b"DMLAMBDA"
_CONTAINER_VERSION = 1
_HEADER_STRUCT = struct.Struct("<8sIIQII32s64s")
_FLAG_SIGNED = 1 << 0
_FLAG_ENCRYPTED = 1 << 1

_METRIC_MAGIC = b"DMMET1"
_TRACE_MAGIC = b"DMTRC1"
_RISCV_MACHINE = 243


@dataclass(frozen=True)
class FirmwareRegion:
    name: str
    offset: int
    capacity: int

    @property
    def end(self) -> int:
        return self.offset + self.capacity

    def as_dict(self) -> dict[str, object]:
        return {"name": self.name, "offset": self.offset, "capacity": self.capacity}


FIRMWARE_LAYOUT: tuple[FirmwareRegion, ...] = (
    FirmwareRegion("genesis", 0x00000000, 1 * MIB),
    FirmwareRegion("qsol", 0x00100000, 383 * MIB),
    FirmwareRegion("metric", 0x18000000, 320 * MIB),
    FirmwareRegion("kernel", 0x2C000000, 192 * MIB),
    FirmwareRegion("trace", 0x38000000, 128 * MIB),
)
_REGION_BY_NAME = {region.name: region for region in FIRMWARE_LAYOUT}

MetricComponents = tuple[float, float, float, float, float, float]
MetricField3D = dict[Coordinate, MetricComponents]


@dataclass(frozen=True)
class FirmwareVerificationReport:
    image_size: int
    signed: bool
    encrypted: bool
    signature_valid: bool
    qsol_cells: int
    metric_cells: int
    kernel_machine: int
    kernel_entry: int
    trace_head: str
    manifest_sha256: str
    sections: dict[str, dict[str, object]]

    def as_dict(self) -> dict[str, object]:
        return cast(dict[str, object], asdict(self))


@dataclass(frozen=True)
class FirmwareRunReport:
    verification: FirmwareVerificationReport
    autonomic: AutonomicRunReport
    metric_cells: int
    metric_relaxation_committed: bool
    trace_head: str

    def as_dict(self) -> dict[str, object]:
        return {
            "verification": self.verification.as_dict(),
            "autonomic": self.autonomic.as_dict(),
            "metric_cells": self.metric_cells,
            "metric_relaxation_committed": self.metric_relaxation_committed,
            "trace_head": self.trace_head,
        }


def validate_layout() -> None:
    """Assert that the firmware map is contiguous and closes at exactly 1 GiB."""

    cursor = 0
    for region in FIRMWARE_LAYOUT:
        if region.offset != cursor:
            raise RuntimeError(f"firmware region {region.name} is not contiguous")
        if region.capacity <= 0:
            raise RuntimeError(f"firmware region {region.name} has invalid capacity")
        cursor = region.end
    if cursor != IMAGE_SIZE:
        raise RuntimeError("firmware layout does not close at exactly 1 GiB")
    if MANIFEST_OFFSET >= _REGION_BY_NAME["genesis"].end:
        raise RuntimeError("manifest offset lies outside genesis region")


def generate_ed25519_keypair() -> tuple[bytes, bytes]:
    """Return raw 32-byte Ed25519 private and public keys."""

    private = Ed25519PrivateKey.generate()
    private_raw = private.private_bytes(
        serialization.Encoding.Raw,
        serialization.PrivateFormat.Raw,
        serialization.NoEncryption(),
    )
    public_raw = private.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )
    return private_raw, public_raw


def generate_aes256_key() -> bytes:
    return os.urandom(32)


def build_reference_riscv_elf() -> bytes:
    """Build a minimal valid ELF64 little-endian RISC-V executable.

    The payload is a monitor stub, not a board-specific supervisor.  It contains
    ``addi a0, x0, 0`` followed by ``jal x0, 0`` (an intentional idle loop).
    Production boards should replace this payload with their signed supervisor.
    """

    entry = 0x80000000
    ident = bytearray(16)
    ident[0:4] = b"\x7fELF"
    ident[4] = 2  # ELFCLASS64
    ident[5] = 1  # little endian
    ident[6] = 1  # ELF version
    header = bytes(ident) + struct.pack(
        "<HHIQQQIHHHHHH",
        2,  # ET_EXEC
        _RISCV_MACHINE,
        1,
        entry,
        64,  # program header table immediately after ELF header
        0,
        0,
        64,
        56,
        1,
        0,
        0,
        0,
    )
    program = struct.pack(
        "<IIQQQQQQ",
        1,  # PT_LOAD
        5,  # PF_R | PF_X
        0x1000,
        entry,
        entry,
        8,
        8,
        0x1000,
    )
    payload = bytearray(header + program)
    payload.extend(b"\x00" * (0x1000 - len(payload)))
    payload.extend(struct.pack("<II", 0x00000513, 0x0000006F))
    return bytes(payload)


def inspect_riscv_elf(payload: bytes) -> dict[str, int]:
    if len(payload) < 64 or payload[:4] != b"\x7fELF":
        raise ValueError("kernel payload is not an ELF file")
    if payload[4] != 2 or payload[5] != 1 or payload[6] != 1:
        raise ValueError("kernel must be ELF64 little-endian version 1")
    e_type, machine, version = struct.unpack_from("<HHI", payload, 16)
    entry = struct.unpack_from("<Q", payload, 24)[0]
    phoff = struct.unpack_from("<Q", payload, 32)[0]
    ehsize, phentsize, phnum = struct.unpack_from("<HHH", payload, 52)
    if e_type != 2 or machine != _RISCV_MACHINE or version != 1:
        raise ValueError("kernel must be an executable RISC-V ELF")
    if ehsize != 64 or phentsize != 56 or phnum < 1:
        raise ValueError("kernel ELF has invalid program-header metadata")
    if phoff + phentsize * phnum > len(payload):
        raise ValueError("kernel ELF program headers are truncated")
    loadable = False
    for index in range(phnum):
        p_type, p_flags = struct.unpack_from("<II", payload, phoff + index * phentsize)
        if p_type == 1 and p_flags & 1:
            loadable = True
            break
    if not loadable:
        raise ValueError("kernel ELF has no executable PT_LOAD segment")
    return {"machine": machine, "entry": entry, "program_headers": phnum}


class SparseMetricCodec3D:
    """Lossless-for-float32 sparse packet for symmetric positive-definite 3D metrics."""

    _record = struct.Struct("<Q6f")

    def encode(self, field: Mapping[Coordinate, Sequence[float]], *, side: int) -> bytes:
        _validate_side(side)
        rows: list[tuple[int, MetricComponents]] = []
        for coordinate, raw in field.items():
            coord = _validate_coordinate(coordinate, side)
            metric = _metric_tuple(raw)
            _validate_spd(metric)
            rows.append((morton3_encode(*coord), metric))
        rows.sort(key=lambda item: item[0])
        raw = bytearray(_METRIC_MAGIC)
        raw.extend(struct.pack("<II", side, len(rows)))
        previous = -1
        for code, metric in rows:
            if code <= previous:
                raise ValueError("metric coordinates must be unique")
            raw.extend(self._record.pack(code, *metric))
            previous = code
        return zlib.compress(bytes(raw), level=9)

    def decode(self, payload: bytes) -> tuple[int, MetricField3D]:
        try:
            raw = zlib.decompress(payload)
        except zlib.error as exc:
            raise ValueError("invalid compressed metric packet") from exc
        if not raw.startswith(_METRIC_MAGIC) or len(raw) < len(_METRIC_MAGIC) + 8:
            raise ValueError("invalid metric packet magic")
        side, count = struct.unpack_from("<II", raw, len(_METRIC_MAGIC))
        _validate_side(side)
        cursor = len(_METRIC_MAGIC) + 8
        expected = cursor + count * self._record.size
        if expected != len(raw):
            raise ValueError("metric packet size/count mismatch")
        field: MetricField3D = {}
        previous = -1
        for _ in range(count):
            unpacked = self._record.unpack_from(raw, cursor)
            cursor += self._record.size
            code = int(unpacked[0])
            if code <= previous:
                raise ValueError("metric Morton stream is not strictly ordered")
            coordinate = morton3_decode(code)
            _validate_coordinate(coordinate, side)
            metric = _metric_tuple(unpacked[1:])
            _validate_spd(metric)
            field[coordinate] = metric
            previous = code
        return side, field


def identity_metric_for_state(field: Mapping[Coordinate, float]) -> MetricField3D:
    identity: MetricComponents = (1.0, 1.0, 1.0, 0.0, 0.0, 0.0)
    return {coordinate: identity for coordinate in field}


def relax_metric_field(
    field: Mapping[Coordinate, Sequence[float]],
    *,
    side: int,
    alpha: float = 0.05,
) -> MetricField3D:
    """Perform a bounded SPD-preserving metric diffusion step.

    This is intentionally described as metric relaxation, not Ricci flow.  Every
    candidate is a convex combination of SPD tensors (explicit neighbours or
    identity), so the SPD invariant is preserved for ``alpha in [0, 1]``.
    """

    _validate_side(side)
    if not math.isfinite(float(alpha)) or not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("alpha must be finite and in [0, 1]")
    current: MetricField3D = {}
    for coordinate, metric in field.items():
        coord = _validate_coordinate(coordinate, side)
        parsed = _metric_tuple(metric)
        _validate_spd(parsed)
        current[coord] = parsed
    identity: MetricComponents = (1.0, 1.0, 1.0, 0.0, 0.0, 0.0)
    offsets = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
    result: MetricField3D = {}
    for coordinate, metric in current.items():
        neighbours: list[MetricComponents] = []
        x, y, z = coordinate
        for dx, dy, dz in offsets:
            candidate = x + dx, y + dy, z + dz
            if all(0 <= axis < side for axis in candidate):
                neighbours.append(current.get(candidate, identity))
        if not neighbours:
            neighbours = [identity]
        average = tuple(
            sum(item[index] for item in neighbours) / len(neighbours) for index in range(6)
        )
        blended = cast(
            MetricComponents,
            tuple((1.0 - alpha) * metric[index] + alpha * average[index] for index in range(6)),
        )
        _validate_spd(blended)
        result[coordinate] = blended
    return result


class FirmwareBuilder:
    """Build exact-size sparse firmware images."""

    def __init__(self) -> None:
        validate_layout()
        self.state_codec = SparseStateCodec3D()
        self.metric_codec = SparseMetricCodec3D()

    def build(
        self,
        path: Path,
        *,
        state: Mapping[Coordinate, float],
        side: int,
        metric: Mapping[Coordinate, Sequence[float]] | None = None,
        supervisor_elf: bytes | None = None,
        signing_private_key: bytes | None = None,
        encryption_key: bytes | None = None,
    ) -> dict[str, object]:
        _validate_side(side)
        if not state:
            raise ValueError("firmware state must be non-empty")
        state_packet = self.state_codec.encode(state, side=side)
        qsol_plain = state_packet.payload
        metric_field = identity_metric_for_state(state) if metric is None else dict(metric)
        metric_plain = self.metric_codec.encode(metric_field, side=side)
        kernel_plain = build_reference_riscv_elf() if supervisor_elf is None else supervisor_elf
        kernel_info = inspect_riscv_elf(kernel_plain)
        trace_anchor = hashlib.sha3_256(
            hashlib.sha256(qsol_plain).digest()
            + hashlib.sha256(metric_plain).digest()
            + hashlib.sha256(kernel_plain).digest()
        ).digest()
        trace_plain = _TRACE_MAGIC + trace_anchor + struct.pack("<Q", 0)

        signed = signing_private_key is not None
        encrypted = encryption_key is not None
        if encryption_key is not None and len(encryption_key) != 32:
            raise ValueError("encryption key must contain exactly 32 bytes")
        salt = os.urandom(16) if encrypted else b""
        sections: dict[str, bytes] = {}
        section_meta: dict[str, dict[str, object]] = {}
        for name, plaintext, codec, encryptable in (
            ("qsol", qsol_plain, "DMOS2", True),
            ("metric", metric_plain, "DMMET1+DEFLATE", True),
            ("kernel", kernel_plain, "ELF64-RISCV", False),
            ("trace", trace_plain, "DMTRC1", False),
        ):
            stored = plaintext
            nonce = b""
            is_encrypted = encrypted and encryptable
            if is_encrypted:
                assert encryption_key is not None
                nonce = os.urandom(12)
                stored = _encrypt_section(name, plaintext, encryption_key, salt, nonce)
            region = _REGION_BY_NAME[name]
            if len(stored) > region.capacity:
                raise ValueError(
                    f"{name} payload uses {len(stored)} bytes but region capacity is {region.capacity}"
                )
            sections[name] = stored
            section_meta[name] = {
                "offset": region.offset,
                "capacity": region.capacity,
                "used_bytes": len(stored),
                "codec": codec,
                "encrypted": is_encrypted,
                "nonce": nonce.hex(),
                "stored_sha256": hashlib.sha256(stored).hexdigest(),
                "content_sha256": hashlib.sha256(plaintext).hexdigest(),
            }

        signer_fingerprint = ""
        private: Ed25519PrivateKey | None = None
        if signing_private_key is not None:
            if len(signing_private_key) != 32:
                raise ValueError("Ed25519 private key must contain exactly 32 raw bytes")
            private = Ed25519PrivateKey.from_private_bytes(signing_private_key)
            public_raw = private.public_key().public_bytes(
                serialization.Encoding.Raw,
                serialization.PublicFormat.Raw,
            )
            signer_fingerprint = hashlib.sha256(public_raw).hexdigest()

        manifest = {
            "format": "DMLAMBDA-1G",
            "version": _CONTAINER_VERSION,
            "image_size": IMAGE_SIZE,
            "side": side,
            "logical_cells": side**3,
            "state_active_cells": len(state),
            "metric_cells": len(metric_field),
            "layout": [region.as_dict() for region in FIRMWARE_LAYOUT],
            "sections": section_meta,
            "kernel": {
                "architecture": "RISC-V",
                "elf_class": 64,
                "machine": kernel_info["machine"],
                "entry": kernel_info["entry"],
                "reference_stub": supervisor_elf is None,
            },
            "crypto": {
                "signature": "Ed25519" if signed else "none",
                "signer_public_key_sha256": signer_fingerprint,
                "section_encryption": "AES-256-GCM" if encrypted else "none",
                "kdf": "HKDF-SHA256" if encrypted else "none",
                "salt": salt.hex(),
            },
            "trace": {
                "algorithm": "SHA3-256 hash chain",
                "anchor": trace_anchor.hex(),
                "externally_anchored_head_required_for_strong_tamper_evidence": True,
            },
            "capability_boundary": {
                "bare_metal_board_support": "board-specific supervisor required",
                "gpu_native_riscv": False,
                "ricci_flow": False,
                "metric_relaxation": True,
                "arbitrary_host_commands": False,
                "self_modifying_executable_code": False,
            },
        }
        manifest_bytes = _canonical_json(manifest)
        genesis_capacity = _REGION_BY_NAME["genesis"].capacity
        if MANIFEST_OFFSET + len(manifest_bytes) > genesis_capacity:
            raise ValueError("manifest does not fit genesis region")
        manifest_digest = hashlib.sha256(manifest_bytes).digest()
        signature = (
            private.sign(_signature_message(manifest_bytes)) if private is not None else bytes(64)
        )
        flags = (_FLAG_SIGNED if signed else 0) | (_FLAG_ENCRYPTED if encrypted else 0)
        header_core = _HEADER_STRUCT.pack(
            _CONTAINER_MAGIC,
            _CONTAINER_VERSION,
            HEADER_SIZE,
            IMAGE_SIZE,
            len(manifest_bytes),
            flags,
            manifest_digest,
            signature,
        )
        header = header_core + bytes(HEADER_SIZE - len(header_core))
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            handle.truncate(IMAGE_SIZE)
            handle.seek(0)
            handle.write(header)
            handle.seek(MANIFEST_OFFSET)
            handle.write(manifest_bytes)
            for name, stored in sections.items():
                handle.seek(_REGION_BY_NAME[name].offset)
                handle.write(stored)
            handle.flush()
            os.fsync(handle.fileno())
        return {
            "path": str(path),
            "image_size": path.stat().st_size,
            "signed": signed,
            "encrypted": encrypted,
            "manifest_sha256": manifest_digest.hex(),
            "trace_anchor": trace_anchor.hex(),
            "sections": section_meta,
        }


class FirmwareImage:
    """Read, verify, decrypt and boot a DMLAMBDA firmware image."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._header, self.manifest, self.manifest_bytes = self._read_header_manifest()

    @property
    def signed(self) -> bool:
        return bool(self._header[5] & _FLAG_SIGNED)

    @property
    def encrypted(self) -> bool:
        return bool(self._header[5] & _FLAG_ENCRYPTED)

    def verify(
        self,
        *,
        public_key: bytes | None = None,
        encryption_key: bytes | None = None,
    ) -> FirmwareVerificationReport:
        if self.path.stat().st_size != IMAGE_SIZE:
            raise ValueError("firmware image is not exactly 1 GiB")
        manifest_digest = hashlib.sha256(self.manifest_bytes).digest()
        if manifest_digest != self._header[6]:
            raise ValueError("firmware manifest checksum mismatch")
        signature_valid = not self.signed
        if self.signed:
            if public_key is None or len(public_key) != 32:
                raise ValueError("a 32-byte Ed25519 public trust anchor is required")
            fingerprint = hashlib.sha256(public_key).hexdigest()
            expected = str(self.manifest["crypto"]["signer_public_key_sha256"])
            if fingerprint != expected:
                raise ValueError("firmware signer fingerprint does not match trust anchor")
            try:
                Ed25519PublicKey.from_public_bytes(public_key).verify(
                    self._header[7], _signature_message(self.manifest_bytes)
                )
            except InvalidSignature as exc:
                raise ValueError("firmware manifest signature is invalid") from exc
            signature_valid = True
        if self.encrypted and (encryption_key is None or len(encryption_key) != 32):
            raise ValueError("a 32-byte AES master key is required for encrypted firmware")

        decoded = self._decoded_sections(encryption_key=encryption_key)
        state_packet, state = SparseStateCodec3D().decode_payload(decoded["qsol"])
        metric_side, metric = SparseMetricCodec3D().decode(decoded["metric"])
        side = int(self.manifest["side"])
        if state_packet.side != side or metric_side != side:
            raise ValueError("firmware section logical-side mismatch")
        kernel = inspect_riscv_elf(decoded["kernel"])
        trace_head = _decode_trace(decoded["trace"])
        if trace_head != str(self.manifest["trace"]["anchor"]):
            raise ValueError("firmware trace anchor mismatch")
        return FirmwareVerificationReport(
            image_size=IMAGE_SIZE,
            signed=self.signed,
            encrypted=self.encrypted,
            signature_valid=signature_valid,
            qsol_cells=len(state),
            metric_cells=len(metric),
            kernel_machine=kernel["machine"],
            kernel_entry=kernel["entry"],
            trace_head=trace_head,
            manifest_sha256=manifest_digest.hex(),
            sections=cast(dict[str, dict[str, object]], self.manifest["sections"]),
        )

    def boot(
        self,
        *,
        public_key: bytes | None = None,
        encryption_key: bytes | None = None,
        max_active_cells: int = 50_000,
    ) -> "FirmwareBootSession":
        verification = self.verify(public_key=public_key, encryption_key=encryption_key)
        decoded = self._decoded_sections(encryption_key=encryption_key)
        state_packet, state = SparseStateCodec3D().decode_payload(decoded["qsol"])
        metric_side, metric = SparseMetricCodec3D().decode(decoded["metric"])
        if metric_side != state_packet.side:
            raise ValueError("firmware metric/state side mismatch")
        config = DrMoagiOSConfig(
            side=state_packet.side,
            max_active_cells=max(max_active_cells, len(state)),
            deep_distiller_max_latent_cells=max(1, min(25_000, max(max_active_cells, len(state)))),
            state_dir=None,
        )
        kernel = DrMoagiOSKernel(config)
        kernel.boot(restore=False)
        kernel.load(state)
        system = SelfOptimizing3DSystem(kernel)
        architecture = SelfEvolving3DArchitecture(system)
        return FirmwareBootSession(
            verification=verification,
            architecture=architecture,
            metric=metric,
            side=state_packet.side,
            trace_head=bytes.fromhex(verification.trace_head),
        )

    def _read_header_manifest(self) -> tuple[tuple[object, ...], dict[str, object], bytes]:
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        with self.path.open("rb") as handle:
            raw_header = handle.read(HEADER_SIZE)
            if len(raw_header) != HEADER_SIZE:
                raise ValueError("truncated firmware header")
            unpacked = _HEADER_STRUCT.unpack_from(raw_header)
            magic, version, header_size, image_size, manifest_length = unpacked[:5]
            if magic != _CONTAINER_MAGIC or version != _CONTAINER_VERSION:
                raise ValueError("unsupported firmware container")
            if header_size != HEADER_SIZE or image_size != IMAGE_SIZE:
                raise ValueError("firmware header size contract mismatch")
            if manifest_length <= 0 or MANIFEST_OFFSET + manifest_length > MIB:
                raise ValueError("firmware manifest length is invalid")
            handle.seek(MANIFEST_OFFSET)
            manifest_bytes = handle.read(manifest_length)
        try:
            parsed = json.loads(manifest_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("firmware manifest is not canonical JSON") from exc
        if not isinstance(parsed, dict):
            raise ValueError("firmware manifest must be an object")
        return cast(tuple[object, ...], unpacked), cast(dict[str, object], parsed), manifest_bytes

    def _decoded_sections(self, *, encryption_key: bytes | None) -> dict[str, bytes]:
        result: dict[str, bytes] = {}
        sections = self.manifest.get("sections")
        if not isinstance(sections, dict):
            raise ValueError("firmware manifest sections are invalid")
        crypto = self.manifest.get("crypto")
        if not isinstance(crypto, dict):
            raise ValueError("firmware crypto metadata is invalid")
        salt_hex = crypto.get("salt", "")
        if not isinstance(salt_hex, str):
            raise ValueError("firmware salt metadata is invalid")
        salt = bytes.fromhex(salt_hex) if salt_hex else b""
        with self.path.open("rb") as handle:
            for name in ("qsol", "metric", "kernel", "trace"):
                raw_meta = sections.get(name)
                if not isinstance(raw_meta, dict):
                    raise ValueError(f"missing firmware section metadata: {name}")
                region = _REGION_BY_NAME[name]
                offset = int(raw_meta.get("offset", -1))
                capacity = int(raw_meta.get("capacity", -1))
                used = int(raw_meta.get("used_bytes", -1))
                if offset != region.offset or capacity != region.capacity or not 0 <= used <= capacity:
                    raise ValueError(f"firmware section layout mismatch: {name}")
                handle.seek(offset)
                stored = handle.read(used)
                if len(stored) != used:
                    raise ValueError(f"truncated firmware section: {name}")
                if hashlib.sha256(stored).hexdigest() != raw_meta.get("stored_sha256"):
                    raise ValueError(f"firmware section checksum mismatch: {name}")
                plaintext = stored
                if bool(raw_meta.get("encrypted")):
                    if encryption_key is None or len(encryption_key) != 32:
                        raise ValueError("firmware encrypted section requires AES key")
                    nonce_hex = raw_meta.get("nonce")
                    if not isinstance(nonce_hex, str):
                        raise ValueError(f"invalid nonce metadata: {name}")
                    plaintext = _decrypt_section(
                        name,
                        stored,
                        encryption_key,
                        salt,
                        bytes.fromhex(nonce_hex),
                    )
                if hashlib.sha256(plaintext).hexdigest() != raw_meta.get("content_sha256"):
                    raise ValueError(f"firmware plaintext checksum mismatch: {name}")
                result[name] = plaintext
        return result


class FirmwareBootSession:
    """Verified runtime session created only after firmware verification succeeds."""

    def __init__(
        self,
        *,
        verification: FirmwareVerificationReport,
        architecture: SelfEvolving3DArchitecture,
        metric: MetricField3D,
        side: int,
        trace_head: bytes,
    ) -> None:
        self.verification = verification
        self.architecture = architecture
        self.metric = dict(metric)
        self.side = side
        self._trace_head = trace_head

    @property
    def trace_head(self) -> str:
        return self._trace_head.hex()

    def run(self, cycles: int) -> FirmwareRunReport:
        autonomic = self.architecture.run_autonomic(cycles)
        before_metric = dict(self.metric)
        metric_committed = False
        if autonomic.state_reports and all(item.committed for item in autonomic.state_reports):
            try:
                candidate = relax_metric_field(self.metric, side=self.side, alpha=0.05)
                for value in candidate.values():
                    _validate_spd(value)
                self.metric = candidate
                metric_committed = True
            except (TypeError, ValueError):
                self.metric = before_metric
        event = {
            "cycles": cycles,
            "state_reports": len(autonomic.state_reports),
            "state_committed": all(item.committed for item in autonomic.state_reports),
            "metric_committed": metric_committed,
            "state_hash": self.architecture.kernel.status()["state_hash"],
        }
        self._trace_head = hashlib.sha3_256(self._trace_head + _canonical_json(event)).digest()
        return FirmwareRunReport(
            verification=self.verification,
            autonomic=autonomic,
            metric_cells=len(self.metric),
            metric_relaxation_committed=metric_committed,
            trace_head=self.trace_head,
        )

    def export_state(self) -> SparseField:
        packet = self.architecture.kernel.export_state_packet()
        return SparseStateCodec3D().decode(packet)


def _encrypt_section(name: str, plaintext: bytes, master: bytes, salt: bytes, nonce: bytes) -> bytes:
    key = _derive_section_key(master, salt, name)
    return AESGCM(key).encrypt(nonce, plaintext, _section_aad(name))


def _decrypt_section(name: str, ciphertext: bytes, master: bytes, salt: bytes, nonce: bytes) -> bytes:
    key = _derive_section_key(master, salt, name)
    try:
        return AESGCM(key).decrypt(nonce, ciphertext, _section_aad(name))
    except Exception as exc:  # cryptography deliberately normalizes auth failures
        raise ValueError(f"firmware section authentication failed: {name}") from exc


def _derive_section_key(master: bytes, salt: bytes, name: str) -> bytes:
    if len(master) != 32:
        raise ValueError("AES master key must contain exactly 32 bytes")
    return HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        info=("DMLAMBDA-SECTION:" + name).encode("ascii"),
    ).derive(master)


def _section_aad(name: str) -> bytes:
    return ("DMLAMBDA:v1:" + name).encode("ascii")


def _signature_message(manifest_bytes: bytes) -> bytes:
    return b"DMLAMBDA-SIGN-v1\x00" + manifest_bytes


def _decode_trace(payload: bytes) -> str:
    if len(payload) != len(_TRACE_MAGIC) + 32 + 8 or not payload.startswith(_TRACE_MAGIC):
        raise ValueError("invalid firmware trace payload")
    count = struct.unpack_from("<Q", payload, len(_TRACE_MAGIC) + 32)[0]
    if count != 0:
        raise ValueError("initial firmware trace record count must be zero")
    return payload[len(_TRACE_MAGIC) : len(_TRACE_MAGIC) + 32].hex()


def _canonical_json(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )


def _validate_side(side: int) -> None:
    if isinstance(side, bool) or not isinstance(side, int) or side <= 0:
        raise ValueError("side must be a positive integer")
    if side > 1_048_576:
        raise ValueError("side exceeds the 20-bit Morton coordinate contract")


def _validate_coordinate(coordinate: Coordinate, side: int) -> Coordinate:
    if (
        not isinstance(coordinate, tuple)
        or len(coordinate) != 3
        or any(isinstance(axis, bool) or not isinstance(axis, int) for axis in coordinate)
    ):
        raise TypeError("coordinates must be integer (x, y, z) tuples")
    if any(axis < 0 or axis >= side for axis in coordinate):
        raise ValueError("coordinate outside logical lattice")
    return coordinate


def _metric_tuple(raw: Sequence[float]) -> MetricComponents:
    if len(raw) != 6:
        raise ValueError("metric must contain six symmetric 3D components")
    values = tuple(float(item) for item in raw)
    if any(not math.isfinite(item) for item in values):
        raise ValueError("metric contains non-finite component")
    return cast(MetricComponents, values)


def _validate_spd(metric: MetricComponents) -> None:
    gxx, gyy, gzz, gxy, gxz, gyz = metric
    minor1 = gxx
    minor2 = gxx * gyy - gxy * gxy
    determinant = (
        gxx * (gyy * gzz - gyz * gyz)
        - gxy * (gxy * gzz - gyz * gxz)
        + gxz * (gxy * gyz - gyy * gxz)
    )
    epsilon = 1.0e-8
    if minor1 <= epsilon or minor2 <= epsilon or determinant <= epsilon:
        raise ValueError("metric tensor is not symmetric positive definite")


__all__ = [
    "FIRMWARE_LAYOUT",
    "FirmwareBootSession",
    "FirmwareBuilder",
    "FirmwareImage",
    "FirmwareRunReport",
    "FirmwareVerificationReport",
    "IMAGE_SIZE",
    "MetricComponents",
    "MetricField3D",
    "SparseMetricCodec3D",
    "build_reference_riscv_elf",
    "generate_aes256_key",
    "generate_ed25519_keypair",
    "identity_metric_for_state",
    "inspect_riscv_elf",
    "relax_metric_field",
    "validate_layout",
]
