"""Bounded PDF-carried bytecode runtime for the Dr Moagi 3D codec loop.

A PDF package is a transport container only. Opening a document never executes it.
Execution requires this runtime to explicitly extract a manifest and DM3D bytecode,
verify integrity, parse a finite instruction set, and run under resource limits.
"""

from __future__ import annotations

import json
import math
import struct
import time
from dataclasses import asdict, dataclass, field
from enum import IntEnum
from hashlib import sha256
from pathlib import Path
from typing import Sequence

PROGRAM_MAGIC = b"DM3DVM1\0"
PROGRAM_VERSION = 1
PROGRAM_HEADER = struct.Struct("<8sHHI")
INSTRUCTION_STRUCT = struct.Struct("<BBBBIII")
PROGRAM_DIGEST_BYTES = 32
MANIFEST_NAME = "manifest.json"
PAYLOAD_NAME = "engine.dm3d"


class Opcode(IntEnum):
    """Finite DM3D instruction set."""

    ENCODE_3D = 0x10
    REFINE_3D = 0x11
    DECODE_3D = 0x20
    REENCODE_3D = 0x21
    ERROR_3D = 0x30
    CYCLE_ERROR = 0x31
    ERROR_MEMORY = 0x32
    CHECK_FIXED = 0x40
    HALT = 0xFF


@dataclass(frozen=True)
class Instruction:
    opcode: Opcode
    dst: int = 0
    src: int = 0
    aux: int = 0
    arg0: int = 0
    arg1: int = 0
    arg2: int = 0


@dataclass(frozen=True)
class ProgramLimits:
    max_instructions: int = 128
    max_voxels: int = 1 << 20
    max_refinement_passes: int = 64
    max_physical_steps: int = 50_000_000

    def validate(self) -> None:
        if self.max_instructions <= 0:
            raise ValueError("max_instructions must be positive")
        if self.max_voxels <= 0:
            raise ValueError("max_voxels must be positive")
        if self.max_refinement_passes <= 0:
            raise ValueError("max_refinement_passes must be positive")
        if self.max_physical_steps <= 0:
            raise ValueError("max_physical_steps must be positive")


@dataclass
class Volume3D:
    nx: int
    ny: int
    nz: int
    values: list[float]

    def __post_init__(self) -> None:
        if min(self.nx, self.ny, self.nz) <= 0:
            raise ValueError("volume dimensions must be positive")
        if len(self.values) != self.voxel_count:
            raise ValueError("volume value count does not match dimensions")
        if not all(math.isfinite(value) for value in self.values):
            raise ValueError("volume values must be finite")

    @property
    def voxel_count(self) -> int:
        return self.nx * self.ny * self.nz

    def index(self, x: int, y: int, z: int) -> int:
        return x + self.nx * (y + self.ny * z)

    def copy(self) -> "Volume3D":
        return Volume3D(self.nx, self.ny, self.nz, list(self.values))


@dataclass
class VmMetrics:
    instructions_executed: int = 0
    physical_steps: int = 0
    voxel_reads: int = 0
    voxel_writes: int = 0
    neighbor_reads: int = 0
    refinement_updates: int = 0
    elapsed_seconds: float = 0.0

    @property
    def physical_steps_per_second(self) -> float:
        if self.elapsed_seconds <= 0.0:
            return 0.0
        return self.physical_steps / self.elapsed_seconds


@dataclass
class VmResult:
    volumes: dict[int, Volume3D]
    scalars: dict[int, float]
    metrics: VmMetrics

    def report(self) -> dict[str, object]:
        return {
            "scalars": {str(key): value for key, value in sorted(self.scalars.items())},
            "volumes": {
                str(key): [value.nx, value.ny, value.nz]
                for key, value in sorted(self.volumes.items())
            },
            "metrics": {
                **asdict(self.metrics),
                "physical_steps_per_second": self.metrics.physical_steps_per_second,
            },
        }


@dataclass(frozen=True)
class PdfPackageManifest:
    manifest_version: int
    engine_format: str
    payload_name: str
    payload_sha256: str
    payload_bytes: int
    instruction_count: int
    execution_policy: str = "explicit-runtime-only"

    def to_bytes(self) -> bytes:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode("utf-8")

    @classmethod
    def from_bytes(cls, payload: bytes) -> "PdfPackageManifest":
        raw = json.loads(payload.decode("utf-8"))
        manifest = cls(**raw)
        if manifest.manifest_version != 1:
            raise ValueError("unsupported PDF package manifest version")
        if manifest.engine_format != "dm3d-bytecode-v1":
            raise ValueError("unsupported PDF package engine format")
        if manifest.execution_policy != "explicit-runtime-only":
            raise ValueError("unsafe or unsupported execution policy")
        if manifest.payload_name != PAYLOAD_NAME:
            raise ValueError("unexpected bytecode payload name")
        if manifest.payload_bytes <= 0 or manifest.instruction_count <= 0:
            raise ValueError("invalid manifest size metadata")
        if len(manifest.payload_sha256) != 64:
            raise ValueError("invalid SHA-256 digest in manifest")
        return manifest


@dataclass
class _Budget:
    limits: ProgramLimits
    metrics: VmMetrics = field(default_factory=VmMetrics)

    def charge(
        self,
        steps: int,
        *,
        reads: int = 0,
        writes: int = 0,
        neighbor_reads: int = 0,
        refinement_updates: int = 0,
    ) -> None:
        if min(steps, reads, writes, neighbor_reads, refinement_updates) < 0:
            raise ValueError("budget charges cannot be negative")
        next_steps = self.metrics.physical_steps + steps
        if next_steps > self.limits.max_physical_steps:
            raise RuntimeError("DM3D physical-step budget exceeded")
        self.metrics.physical_steps = next_steps
        self.metrics.voxel_reads += reads
        self.metrics.voxel_writes += writes
        self.metrics.neighbor_reads += neighbor_reads
        self.metrics.refinement_updates += refinement_updates


def serialize_program(instructions: Sequence[Instruction]) -> bytes:
    if not instructions:
        raise ValueError("DM3D program cannot be empty")
    if len(instructions) > 0xFFFF:
        raise ValueError("DM3D instruction count exceeds uint16")
    if instructions[-1].opcode is not Opcode.HALT:
        raise ValueError("DM3D program must end with HALT")
    if any(instruction.opcode is Opcode.HALT for instruction in instructions[:-1]):
        raise ValueError("DM3D HALT must be the final instruction")

    body = bytearray(PROGRAM_HEADER.pack(PROGRAM_MAGIC, PROGRAM_VERSION, len(instructions), 0))
    for instruction in instructions:
        body.extend(
            INSTRUCTION_STRUCT.pack(
                int(instruction.opcode),
                instruction.dst,
                instruction.src,
                instruction.aux,
                instruction.arg0,
                instruction.arg1,
                instruction.arg2,
            )
        )
    return bytes(body) + sha256(body).digest()


def parse_program(payload: bytes, limits: ProgramLimits | None = None) -> list[Instruction]:
    limits = limits or ProgramLimits()
    limits.validate()
    minimum = PROGRAM_HEADER.size + INSTRUCTION_STRUCT.size + PROGRAM_DIGEST_BYTES
    if len(payload) < minimum:
        raise ValueError("DM3D program is truncated")

    body = payload[:-PROGRAM_DIGEST_BYTES]
    digest = payload[-PROGRAM_DIGEST_BYTES:]
    if sha256(body).digest() != digest:
        raise ValueError("DM3D program SHA-256 mismatch")

    magic, version, count, reserved = PROGRAM_HEADER.unpack_from(body, 0)
    if magic != PROGRAM_MAGIC:
        raise ValueError("invalid DM3D program magic")
    if version != PROGRAM_VERSION:
        raise ValueError("unsupported DM3D program version")
    if reserved != 0:
        raise ValueError("DM3D reserved header field must be zero")
    if count == 0 or count > limits.max_instructions:
        raise ValueError("DM3D instruction count exceeds configured limit")

    expected = PROGRAM_HEADER.size + count * INSTRUCTION_STRUCT.size
    if len(body) != expected:
        raise ValueError("DM3D program size does not match instruction count")

    instructions: list[Instruction] = []
    offset = PROGRAM_HEADER.size
    for index in range(count):
        opcode_raw, dst, src, aux, arg0, arg1, arg2 = INSTRUCTION_STRUCT.unpack_from(body, offset)
        offset += INSTRUCTION_STRUCT.size
        try:
            opcode = Opcode(opcode_raw)
        except ValueError as exc:
            raise ValueError(f"unknown DM3D opcode 0x{opcode_raw:02x}") from exc
        if opcode is Opcode.HALT and index != count - 1:
            raise ValueError("DM3D HALT must be final")
        if opcode is Opcode.REFINE_3D and (arg0 == 0 or arg0 > limits.max_refinement_passes):
            raise ValueError("DM3D refinement pass count exceeds configured limit")
        instructions.append(Instruction(opcode, dst, src, aux, arg0, arg1, arg2))

    if instructions[-1].opcode is not Opcode.HALT:
        raise ValueError("DM3D program is missing HALT")
    return instructions


def canonical_autoencoder_program(*, pool: int = 2, refinement_passes: int = 4) -> bytes:
    if pool <= 0:
        raise ValueError("pool must be positive")
    if refinement_passes <= 0:
        raise ValueError("refinement_passes must be positive")
    parts_per_million = 1_000_000
    instructions = [
        Instruction(Opcode.ENCODE_3D, dst=1, src=0, arg0=pool),
        Instruction(Opcode.REFINE_3D, dst=2, src=1, arg0=refinement_passes),
        Instruction(Opcode.DECODE_3D, dst=3, src=2, arg0=pool),
        Instruction(Opcode.REENCODE_3D, dst=4, src=3, arg0=pool),
        Instruction(Opcode.ERROR_3D, dst=10, src=0, aux=3),
        Instruction(Opcode.CYCLE_ERROR, dst=11, src=2, aux=4),
        Instruction(
            Opcode.ERROR_MEMORY,
            dst=12,
            src=10,
            aux=11,
            arg0=int(0.9 * parts_per_million),
            arg1=int(0.7 * parts_per_million),
            arg2=int(0.3 * parts_per_million),
        ),
        Instruction(Opcode.CHECK_FIXED, dst=13, src=2, aux=4, arg0=1),
        Instruction(Opcode.HALT),
    ]
    return serialize_program(instructions)


def execute_program(
    payload: bytes,
    initial_volume: Volume3D,
    limits: ProgramLimits | None = None,
) -> VmResult:
    limits = limits or ProgramLimits()
    instructions = parse_program(payload, limits)
    if initial_volume.voxel_count > limits.max_voxels:
        raise RuntimeError("initial volume exceeds configured voxel limit")

    volumes: dict[int, Volume3D] = {0: initial_volume.copy()}
    scalars: dict[int, float] = {12: 0.0}
    budget = _Budget(limits)
    started = time.perf_counter()

    for instruction in instructions:
        budget.metrics.instructions_executed += 1
        opcode = instruction.opcode
        if opcode is Opcode.HALT:
            break
        if opcode in (Opcode.ENCODE_3D, Opcode.REENCODE_3D):
            source = _volume(volumes, instruction.src)
            result = _encode(source, instruction.arg0, budget, limits)
            volumes[instruction.dst] = result
        elif opcode is Opcode.REFINE_3D:
            source = _volume(volumes, instruction.src)
            volumes[instruction.dst] = _refine(source, instruction.arg0, budget, limits)
        elif opcode is Opcode.DECODE_3D:
            source = _volume(volumes, instruction.src)
            volumes[instruction.dst] = _decode(source, instruction.arg0, budget, limits)
        elif opcode in (Opcode.ERROR_3D, Opcode.CYCLE_ERROR):
            left = _volume(volumes, instruction.src)
            right = _volume(volumes, instruction.aux)
            scalars[instruction.dst] = _mse(left, right, budget)
        elif opcode is Opcode.ERROR_MEMORY:
            rho = _ppm(instruction.arg0)
            lambda_x = _ppm(instruction.arg1)
            lambda_z = _ppm(instruction.arg2)
            if not 0.0 <= rho <= 1.0:
                raise RuntimeError("rho must lie in [0, 1]")
            error_x = _scalar(scalars, instruction.src)
            error_z = _scalar(scalars, instruction.aux)
            previous = scalars.get(instruction.dst, 0.0)
            scalars[instruction.dst] = rho * previous + (1.0 - rho) * (
                lambda_x * error_x + lambda_z * error_z
            )
            budget.charge(8)
        elif opcode is Opcode.CHECK_FIXED:
            left = _volume(volumes, instruction.src)
            right = _volume(volumes, instruction.aux)
            epsilon = instruction.arg0 * 1e-12
            error = _mse(left, right, budget)
            scalars[instruction.dst] = 1.0 if error <= epsilon else 0.0
        else:  # pragma: no cover - parse_program rejects unknown opcodes
            raise RuntimeError(f"unhandled DM3D opcode {opcode}")

    budget.metrics.elapsed_seconds = time.perf_counter() - started
    return VmResult(volumes, scalars, budget.metrics)


def make_seed_volume(size: int = 16) -> Volume3D:
    if size <= 0:
        raise ValueError("size must be positive")
    center = (size - 1) / 2.0
    values: list[float] = []
    for z in range(size):
        dz = (z - center) / size
        for y in range(size):
            dy = (y - center) / size
            for x in range(size):
                dx = (x - center) / size
                radius = math.sqrt(dx * dx + dy * dy + dz * dz)
                value = math.sin(12.0 * radius) * math.exp(-2.2 * radius)
                value += 0.25 * math.cos(5.0 * (dx - dy + dz))
                values.append(value)
    return Volume3D(size, size, size, values)


def build_pdf_package(
    path: str | Path,
    program: bytes,
    *,
    title: str = "Dr Moagi 3D Bytecode Runtime",
) -> PdfPackageManifest:
    """Create a PDF transport package with verified DM3D bytecode attachments.

    PyMuPDF is imported lazily so the core VM remains dependency-free.
    """

    fitz = _require_pymupdf()
    instructions = parse_program(program)
    manifest = PdfPackageManifest(
        manifest_version=1,
        engine_format="dm3d-bytecode-v1",
        payload_name=PAYLOAD_NAME,
        payload_sha256=sha256(program).hexdigest(),
        payload_bytes=len(program),
        instruction_count=len(instructions),
    )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    document = fitz.open()
    page = document.new_page(width=842, height=595)
    text = (
        f"{title}\n\n"
        "Execution policy: explicit runtime only\n"
        f"Payload: {PAYLOAD_NAME}\n"
        f"Instructions: {manifest.instruction_count}\n"
        f"SHA-256: {manifest.payload_sha256}\n\n"
        "Pipeline:\n"
        "PDF -> verify manifest -> parse DM3D bytecode -> bounded 3D VM -> metrics\n"
    )
    page.insert_textbox(fitz.Rect(48, 48, 794, 540), text, fontsize=12)
    document.embfile_add(
        MANIFEST_NAME,
        manifest.to_bytes(),
        filename=MANIFEST_NAME,
        ufilename=MANIFEST_NAME,
        desc="Dr Moagi PDF runtime manifest",
    )
    document.embfile_add(
        PAYLOAD_NAME,
        program,
        filename=PAYLOAD_NAME,
        ufilename=PAYLOAD_NAME,
        desc="Verified DM3D bytecode payload",
    )
    document.save(str(output), deflate=True, garbage=4)
    document.close()
    return manifest


def load_pdf_package(path: str | Path) -> tuple[PdfPackageManifest, bytes]:
    fitz = _require_pymupdf()
    document = fitz.open(str(path))
    try:
        names = set(document.embfile_names())
        if MANIFEST_NAME not in names or PAYLOAD_NAME not in names:
            raise ValueError("PDF package is missing required attachments")
        manifest = PdfPackageManifest.from_bytes(document.embfile_get(MANIFEST_NAME))
        payload = bytes(document.embfile_get(PAYLOAD_NAME))
    finally:
        document.close()

    if len(payload) != manifest.payload_bytes:
        raise ValueError("PDF bytecode payload length does not match manifest")
    if sha256(payload).hexdigest() != manifest.payload_sha256:
        raise ValueError("PDF bytecode payload SHA-256 does not match manifest")
    instructions = parse_program(payload)
    if len(instructions) != manifest.instruction_count:
        raise ValueError("PDF bytecode instruction count does not match manifest")
    return manifest, payload


def run_pdf_package(
    path: str | Path,
    initial_volume: Volume3D,
    limits: ProgramLimits | None = None,
) -> VmResult:
    _, payload = load_pdf_package(path)
    return execute_program(payload, initial_volume, limits)


def _encode(
    source: Volume3D,
    pool: int,
    budget: _Budget,
    limits: ProgramLimits,
) -> Volume3D:
    if pool <= 0:
        raise RuntimeError("pool factor must be positive")
    if source.nx % pool or source.ny % pool or source.nz % pool:
        raise RuntimeError("volume dimensions must be divisible by pool factor")
    nx, ny, nz = source.nx // pool, source.ny // pool, source.nz // pool
    output_count = nx * ny * nz
    if output_count > limits.max_voxels:
        raise RuntimeError("encoded volume exceeds configured voxel limit")
    block = pool**3
    values = [0.0] * output_count
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                total = 0.0
                for dz in range(pool):
                    for dy in range(pool):
                        for dx in range(pool):
                            total += source.values[
                                source.index(x * pool + dx, y * pool + dy, z * pool + dz)
                            ]
                values[x + nx * (y + ny * z)] = total / block
    budget.charge(source.voxel_count + output_count, reads=source.voxel_count, writes=output_count)
    return Volume3D(nx, ny, nz, values)


def _decode(
    source: Volume3D,
    pool: int,
    budget: _Budget,
    limits: ProgramLimits,
) -> Volume3D:
    if pool <= 0:
        raise RuntimeError("pool factor must be positive")
    nx, ny, nz = source.nx * pool, source.ny * pool, source.nz * pool
    count = nx * ny * nz
    if count > limits.max_voxels:
        raise RuntimeError("decoded volume exceeds configured voxel limit")
    values = [0.0] * count
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                values[x + nx * (y + ny * z)] = source.values[
                    source.index(x // pool, y // pool, z // pool)
                ]
    budget.charge(count * 2, reads=count, writes=count)
    return Volume3D(nx, ny, nz, values)


def _refine(
    source: Volume3D,
    passes: int,
    budget: _Budget,
    limits: ProgramLimits,
) -> Volume3D:
    if passes <= 0 or passes > limits.max_refinement_passes:
        raise RuntimeError("refinement passes exceed configured limit")
    current = list(source.values)
    nx, ny, nz = source.nx, source.ny, source.nz
    count = source.voxel_count
    directions = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))

    for _ in range(passes):
        mean = sum(current) / count
        next_values = [0.0] * count
        valid_neighbor_reads = 0
        for z in range(nz):
            for y in range(ny):
                for x in range(nx):
                    index = x + nx * (y + ny * z)
                    neighbor_total = 0.0
                    neighbor_count = 0
                    for dx, dy, dz in directions:
                        xx, yy, zz = x + dx, y + dy, z + dz
                        if 0 <= xx < nx and 0 <= yy < ny and 0 <= zz < nz:
                            neighbor_total += current[xx + nx * (yy + ny * zz)]
                            neighbor_count += 1
                    neighbor_mean = (
                        neighbor_total / neighbor_count if neighbor_count else current[index]
                    )
                    next_values[index] = 0.88 * current[index] + 0.10 * neighbor_mean + 0.02 * mean
                    valid_neighbor_reads += neighbor_count
        budget.charge(
            count * 5 + valid_neighbor_reads,
            reads=count * 2 + valid_neighbor_reads,
            writes=count,
            neighbor_reads=valid_neighbor_reads,
            refinement_updates=count,
        )
        current = next_values
    return Volume3D(nx, ny, nz, current)


def _mse(left: Volume3D, right: Volume3D, budget: _Budget) -> float:
    if (left.nx, left.ny, left.nz) != (right.nx, right.ny, right.nz):
        raise RuntimeError("MSE operands must have identical dimensions")
    total = 0.0
    for a, b in zip(left.values, right.values):
        delta = a - b
        total += delta * delta
    count = left.voxel_count
    budget.charge(count * 4, reads=count * 2)
    return total / count


def _volume(volumes: dict[int, Volume3D], register: int) -> Volume3D:
    try:
        return volumes[register]
    except KeyError as exc:
        raise RuntimeError(f"DM3D volume register r{register} is uninitialized") from exc


def _scalar(scalars: dict[int, float], register: int) -> float:
    try:
        return scalars[register]
    except KeyError as exc:
        raise RuntimeError(f"DM3D scalar register s{register} is uninitialized") from exc


def _ppm(value: int) -> float:
    return value / 1_000_000.0


def _require_pymupdf():
    try:
        import fitz
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise RuntimeError(
            "PDF packaging requires PyMuPDF; install jarvisx[pdf] or pymupdf>=1.24"
        ) from exc
    return fitz


__all__ = [
    "Instruction",
    "Opcode",
    "PdfPackageManifest",
    "ProgramLimits",
    "VmMetrics",
    "VmResult",
    "Volume3D",
    "build_pdf_package",
    "canonical_autoencoder_program",
    "execute_program",
    "load_pdf_package",
    "make_seed_volume",
    "parse_program",
    "run_pdf_package",
    "serialize_program",
]
