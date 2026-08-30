"""PDF transport for the bounded DM3D 3D-animation auto-execution loop."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path

from .dr_moagi_animation_loop import (
    AutoLoopLimits,
    AutoLoopResult,
    execute_auto_loop,
    parse_auto_loop_program,
)
from .dr_moagi_pdf_bytecode import ProgramLimits, Volume3D, parse_program

ANIMATION_MANIFEST_NAME = "manifest.json"
ANIMATION_PAYLOAD_NAME = "animation.dm3dloop"
ANIMATION_ENGINE_FORMAT = "dm3d-animation-loop-v1"


@dataclass(frozen=True)
class AnimationPdfManifest:
    manifest_version: int
    engine_format: str
    payload_name: str
    payload_sha256: str
    payload_bytes: int
    cycles: int
    inner_instruction_count: int
    execution_policy: str = "explicit-runtime-only"

    def to_bytes(self) -> bytes:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode("utf-8")

    @classmethod
    def from_bytes(cls, payload: bytes) -> "AnimationPdfManifest":
        manifest = cls(**json.loads(payload.decode("utf-8")))
        if manifest.manifest_version != 1:
            raise ValueError("unsupported animation PDF manifest version")
        if manifest.engine_format != ANIMATION_ENGINE_FORMAT:
            raise ValueError("unsupported animation PDF engine format")
        if manifest.payload_name != ANIMATION_PAYLOAD_NAME:
            raise ValueError("unexpected animation PDF payload name")
        if manifest.execution_policy != "explicit-runtime-only":
            raise ValueError("unsafe or unsupported animation execution policy")
        if manifest.payload_bytes <= 0 or manifest.cycles <= 0:
            raise ValueError("invalid animation PDF size or cycle metadata")
        if manifest.inner_instruction_count <= 0:
            raise ValueError("invalid animation PDF instruction metadata")
        if len(manifest.payload_sha256) != 64:
            raise ValueError("invalid animation PDF SHA-256 digest")
        return manifest


def build_animation_pdf_package(
    path: str | Path,
    payload: bytes,
    *,
    title: str = "Dr Moagi 3D Animation Auto-Execution Loop",
) -> AnimationPdfManifest:
    """Write verified DM3D-LOOP bytecode as an inert PDF attachment package."""

    loop = parse_auto_loop_program(payload)
    inner_instructions = parse_program(loop.inner_program)
    manifest = AnimationPdfManifest(
        manifest_version=1,
        engine_format=ANIMATION_ENGINE_FORMAT,
        payload_name=ANIMATION_PAYLOAD_NAME,
        payload_sha256=sha256(payload).hexdigest(),
        payload_bytes=len(payload),
        cycles=loop.cycles,
        inner_instruction_count=len(inner_instructions),
    )
    fitz = _require_pymupdf()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    document = fitz.open()
    page = document.new_page(width=842, height=595)
    text = (
        f"{title}\n\n"
        "Execution policy: explicit runtime only\n"
        f"Payload: {ANIMATION_PAYLOAD_NAME}\n"
        f"Cycles: {manifest.cycles}\n"
        f"Inner instructions: {manifest.inner_instruction_count}\n"
        f"SHA-256: {manifest.payload_sha256}\n\n"
        "Pipeline:\n"
        "PDF -> verify loop -> verify DM3D -> bounded cycles -> volumetric frames\n"
    )
    page.insert_textbox(fitz.Rect(48, 48, 794, 540), text, fontsize=12)
    document.embfile_add(
        ANIMATION_MANIFEST_NAME,
        manifest.to_bytes(),
        filename=ANIMATION_MANIFEST_NAME,
        ufilename=ANIMATION_MANIFEST_NAME,
        desc="Dr Moagi animation auto-loop manifest",
    )
    document.embfile_add(
        ANIMATION_PAYLOAD_NAME,
        payload,
        filename=ANIMATION_PAYLOAD_NAME,
        ufilename=ANIMATION_PAYLOAD_NAME,
        desc="Verified DM3D animation loop payload",
    )
    document.save(str(output), deflate=True, garbage=4)
    document.close()
    return manifest


def load_animation_pdf_package(path: str | Path) -> tuple[AnimationPdfManifest, bytes]:
    """Extract and verify the inert animation loop payload from a PDF package."""

    fitz = _require_pymupdf()
    document = fitz.open(str(path))
    try:
        names = set(document.embfile_names())
        if ANIMATION_MANIFEST_NAME not in names or ANIMATION_PAYLOAD_NAME not in names:
            raise ValueError("animation PDF package is missing required attachments")
        manifest = AnimationPdfManifest.from_bytes(
            document.embfile_get(ANIMATION_MANIFEST_NAME)
        )
        payload = bytes(document.embfile_get(ANIMATION_PAYLOAD_NAME))
    finally:
        document.close()

    if len(payload) != manifest.payload_bytes:
        raise ValueError("animation PDF payload length does not match manifest")
    if sha256(payload).hexdigest() != manifest.payload_sha256:
        raise ValueError("animation PDF payload SHA-256 does not match manifest")
    loop = parse_auto_loop_program(payload)
    if loop.cycles != manifest.cycles:
        raise ValueError("animation PDF cycle count does not match manifest")
    if len(parse_program(loop.inner_program)) != manifest.inner_instruction_count:
        raise ValueError("animation PDF instruction count does not match manifest")
    return manifest, payload


def run_animation_pdf_package(
    path: str | Path,
    initial_volume: Volume3D,
    *,
    loop_limits: AutoLoopLimits | None = None,
    vm_limits: ProgramLimits | None = None,
) -> AutoLoopResult:
    """Explicitly verify and execute a PDF-carried DM3D animation loop."""

    _, payload = load_animation_pdf_package(path)
    return execute_auto_loop(
        payload,
        initial_volume,
        loop_limits=loop_limits,
        vm_limits=vm_limits,
    )


def _require_pymupdf():
    try:
        import fitz
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise RuntimeError(
            "animation PDF packaging requires PyMuPDF; install jarvisx[pdf] or pymupdf>=1.24"
        ) from exc
    return fitz


__all__ = [
    "AnimationPdfManifest",
    "build_animation_pdf_package",
    "load_animation_pdf_package",
    "run_animation_pdf_package",
]
