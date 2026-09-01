"""Jarvis-X MMVM full-stack auto-encoding/decoding kernel.

The MMVM kernel turns the research architecture into concrete operating
primitives: queued tasks, a lossless byte codec plus bounded latent state,
Lambda policy projection, persistent Omega telemetry, sparse 3D virtual
memory, deterministic multimedia generators, and transactional commit.

The virtual memory contract is exact: ``100_000 ** 3 == 10**15`` logical
byte-addressable voxels (1,000,000 decimal GB). Only committed objects are
physically materialized in SQLite.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import os
import shutil
import sqlite3
import struct
import subprocess
import tempfile
import threading
import time
import uuid
import wave
import zlib
from collections import deque
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Sequence


class TaskState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMMITTED = "committed"
    REJECTED = "rejected"
    FAILED = "failed"


@dataclass(frozen=True)
class MMVMConfig:
    side: int = 100_000
    latent_dim: int = 128
    refinement_steps: int = 3
    max_payload_bytes: int = 32 * 1024 * 1024
    max_queue: int = 1024
    omega_window: int = 256
    zlib_level: int = 6

    def __post_init__(self) -> None:
        if self.side <= 0:
            raise ValueError("side must be positive")
        if self.latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if self.refinement_steps < 0:
            raise ValueError("refinement_steps must be non-negative")
        if self.max_payload_bytes <= 0 or self.max_queue <= 0:
            raise ValueError("resource limits must be positive")
        if not 0 <= self.zlib_level <= 9:
            raise ValueError("zlib_level must be between 0 and 9")

    @property
    def virtual_cells(self) -> int:
        return self.side**3

    @property
    def virtual_bytes(self) -> int:
        return self.virtual_cells

    @property
    def virtual_decimal_gb(self) -> int:
        return self.virtual_bytes // 1_000_000_000


@dataclass(frozen=True)
class MemoryAddress:
    index: int
    x: int
    y: int
    z: int


@dataclass(frozen=True)
class CodecPacket:
    codec: str
    source_sha256: str
    source_size: int
    encoded: bytes
    latent: tuple[float, ...]


@dataclass(frozen=True)
class LambdaDecision:
    accepted: bool
    reason: str | None
    checksum_valid: bool
    finite_latent: bool
    bounded_latent: bool
    payload_size_valid: bool


@dataclass(frozen=True)
class OmegaSnapshot:
    commits: int
    rejects: int
    failures: int
    mean_activity: float
    last_error: float
    activation: float


@dataclass
class MMVMTask:
    task_id: str
    modality: str
    target: str | None
    payload: bytes
    created_at: float
    state: TaskState = TaskState.QUEUED
    error: str | None = None
    object_id: str | None = None
    artifact_id: str | None = None
    completed_at: float | None = None

    def public(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("payload", None)
        data["state"] = self.state.value
        data["payload_bytes"] = len(self.payload)
        return data


@dataclass(frozen=True)
class CycleMetrics:
    cycle: int
    task_id: str
    state: str
    source_bytes: int
    encoded_bytes: int
    compression_ratio: float
    xi_norm: float
    xi_dot: float
    reconstruction_error: float
    lambda_accepted: bool
    memory_address: MemoryAddress | None
    artifact_id: str | None
    elapsed_ms: float

    def public(self) -> dict[str, Any]:
        data = asdict(self)
        if self.memory_address is not None:
            data["memory_address"] = asdict(self.memory_address)
        return data


@dataclass(frozen=True)
class GeneratedArtifact:
    artifact_id: str
    modality: str
    media_type: str
    filename: str
    payload: bytes
    sha256: str


class MMVMCodec:
    """Lossless byte codec with a deterministic bounded latent side-channel."""

    def __init__(self, config: MMVMConfig) -> None:
        self.config = config

    def encode(self, payload: bytes) -> CodecPacket:
        if len(payload) > self.config.max_payload_bytes:
            raise ValueError("payload exceeds configured byte budget")
        checksum = hashlib.sha256(payload).hexdigest()
        encoded = zlib.compress(payload, self.config.zlib_level)
        latent = self._latent(payload)
        return CodecPacket(
            codec="zlib",
            source_sha256=checksum,
            source_size=len(payload),
            encoded=encoded,
            latent=latent,
        )

    def decode(self, packet: CodecPacket) -> bytes:
        if packet.codec != "zlib":
            raise ValueError(f"unsupported codec: {packet.codec}")
        decoded = zlib.decompress(packet.encoded)
        if len(decoded) != packet.source_size:
            raise ValueError("decoded length does not match packet contract")
        return decoded

    def refine(
        self,
        latent: Sequence[float],
        *,
        omega: float,
        cycle: int,
    ) -> tuple[tuple[float, ...], float]:
        current = tuple(float(value) for value in latent)
        if not current:
            return (), 0.0
        initial = current
        n = len(current)
        for step in range(self.config.refinement_steps):
            candidate: list[float] = []
            for index, center in enumerate(current):
                left = current[(index - 1) % n]
                right = current[(index + 1) % n]
                drive = 0.025 * math.sin((cycle + step + 1) * 0.31 + index * 0.07)
                value = math.tanh(
                    0.18 * left + 0.64 * center + 0.18 * right + 0.025 * omega + drive
                )
                candidate.append(max(-1.0, min(1.0, value)))
            current = tuple(candidate)
        xi_dot = math.sqrt(
            sum((after - before) ** 2 for before, after in zip(initial, current)) / n
        )
        return current, xi_dot

    def _latent(self, payload: bytes) -> tuple[float, ...]:
        digest = hashlib.shake_256(payload).digest(self.config.latent_dim)
        if not payload:
            mean = 0.0
            spread = 0.0
        else:
            mean = sum(payload) / (255.0 * len(payload))
            spread = (
                sum(((value / 255.0) - mean) ** 2 for value in payload) / len(payload)
            ) ** 0.5
        latent = []
        for index, value in enumerate(digest):
            base = (value / 127.5) - 1.0
            modulation = 0.08 * math.sin(index * 0.17 + mean * math.pi)
            modulation += 0.04 * spread * math.cos(index * 0.11)
            latent.append(max(-1.0, min(1.0, base + modulation)))
        return tuple(latent)


class LambdaPolicy:
    """Constitutional validation boundary for an MMVM transaction."""

    def __init__(self, config: MMVMConfig) -> None:
        self.config = config

    def validate(
        self,
        packet: CodecPacket,
        latent: Sequence[float],
        decoded: bytes,
    ) -> LambdaDecision:
        checksum_valid = hashlib.sha256(decoded).hexdigest() == packet.source_sha256
        finite_latent = all(math.isfinite(float(value)) for value in latent)
        bounded_latent = all(-1.0 <= float(value) <= 1.0 for value in latent)
        payload_size_valid = (
            len(decoded) == packet.source_size
            and len(decoded) <= self.config.max_payload_bytes
        )
        accepted = checksum_valid and finite_latent and bounded_latent and payload_size_valid
        reason = None
        if not checksum_valid:
            reason = "checksum mismatch"
        elif not finite_latent:
            reason = "non-finite latent state"
        elif not bounded_latent:
            reason = "latent state escaped Lambda bounds"
        elif not payload_size_valid:
            reason = "payload violates resource contract"
        return LambdaDecision(
            accepted=accepted,
            reason=reason,
            checksum_valid=checksum_valid,
            finite_latent=finite_latent,
            bounded_latent=bounded_latent,
            payload_size_valid=payload_size_valid,
        )


class SparseVoxelMemory:
    """Persistent sparse backing store for the exact 100000^3 logical lattice."""

    def __init__(self, path: str | os.PathLike[str], config: MMVMConfig) -> None:
        self.path = str(path)
        self.config = config
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._initialize()

    def _initialize(self) -> None:
        with self._lock, self._conn:
            self._conn.executescript(
                """
                PRAGMA journal_mode=WAL;
                PRAGMA synchronous=NORMAL;
                CREATE TABLE IF NOT EXISTS objects (
                    object_id TEXT PRIMARY KEY,
                    address INTEGER NOT NULL UNIQUE,
                    x INTEGER NOT NULL,
                    y INTEGER NOT NULL,
                    z INTEGER NOT NULL,
                    modality TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    source_size INTEGER NOT NULL,
                    codec TEXT NOT NULL,
                    encoded BLOB NOT NULL,
                    latent_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS events (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    cycle INTEGER NOT NULL,
                    task_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS artifacts (
                    artifact_id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    modality TEXT NOT NULL,
                    media_type TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    size INTEGER NOT NULL,
                    payload BLOB NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_events_cycle ON events(cycle);
                CREATE INDEX IF NOT EXISTS idx_objects_address ON objects(address);
                """
            )

    def index_to_address(self, index: int) -> MemoryAddress:
        if index < 0 or index >= self.config.virtual_cells:
            raise ValueError("address index outside virtual lattice")
        side = self.config.side
        x = index % side
        y = (index // side) % side
        z = index // (side * side)
        return MemoryAddress(index=index, x=x, y=y, z=z)

    def coordinates_to_index(self, x: int, y: int, z: int) -> int:
        side = self.config.side
        if not (0 <= x < side and 0 <= y < side and 0 <= z < side):
            raise ValueError("coordinates outside virtual lattice")
        return x + side * (y + side * z)

    def commit_object(
        self,
        *,
        modality: str,
        packet: CodecPacket,
        latent: Sequence[float],
        metadata: dict[str, Any],
    ) -> tuple[str, MemoryAddress]:
        object_id = hashlib.sha256(
            modality.encode("utf-8") + b"\0" + bytes.fromhex(packet.source_sha256)
        ).hexdigest()
        with self._lock, self._conn:
            existing = self._conn.execute(
                "SELECT address, x, y, z FROM objects WHERE object_id = ?", (object_id,)
            ).fetchone()
            if existing is not None:
                return object_id, MemoryAddress(
                    index=int(existing["address"]),
                    x=int(existing["x"]),
                    y=int(existing["y"]),
                    z=int(existing["z"]),
                )

            address = self._allocate_address(object_id)
            self._conn.execute(
                """
                INSERT INTO objects(
                    object_id, address, x, y, z, modality, sha256, source_size,
                    codec, encoded, latent_json, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    object_id,
                    address.index,
                    address.x,
                    address.y,
                    address.z,
                    modality,
                    packet.source_sha256,
                    packet.source_size,
                    packet.codec,
                    packet.encoded,
                    json.dumps([float(value) for value in latent], separators=(",", ":")),
                    json.dumps(metadata, sort_keys=True, separators=(",", ":")),
                    time.time(),
                ),
            )
            return object_id, address

    def _allocate_address(self, object_id: str) -> MemoryAddress:
        seed = int.from_bytes(hashlib.sha256(object_id.encode("ascii")).digest()[:8], "big")
        start = seed % self.config.virtual_cells
        for offset in range(4096):
            candidate = (start + offset) % self.config.virtual_cells
            occupied = self._conn.execute(
                "SELECT object_id FROM objects WHERE address = ?", (candidate,)
            ).fetchone()
            if occupied is None or occupied["object_id"] == object_id:
                return self.index_to_address(candidate)
        raise RuntimeError("unable to allocate collision-free sparse address")

    def fetch_object(self, object_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM objects WHERE object_id = ?", (object_id,)
            ).fetchone()
        if row is None:
            return None
        return {
            "object_id": row["object_id"],
            "address": {
                "index": int(row["address"]),
                "x": int(row["x"]),
                "y": int(row["y"]),
                "z": int(row["z"]),
            },
            "modality": row["modality"],
            "sha256": row["sha256"],
            "source_size": int(row["source_size"]),
            "codec": row["codec"],
            "encoded": bytes(row["encoded"]),
            "latent": tuple(json.loads(row["latent_json"])),
            "metadata": json.loads(row["metadata_json"]),
        }

    def record_event(self, cycle: int, task_id: str, kind: str, payload: dict[str, Any]) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO events(cycle, task_id, kind, payload_json, created_at) VALUES (?, ?, ?, ?, ?)",
                (
                    cycle,
                    task_id,
                    kind,
                    json.dumps(payload, sort_keys=True, separators=(",", ":")),
                    time.time(),
                ),
            )

    def omega_snapshot(self, window: int) -> OmegaSnapshot:
        with self._lock:
            rows = self._conn.execute(
                "SELECT kind, payload_json FROM events ORDER BY seq DESC LIMIT ?", (window,)
            ).fetchall()
        commits = rejects = failures = 0
        activities: list[float] = []
        last_error = 0.0
        for row in rows:
            kind = str(row["kind"])
            payload = json.loads(row["payload_json"])
            if kind == "commit":
                commits += 1
            elif kind == "reject":
                rejects += 1
            elif kind == "failure":
                failures += 1
            if "xi_dot" in payload:
                activities.append(float(payload["xi_dot"]))
            if "reconstruction_error" in payload and last_error == 0.0:
                last_error = float(payload["reconstruction_error"])
        total = max(1, commits + rejects + failures)
        mean_activity = sum(activities) / len(activities) if activities else 0.0
        activation = max(0.0, min(1.0, commits / total * 0.7 + mean_activity * 0.3))
        return OmegaSnapshot(
            commits=commits,
            rejects=rejects,
            failures=failures,
            mean_activity=mean_activity,
            last_error=last_error,
            activation=activation,
        )

    def store_artifact(self, task_id: str, artifact: GeneratedArtifact) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO artifacts(
                    artifact_id, task_id, modality, media_type, filename,
                    sha256, size, payload, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    artifact.artifact_id,
                    task_id,
                    artifact.modality,
                    artifact.media_type,
                    artifact.filename,
                    artifact.sha256,
                    len(artifact.payload),
                    artifact.payload,
                    time.time(),
                ),
            )

    def fetch_artifact(self, artifact_id: str) -> GeneratedArtifact | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM artifacts WHERE artifact_id = ?", (artifact_id,)
            ).fetchone()
        if row is None:
            return None
        return GeneratedArtifact(
            artifact_id=row["artifact_id"],
            modality=row["modality"],
            media_type=row["media_type"],
            filename=row["filename"],
            payload=bytes(row["payload"]),
            sha256=row["sha256"],
        )

    def stats(self) -> dict[str, Any]:
        with self._lock:
            objects = int(self._conn.execute("SELECT COUNT(*) FROM objects").fetchone()[0])
            artifacts = int(self._conn.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0])
            events = int(self._conn.execute("SELECT COUNT(*) FROM events").fetchone()[0])
            encoded_bytes = int(
                self._conn.execute("SELECT COALESCE(SUM(length(encoded)), 0) FROM objects").fetchone()[0]
            )
        physical_bytes = os.path.getsize(self.path) if os.path.exists(self.path) else 0
        return {
            "virtual_side": self.config.side,
            "virtual_cells": self.config.virtual_cells,
            "virtual_bytes": self.config.virtual_bytes,
            "virtual_decimal_gb": self.config.virtual_decimal_gb,
            "resident_objects": objects,
            "artifacts": artifacts,
            "events": events,
            "encoded_object_bytes": encoded_bytes,
            "sqlite_physical_bytes": physical_bytes,
        }

    def close(self) -> None:
        with self._lock:
            self._conn.close()


class MultimediaGenerator:
    """Deterministic local decoders for text, image, audio, video and 3D state."""

    SUPPORTED = {"text", "image", "audio", "video", "3d"}

    def generate(
        self,
        target: str,
        source: bytes,
        latent: Sequence[float],
        cycle: int,
    ) -> GeneratedArtifact:
        target = target.lower()
        if target not in self.SUPPORTED:
            raise ValueError(f"unsupported generation target: {target}")
        prompt = source.decode("utf-8", errors="replace")[:2000]
        if target == "text":
            payload = self._text(prompt, latent, cycle)
            media_type, suffix = "text/plain; charset=utf-8", "txt"
        elif target == "image":
            payload = self._svg(prompt, latent, cycle)
            media_type, suffix = "image/svg+xml", "svg"
        elif target == "audio":
            payload = self._wav(latent)
            media_type, suffix = "audio/wav", "wav"
        elif target == "video":
            payload = self._video(latent, cycle)
            media_type, suffix = "video/mp4", "mp4"
        else:
            payload = self._voxel_json(prompt, latent, cycle)
            media_type, suffix = "application/json", "json"
        sha256 = hashlib.sha256(payload).hexdigest()
        artifact_id = hashlib.sha256(
            target.encode("utf-8") + b"\0" + bytes.fromhex(sha256)
        ).hexdigest()[:24]
        return GeneratedArtifact(
            artifact_id=artifact_id,
            modality=target,
            media_type=media_type,
            filename=f"mmvm-{artifact_id}.{suffix}",
            payload=payload,
            sha256=sha256,
        )

    @staticmethod
    def _text(prompt: str, latent: Sequence[float], cycle: int) -> bytes:
        dominant = sorted(range(len(latent)), key=lambda index: abs(latent[index]), reverse=True)[:8]
        summary = {
            "cycle": cycle,
            "prompt": prompt,
            "latent_energy": round(sum(value * value for value in latent) / max(1, len(latent)), 8),
            "dominant_dimensions": dominant,
            "status": "decoded",
        }
        return (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode("utf-8")

    @staticmethod
    def _svg(prompt: str, latent: Sequence[float], cycle: int) -> bytes:
        esc = (
            prompt.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )
        circles = []
        for index in range(24):
            value = latent[index % len(latent)] if latent else 0.0
            x = 40 + ((index * 71 + cycle * 13) % 560)
            y = 45 + ((index * 47 + cycle * 7) % 250)
            radius = 5 + int(abs(value) * 22)
            hue = int((180 + 150 * value + index * 11) % 360)
            opacity = 0.25 + 0.65 * abs(value)
            circles.append(
                f'<circle cx="{x}" cy="{y}" r="{radius}" fill="hsl({hue} 90% 60%)" opacity="{opacity:.3f}"/>'
            )
        svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="640" height="360" viewBox="0 0 640 360">
<defs><radialGradient id="g"><stop stop-color="#071d2c"/><stop offset="1" stop-color="#020306"/></radialGradient></defs>
<rect width="640" height="360" fill="url(#g)"/>
<g style="mix-blend-mode:screen">{''.join(circles)}</g>
<text x="24" y="326" fill="#d9f7ff" font-family="monospace" font-size="14">MMVM cycle {cycle}</text>
<text x="24" y="346" fill="#75ffb0" font-family="monospace" font-size="11">{esc[:84]}</text>
</svg>"""
        return svg.encode("utf-8")

    @staticmethod
    def _wav(latent: Sequence[float]) -> bytes:
        sample_rate = 44_100
        duration = 2.0
        frames = int(sample_rate * duration)
        frequencies = [
            110.0 + 330.0 * abs(latent[index % len(latent)]) if latent else 220.0
            for index in range(3)
        ]
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            chunks = bytearray()
            for frame in range(frames):
                t = frame / sample_rate
                envelope = min(1.0, frame / 2000.0) * min(1.0, (frames - frame) / 3000.0)
                value = sum(math.sin(2.0 * math.pi * freq * t) for freq in frequencies) / 3.0
                sample = int(max(-1.0, min(1.0, value * 0.42 * envelope)) * 32767)
                chunks += struct.pack("<h", sample)
            wav.writeframes(bytes(chunks))
        return buffer.getvalue()

    @staticmethod
    def _video(latent: Sequence[float], cycle: int) -> bytes:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("video generation requires ffmpeg")
        width, height, fps, frame_count = 320, 180, 12, 36
        frames = bytearray()
        energy = sum(abs(value) for value in latent) / max(1, len(latent))
        for frame in range(frame_count):
            phase = frame / frame_count * math.tau
            for y in range(height):
                ny = (y - height / 2) / height
                for x in range(width):
                    nx = (x - width / 2) / width
                    radius = math.sqrt(nx * nx + ny * ny)
                    wave_value = math.sin(18 * radius - phase * 3 + cycle * 0.05)
                    glow = max(0.0, 1.0 - radius * 2.2) * (0.45 + energy)
                    r = int(max(0, min(255, 18 + 180 * glow + 45 * wave_value)))
                    g = int(max(0, min(255, 22 + 210 * glow - 25 * wave_value)))
                    b = int(max(0, min(255, 38 + 150 * glow + 55 * wave_value)))
                    frames.extend((r, g, b))
        with tempfile.TemporaryDirectory(prefix="jarvisx-mmvm-") as tmp:
            output = Path(tmp) / "artifact.mp4"
            command = [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{width}x{height}",
                "-r",
                str(fps),
                "-i",
                "pipe:0",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(output),
            ]
            result = subprocess.run(
                command,
                input=bytes(frames),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=30,
            )
            if result.returncode != 0 or not output.exists():
                detail = result.stderr.decode("utf-8", errors="replace")[-1200:]
                raise RuntimeError(f"ffmpeg generation failed: {detail}")
            return output.read_bytes()

    @staticmethod
    def _voxel_json(prompt: str, latent: Sequence[float], cycle: int) -> bytes:
        voxels = []
        for index in range(min(96, len(latent))):
            value = float(latent[index])
            voxels.append(
                {
                    "x": index % 8,
                    "y": (index // 8) % 4,
                    "z": index // 32,
                    "value": round(value, 7),
                    "energy": round(abs(value), 7),
                }
            )
        return json.dumps(
            {"cycle": cycle, "prompt": prompt, "voxels": voxels},
            separators=(",", ":"),
        ).encode("utf-8")


class MMVMKernel:
    """Transactional task scheduler and state machine for the MMVM runtime."""

    def __init__(
        self,
        db_path: str | os.PathLike[str],
        config: MMVMConfig | None = None,
    ) -> None:
        self.config = config or MMVMConfig()
        self.codec = MMVMCodec(self.config)
        self.policy = LambdaPolicy(self.config)
        self.memory = SparseVoxelMemory(db_path, self.config)
        self.generator = MultimediaGenerator()
        self._queue: deque[str] = deque()
        self._tasks: dict[str, MMVMTask] = {}
        self._lock = threading.RLock()
        self._cycle = 0
        self._last_metrics: CycleMetrics | None = None

    def submit(
        self,
        payload: bytes,
        *,
        modality: str = "binary",
        target: str | None = None,
    ) -> MMVMTask:
        if not isinstance(payload, (bytes, bytearray, memoryview)):
            raise TypeError("payload must be bytes-like")
        payload = bytes(payload)
        if len(payload) > self.config.max_payload_bytes:
            raise ValueError("payload exceeds configured byte budget")
        if target is not None and target.lower() not in MultimediaGenerator.SUPPORTED:
            raise ValueError(f"unsupported target: {target}")
        with self._lock:
            if len(self._queue) >= self.config.max_queue:
                raise RuntimeError("task queue is full")
            task = MMVMTask(
                task_id=uuid.uuid4().hex,
                modality=modality.strip().lower() or "binary",
                target=target.lower() if target else None,
                payload=payload,
                created_at=time.time(),
            )
            self._tasks[task.task_id] = task
            self._queue.append(task.task_id)
            self.memory.record_event(self._cycle, task.task_id, "queued", task.public())
            return task

    def run_next(self) -> CycleMetrics | None:
        with self._lock:
            if not self._queue:
                return None
            task_id = self._queue.popleft()
            task = self._tasks[task_id]
            task.state = TaskState.RUNNING
            self._cycle += 1
            cycle = self._cycle
        started = time.perf_counter()
        packet: CodecPacket | None = None
        refined: tuple[float, ...] = ()
        xi_dot = 0.0
        reconstruction_error = 1.0
        address: MemoryAddress | None = None
        artifact_id: str | None = None
        try:
            omega = self.memory.omega_snapshot(self.config.omega_window)
            packet = self.codec.encode(task.payload)
            refined, xi_dot = self.codec.refine(packet.latent, omega=omega.activation, cycle=cycle)
            decoded = self.codec.decode(packet)
            reconstruction_error = self._reconstruction_error(task.payload, decoded)
            decision = self.policy.validate(packet, refined, decoded)
            if not decision.accepted:
                task.state = TaskState.REJECTED
                task.error = decision.reason
                task.completed_at = time.time()
                elapsed = (time.perf_counter() - started) * 1000.0
                metrics = self._metrics(
                    cycle,
                    task,
                    packet,
                    refined,
                    xi_dot,
                    reconstruction_error,
                    False,
                    None,
                    None,
                    elapsed,
                )
                self.memory.record_event(cycle, task.task_id, "reject", metrics.public())
                self._last_metrics = metrics
                return metrics

            object_id, address = self.memory.commit_object(
                modality=task.modality,
                packet=packet,
                latent=refined,
                metadata={
                    "cycle": cycle,
                    "task_id": task.task_id,
                    "target": task.target,
                    "reconstruction_error": reconstruction_error,
                    "xi_dot": xi_dot,
                },
            )
            task.object_id = object_id
            if task.target:
                artifact = self.generator.generate(task.target, decoded, refined, cycle)
                self.memory.store_artifact(task.task_id, artifact)
                task.artifact_id = artifact.artifact_id
                artifact_id = artifact.artifact_id
            task.state = TaskState.COMMITTED
            task.completed_at = time.time()
            elapsed = (time.perf_counter() - started) * 1000.0
            metrics = self._metrics(
                cycle,
                task,
                packet,
                refined,
                xi_dot,
                reconstruction_error,
                True,
                address,
                artifact_id,
                elapsed,
            )
            self.memory.record_event(cycle, task.task_id, "commit", metrics.public())
            self._last_metrics = metrics
            return metrics
        except Exception as exc:
            task.state = TaskState.FAILED
            task.error = f"{type(exc).__name__}: {exc}"
            task.completed_at = time.time()
            elapsed = (time.perf_counter() - started) * 1000.0
            if packet is None:
                packet = CodecPacket("zlib", hashlib.sha256(task.payload).hexdigest(), len(task.payload), b"", ())
            metrics = self._metrics(
                cycle,
                task,
                packet,
                refined,
                xi_dot,
                reconstruction_error,
                False,
                address,
                artifact_id,
                elapsed,
            )
            self.memory.record_event(
                cycle,
                task.task_id,
                "failure",
                {**metrics.public(), "error": task.error},
            )
            self._last_metrics = metrics
            return metrics

    def run_until_idle(self, limit: int | None = None) -> list[CycleMetrics]:
        results: list[CycleMetrics] = []
        while limit is None or len(results) < limit:
            metrics = self.run_next()
            if metrics is None:
                break
            results.append(metrics)
        return results

    def task(self, task_id: str) -> dict[str, Any] | None:
        with self._lock:
            task = self._tasks.get(task_id)
            return task.public() if task else None

    def tasks(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            ordered = sorted(self._tasks.values(), key=lambda task: task.created_at, reverse=True)
            return [task.public() for task in ordered[: max(1, limit)]]

    def status(self) -> dict[str, Any]:
        with self._lock:
            queue_depth = len(self._queue)
            task_counts: dict[str, int] = {state.value: 0 for state in TaskState}
            for task in self._tasks.values():
                task_counts[task.state.value] += 1
            last = self._last_metrics.public() if self._last_metrics else None
            cycle = self._cycle
        omega = asdict(self.memory.omega_snapshot(self.config.omega_window))
        return {
            "system": "Jarvis-X MMVM",
            "mode": "operational",
            "cycle": cycle,
            "queue_depth": queue_depth,
            "task_counts": task_counts,
            "memory": self.memory.stats(),
            "omega": omega,
            "lambda": {"policy": "checksum+finite+bounds+resource", "active": True},
            "last_metrics": last,
            "supported_generation": sorted(MultimediaGenerator.SUPPORTED),
        }

    @staticmethod
    def _reconstruction_error(source: bytes, decoded: bytes) -> float:
        if source == decoded:
            return 0.0
        denominator = max(1, len(source), len(decoded))
        mismatches = abs(len(source) - len(decoded))
        mismatches += sum(a != b for a, b in zip(source, decoded))
        return mismatches / denominator

    @staticmethod
    def _metrics(
        cycle: int,
        task: MMVMTask,
        packet: CodecPacket,
        latent: Sequence[float],
        xi_dot: float,
        reconstruction_error: float,
        accepted: bool,
        address: MemoryAddress | None,
        artifact_id: str | None,
        elapsed_ms: float,
    ) -> CycleMetrics:
        xi_norm = math.sqrt(sum(float(value) ** 2 for value in latent) / max(1, len(latent)))
        ratio = len(packet.encoded) / max(1, packet.source_size)
        return CycleMetrics(
            cycle=cycle,
            task_id=task.task_id,
            state=task.state.value,
            source_bytes=packet.source_size,
            encoded_bytes=len(packet.encoded),
            compression_ratio=ratio,
            xi_norm=xi_norm,
            xi_dot=xi_dot,
            reconstruction_error=reconstruction_error,
            lambda_accepted=accepted,
            memory_address=address,
            artifact_id=artifact_id,
            elapsed_ms=elapsed_ms,
        )


def encode_base64(payload: bytes) -> str:
    return base64.b64encode(payload).decode("ascii")


def decode_base64(payload: str) -> bytes:
    return base64.b64decode(payload.encode("ascii"), validate=True)
