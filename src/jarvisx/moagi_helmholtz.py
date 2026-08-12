"""Transactional orchestration contract for the Moagi-Helmholtz generative cycle.

The module composes multimodal conditioning, geometry encoding/decoding,
refinement, rendering, archival coding, inverse inference, verification and
commit without making any backend-specific performance or invertibility claim.

Production implementations may bind these protocols to neural, C++/CUDA,
renderer, or video-codec backends. The included reference components are
small deterministic conformance fixtures only.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, Sequence

Vertex = tuple[float, float, float]
Face = tuple[int, int, int]


@dataclass(frozen=True)
class Mesh:
    """Finite triangle mesh used at the orchestration boundary."""

    vertices: tuple[Vertex, ...]
    faces: tuple[Face, ...]

    def __post_init__(self) -> None:
        for vertex in self.vertices:
            if len(vertex) != 3 or not all(math.isfinite(float(v)) for v in vertex):
                raise ValueError("mesh vertices must contain three finite coordinates")
        count = len(self.vertices)
        for face in self.faces:
            if len(face) != 3:
                raise ValueError("faces must be triangles")
            if any(isinstance(i, bool) or not isinstance(i, int) for i in face):
                raise TypeError("face indices must be integers")
            if any(i < 0 or i >= count for i in face):
                raise ValueError("face index is outside the vertex buffer")
            if len(set(face)) != 3:
                raise ValueError("degenerate index triangle is not admissible")


class Conditioner(Protocol):
    def condition(self, multimodal: Mapping[str, Any]) -> Any:
        ...


class GeometryEncoder(Protocol):
    def encode(self, mesh: Mesh) -> Any:
        ...


class GeometryDecoder(Protocol):
    def decode(self, latent: Any, condition: Any) -> Mesh:
        ...


class GeometryRefiner(Protocol):
    def refine(self, mesh: Mesh) -> Mesh:
        ...


class Renderer(Protocol):
    def render(self, mesh: Mesh) -> Sequence[Any]:
        ...


class ArchiveCodec(Protocol):
    def encode(self, frames: Sequence[Any], side_info: Mapping[str, Any]) -> bytes:
        ...

    def decode(self, archive: bytes) -> tuple[Sequence[Any], Mapping[str, Any]]:
        ...


class InverseModel(Protocol):
    def infer(
        self, frames: Sequence[Any], side_info: Mapping[str, Any]
    ) -> tuple[Any, Any]:
        ...


@dataclass(frozen=True)
class MoagiHelmholtzMetrics:
    cycle: int
    source_vertices: int
    archive_bytes: int
    generation_rms: float
    cycle_rms: float
    anchor_rms: float


@dataclass(frozen=True)
class MoagiHelmholtzState:
    cycle: int
    condition: Any
    latent: Any
    generated_mesh: Mesh
    refined_mesh: Mesh
    archive: bytes
    reconstructed_mesh: Mesh
    metrics: MoagiHelmholtzMetrics


StateValidator = Callable[[MoagiHelmholtzState], bool]


@dataclass(frozen=True)
class MoagiHelmholtzConfig:
    max_vertices: int = 100_000
    max_archive_bytes: int = 64 * 1024 * 1024
    max_cycle_rms: float = math.inf

    def __post_init__(self) -> None:
        if isinstance(self.max_vertices, bool) or not isinstance(self.max_vertices, int):
            raise TypeError("max_vertices must be an integer")
        if self.max_vertices <= 0:
            raise ValueError("max_vertices must be positive")
        if isinstance(self.max_archive_bytes, bool) or not isinstance(self.max_archive_bytes, int):
            raise TypeError("max_archive_bytes must be an integer")
        if self.max_archive_bytes <= 0:
            raise ValueError("max_archive_bytes must be positive")
        if not isinstance(self.max_cycle_rms, (int, float)) or isinstance(
            self.max_cycle_rms, bool
        ):
            raise TypeError("max_cycle_rms must be numeric")
        if math.isnan(float(self.max_cycle_rms)) or self.max_cycle_rms < 0.0:
            raise ValueError("max_cycle_rms must be non-negative")


class MoagiHelmholtzEngine:
    """Candidate-first transactional implementation of the unified cycle."""

    def __init__(
        self,
        *,
        conditioner: Conditioner,
        encoder: GeometryEncoder,
        decoder: GeometryDecoder,
        refiner: GeometryRefiner,
        renderer: Renderer,
        archive_codec: ArchiveCodec,
        inverse_model: InverseModel,
        config: MoagiHelmholtzConfig | None = None,
    ) -> None:
        self.conditioner = conditioner
        self.encoder = encoder
        self.decoder = decoder
        self.refiner = refiner
        self.renderer = renderer
        self.archive_codec = archive_codec
        self.inverse_model = inverse_model
        self.config = config or MoagiHelmholtzConfig()
        self._state: MoagiHelmholtzState | None = None
        self._anchor: Mesh | None = None

    @property
    def state(self) -> MoagiHelmholtzState | None:
        return self._state

    @property
    def anchor(self) -> Mesh | None:
        return self._anchor

    def step(
        self,
        multimodal: Mapping[str, Any],
        source_mesh: Mesh,
        *,
        validator: StateValidator | None = None,
    ) -> MoagiHelmholtzState:
        """Execute one full cycle and atomically publish only an admissible candidate."""

        self._check_mesh_budget(source_mesh)
        anchor = self._anchor or source_mesh

        condition = self.conditioner.condition(multimodal)
        latent = self.encoder.encode(source_mesh)
        generated = self.decoder.decode(latent, condition)
        self._check_mesh_budget(generated)

        refined = self.refiner.refine(generated)
        self._check_mesh_budget(refined)

        frames = tuple(self.renderer.render(refined))
        side_info = {"latent": latent, "condition": condition}
        archive = bytes(self.archive_codec.encode(frames, side_info))
        if len(archive) > self.config.max_archive_bytes:
            raise RuntimeError("archive exceeds configured byte budget")

        decoded_frames, decoded_side_info = self.archive_codec.decode(archive)
        inferred_latent, inferred_condition = self.inverse_model.infer(
            decoded_frames, decoded_side_info
        )
        reconstructed = self.decoder.decode(inferred_latent, inferred_condition)
        reconstructed = self.refiner.refine(reconstructed)
        self._check_mesh_budget(reconstructed)

        next_cycle = 1 if self._state is None else self._state.cycle + 1
        generation_rms = mesh_rms(source_mesh, generated)
        cycle_rms = mesh_rms(refined, reconstructed)
        anchor_rms = mesh_rms(anchor, refined)

        metrics = MoagiHelmholtzMetrics(
            cycle=next_cycle,
            source_vertices=len(source_mesh.vertices),
            archive_bytes=len(archive),
            generation_rms=generation_rms,
            cycle_rms=cycle_rms,
            anchor_rms=anchor_rms,
        )
        candidate = MoagiHelmholtzState(
            cycle=next_cycle,
            condition=condition,
            latent=latent,
            generated_mesh=generated,
            refined_mesh=refined,
            archive=archive,
            reconstructed_mesh=reconstructed,
            metrics=metrics,
        )

        if cycle_rms > self.config.max_cycle_rms:
            raise RuntimeError("cycle reconstruction exceeds configured tolerance")
        if validator is not None and not bool(validator(candidate)):
            raise RuntimeError("candidate rejected by Moagi-Helmholtz validator")

        # Atomic publication boundary: no authoritative state changes occur before here.
        if self._anchor is None:
            self._anchor = source_mesh
        self._state = candidate
        return candidate

    def _check_mesh_budget(self, mesh: Mesh) -> None:
        if len(mesh.vertices) > self.config.max_vertices:
            raise RuntimeError("mesh exceeds configured vertex budget")


def mesh_rms(left: Mesh, right: Mesh) -> float:
    """Topology-preserving RMS vertex displacement used for reference telemetry."""

    if left.faces != right.faces or len(left.vertices) != len(right.vertices):
        return math.inf
    if not left.vertices:
        return 0.0
    squared = 0.0
    for a, b in zip(left.vertices, right.vertices):
        squared += sum((float(x) - float(y)) ** 2 for x, y in zip(a, b))
    return math.sqrt(squared / (3 * len(left.vertices)))


class ReferenceConditioner:
    """Deterministic JSON-native conditioning fixture."""

    def condition(self, multimodal: Mapping[str, Any]) -> Any:
        return {key: multimodal[key] for key in sorted(multimodal)}


class ReferenceGeometryCodec:
    """Lossless mesh descriptor fixture; it is not a compression model."""

    def encode(self, mesh: Mesh) -> Any:
        return {
            "vertices": [list(vertex) for vertex in mesh.vertices],
            "faces": [list(face) for face in mesh.faces],
        }

    def decode(self, latent: Any, condition: Any) -> Mesh:
        del condition
        return Mesh(
            vertices=tuple(
                (float(vertex[0]), float(vertex[1]), float(vertex[2]))
                for vertex in latent["vertices"]
            ),
            faces=tuple(
                (int(face[0]), int(face[1]), int(face[2])) for face in latent["faces"]
            ),
        )


class IdentityGeometryRefiner:
    def refine(self, mesh: Mesh) -> Mesh:
        return mesh


class ReferenceRenderer:
    """Serializes geometry into one logical frame for conformance testing."""

    def render(self, mesh: Mesh) -> Sequence[Any]:
        return (
            {
                "vertices": [list(vertex) for vertex in mesh.vertices],
                "faces": [list(face) for face in mesh.faces],
            },
        )


class ReferenceArchiveCodec:
    """Deterministic JSON archive fixture; not an MP4/video implementation."""

    def encode(self, frames: Sequence[Any], side_info: Mapping[str, Any]) -> bytes:
        payload = {"frames": list(frames), "side_info": dict(side_info)}
        return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")

    def decode(self, archive: bytes) -> tuple[Sequence[Any], Mapping[str, Any]]:
        payload = json.loads(archive.decode("utf-8"))
        return tuple(payload["frames"]), payload["side_info"]


class SideInfoInverseModel:
    """Exact conformance inverse using explicitly archived latent/condition side information."""

    def infer(
        self, frames: Sequence[Any], side_info: Mapping[str, Any]
    ) -> tuple[Any, Any]:
        del frames
        return side_info["latent"], side_info["condition"]
