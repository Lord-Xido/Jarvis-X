"""Executable reference operator for the Dr. Moagi Psi engine.

The module implements the compositional field equation

    Psi_Engine(X, Y, Z, t) =
        D_theta(Omega_fusion(sum_k M_octree(x, y, z) E_phi(V^k_(x,y,z,t))))

as a deterministic, dependency-free Python reference.  The learned encoder,
fusion operator and decoder remain injectable callables; this module fixes the
runtime semantics, validation and octree gating needed to execute and test the
equation.

The current ``OctreeSpatialMask`` realizes ``M_octree`` as an isotropic
projection ``m(x, y, z) I`` where ``m`` is 1 inside an active octree region and
0 outside it.  A future implementation can replace the scalar gate with a
full learned spatial matrix without changing the outer engine contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from .fractal_octree import FractalOctreeNode

Vector = tuple[float, ...]


class VectorOperator(Protocol):
    """Map one finite-dimensional vector to another."""

    def __call__(self, values: Vector) -> Sequence[float]:
        """Transform ``values`` and return the resulting vector."""


class SpatialMask(Protocol):
    """Return the scalar octree gate at one spatial coordinate."""

    def __call__(self, x: float, y: float, z: float) -> float:
        """Return a finite scalar spatial weight."""


class BlockField(Protocol):
    """Supply block ``V^(k)_(x,y,z,t)`` at a spacetime coordinate."""

    def __call__(
        self,
        x: float,
        y: float,
        z: float,
        t: float,
        block_index: int,
    ) -> Sequence[float]:
        """Return one block vector for ``block_index``."""


@dataclass(frozen=True)
class EngineTrace:
    """Observable intermediate state for one Psi-engine evaluation."""

    position: tuple[float, float, float]
    time: float
    octree_mask: float
    encoded_blocks: tuple[Vector, ...]
    latent_sum: Vector
    fused_latent: Vector
    output: Vector

    @property
    def block_count(self) -> int:
        """Number of encoded blocks contributing to this evaluation."""

        return len(self.encoded_blocks)


@dataclass(frozen=True)
class OctreeSpatialMask:
    """Binary spatial projector derived from ``FractalOctreeNode``.

    The root and its descendants are interpreted as axis-aligned cubes.  A
    coordinate receives weight 1 exactly when it lies inside the root and its
    traversal ends on an active node.  Coordinates entering any culled octant
    receive weight 0.
    """

    root: FractalOctreeNode

    def __call__(self, x: float, y: float, z: float) -> float:
        node = self.root
        if not _contains(node, x, y, z) or not node.is_active:
            return 0.0

        while node.children:
            half = node.size / 2.0
            dx = 1 if x >= node.x + half else 0
            dy = 1 if y >= node.y + half else 0
            dz = 1 if z >= node.z + half else 0
            child_index = dx + (2 * dy) + (4 * dz)
            child = node.children[child_index]
            if not child.is_active:
                return 0.0
            node = child

        return 1.0


@dataclass(frozen=True)
class PsiEngine:
    """Composable runtime for the Psi-engine equation."""

    encoder: VectorOperator
    fusion: VectorOperator
    decoder: VectorOperator
    octree_mask: SpatialMask

    def evaluate_blocks(
        self,
        x: float,
        y: float,
        z: float,
        t: float,
        blocks: Sequence[Sequence[float]],
    ) -> EngineTrace:
        """Evaluate already-materialized block vectors at ``(x, y, z, t)``."""

        if not blocks:
            raise ValueError("at least one block is required")

        mask = float(self.octree_mask(x, y, z))
        encoded_blocks: list[Vector] = []
        latent_sum: list[float] | None = None

        for raw_block in blocks:
            block = _vector(raw_block, "block")
            encoded = _vector(self.encoder(block), "encoder output")
            if not encoded:
                raise ValueError("encoder output must not be empty")

            if latent_sum is None:
                latent_sum = [0.0] * len(encoded)
            elif len(encoded) != len(latent_sum):
                raise ValueError("all encoder outputs must have the same dimension")

            for index, value in enumerate(encoded):
                latent_sum[index] += mask * value
            encoded_blocks.append(encoded)

        assert latent_sum is not None
        latent = tuple(latent_sum)
        fused = _vector(self.fusion(latent), "fusion output")
        if not fused:
            raise ValueError("fusion output must not be empty")
        output = _vector(self.decoder(fused), "decoder output")
        if not output:
            raise ValueError("decoder output must not be empty")

        return EngineTrace(
            position=(float(x), float(y), float(z)),
            time=float(t),
            octree_mask=mask,
            encoded_blocks=tuple(encoded_blocks),
            latent_sum=latent,
            fused_latent=fused,
            output=output,
        )

    def evaluate_field(
        self,
        x: float,
        y: float,
        z: float,
        t: float,
        n_blocks: int,
        field: BlockField,
    ) -> EngineTrace:
        """Sample ``V^(k)_(x,y,z,t)`` then execute the full engine equation."""

        if n_blocks <= 0:
            raise ValueError("n_blocks must be positive")
        blocks = tuple(field(x, y, z, t, index) for index in range(n_blocks))
        return self.evaluate_blocks(x, y, z, t, blocks)


def identity_operator(values: Vector) -> Vector:
    """Return an immutable copy of ``values`` for reference configurations."""

    return tuple(values)


def _contains(node: FractalOctreeNode, x: float, y: float, z: float) -> bool:
    upper_x = node.x + node.size
    upper_y = node.y + node.size
    upper_z = node.z + node.size
    return bool(
        node.x <= x <= upper_x
        and node.y <= y <= upper_y
        and node.z <= z <= upper_z
    )


def _vector(values: Sequence[float], label: str) -> Vector:
    try:
        vector = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite numeric sequence") from exc

    for value in vector:
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError(f"{label} must contain only finite values")
    return vector
