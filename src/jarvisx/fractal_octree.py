"""Sparse recursive octree substrate for the Jarvis-X fractal hierarchy.

This module implements the exact four-survivor rule discussed for the
inward-folding octree scaffold.  Each active cube is divided into eight
spatial octants; only octants with ``dx + dy + dz < 2`` remain active.

The resulting object is an octree because each parent owns eight child
locations.  It is not a classical Menger sponge, which uses a 3 x 3 x 3
subdivision.  At recursion depth ``D`` this rule has:

    active leaves   = 4 ** D
    active nodes    = (4 ** (D + 1) - 1) / 3
    retained volume = initial_volume * 2 ** (-D)
    fractal dimension = log(4) / log(2) = 2
"""

from dataclasses import dataclass, field
from typing import Iterator, List, Tuple


Octant = Tuple[int, int, int]
MetricsTuple = Tuple[int, float]

OCTANTS: Tuple[Octant, ...] = (
    (0, 0, 0),
    (1, 0, 0),
    (0, 1, 0),
    (1, 1, 0),
    (0, 0, 1),
    (1, 0, 1),
    (0, 1, 1),
    (1, 1, 1),
)


@dataclass(frozen=True)
class FractalOctreeMetrics:
    """Deterministic metrics for one materialized octree."""

    active_nodes: int
    active_leaves: int
    retained_volume: float
    max_depth: int

    @property
    def fractal_dimension(self) -> float:
        """Similarity dimension of the four-survivor, half-scale rule."""

        return 2.0


@dataclass
class FractalOctreeNode:
    """One spatial region in the sparse Jarvis-X recursive hierarchy.

    Coordinates identify the minimum corner of an axis-aligned cube.  All
    eight child locations are retained as structural metadata, while only
    four children are active under the default inward-folding rule.
    """

    x: float
    y: float
    z: float
    size: float
    depth: int = 0
    max_depth: int = 3
    is_active: bool = True
    children: List["FractalOctreeNode"] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.size <= 0:
            raise ValueError("size must be positive")
        if self.depth < 0:
            raise ValueError("depth must be non-negative")
        if self.max_depth < self.depth:
            raise ValueError("max_depth must be greater than or equal to depth")

    @staticmethod
    def survives(dx: int, dy: int, dz: int) -> bool:
        """Return whether an octant survives the four-branch cull rule."""

        if dx not in (0, 1) or dy not in (0, 1) or dz not in (0, 1):
            raise ValueError("octant coordinates must each be 0 or 1")
        return (dx + dy + dz) < 2

    def subdivide_and_optimize(self) -> None:
        """Materialize the bounded sparse hierarchy deterministically.

        Re-running this method rebuilds the same subtree instead of appending
        duplicate children.  Inactive nodes and terminal nodes do not split.
        """

        self.children.clear()
        if not self.is_active or self.depth >= self.max_depth:
            return

        half = self.size / 2.0
        next_depth = self.depth + 1

        for dx, dy, dz in OCTANTS:
            active = self.survives(dx, dy, dz)
            child = FractalOctreeNode(
                x=self.x + dx * half,
                y=self.y + dy * half,
                z=self.z + dz * half,
                size=half,
                depth=next_depth,
                max_depth=self.max_depth,
                is_active=active,
            )
            if active:
                child.subdivide_and_optimize()
            self.children.append(child)

    def iter_active(self) -> Iterator["FractalOctreeNode"]:
        """Yield active nodes in deterministic depth-first octant order."""

        if not self.is_active:
            return
        yield self
        for child in self.children:
            yield from child.iter_active()

    def metrics(self) -> FractalOctreeMetrics:
        """Calculate active-node, active-leaf, and retained-volume metrics."""

        if not self.is_active:
            return FractalOctreeMetrics(0, 0, 0.0, self.max_depth)

        if not self.children:
            return FractalOctreeMetrics(1, 1, self.size ** 3, self.max_depth)

        active_nodes = 1
        active_leaves = 0
        retained_volume = 0.0

        for child in self.children:
            child_metrics = child.metrics()
            active_nodes += child_metrics.active_nodes
            active_leaves += child_metrics.active_leaves
            retained_volume += child_metrics.retained_volume

        return FractalOctreeMetrics(
            active_nodes=active_nodes,
            active_leaves=active_leaves,
            retained_volume=retained_volume,
            max_depth=self.max_depth,
        )

    def calculate_metrics(self) -> MetricsTuple:
        """Compatibility wrapper returning ``(active_nodes, volume)``."""

        result = self.metrics()
        return result.active_nodes, result.retained_volume

    def expected_metrics(self) -> FractalOctreeMetrics:
        """Return the closed-form metrics for this root and depth interval."""

        if not self.is_active:
            return FractalOctreeMetrics(0, 0, 0.0, self.max_depth)

        levels = self.max_depth - self.depth
        active_leaves = 4 ** levels
        active_nodes = (4 ** (levels + 1) - 1) // 3
        retained_volume = (self.size ** 3) * (0.5 ** levels)
        return FractalOctreeMetrics(
            active_nodes=active_nodes,
            active_leaves=active_leaves,
            retained_volume=retained_volume,
            max_depth=self.max_depth,
        )


def build_fractal_octree(
    size: float = 1.0,
    max_depth: int = 3,
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> FractalOctreeNode:
    """Construct and materialize a complete bounded four-branch octree."""

    root = FractalOctreeNode(
        x=origin[0],
        y=origin[1],
        z=origin[2],
        size=size,
        max_depth=max_depth,
    )
    root.subdivide_and_optimize()
    return root


def main() -> None:
    """Run the canonical unit-cube demonstration."""

    root = build_fractal_octree(size=1.0, max_depth=3)
    metrics = root.metrics()

    print("System State: LOCKED IN")
    print("Active Nodes Traversed: {}".format(metrics.active_nodes))
    print(
        "Optimized System Volume: {:.4f} (Original: 1.0000)".format(
            metrics.retained_volume
        )
    )


if __name__ == "__main__":
    main()
