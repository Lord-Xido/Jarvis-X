"""Deterministic reference model for the 3D-DSMPE-Ω visualization.

The browser demo is a renderer. This module defines the bounded, testable computation
behind the displayed metrics: adaptive octree encoding of a displaced torus signed-
distance field, piecewise-constant decoding, reconstruction RMSE and a complexity-
regularized Dr. Moagi objective.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

Vec3 = tuple[float, float, float]


@dataclass(frozen=True)
class DsmpeConfig:
    """Configuration for one deterministic octree encoding experiment."""

    root_size: float = 3.6
    max_depth: int = 4
    gamma: float = 0.5
    field_time: float = 0.0
    sample_resolution: int = 9
    lipschitz_bound: float = 1.4
    collapse_margin: float = 0.02

    def __post_init__(self) -> None:
        if self.root_size <= 0.0:
            raise ValueError("root_size must be positive")
        if self.max_depth < 1:
            raise ValueError("max_depth must be at least 1")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be between 0 and 1")
        if self.sample_resolution < 2:
            raise ValueError("sample_resolution must be at least 2")
        if self.lipschitz_bound <= 0.0:
            raise ValueError("lipschitz_bound must be positive")
        if self.collapse_margin < 0.0:
            raise ValueError("collapse_margin must be non-negative")


@dataclass(frozen=True)
class OctreeNode:
    """One encoded octree node."""

    center: Vec3
    size: float
    depth: int
    path: int
    sdf_value: float
    boundary: bool
    children: tuple["OctreeNode", ...] = ()

    @property
    def is_leaf(self) -> bool:
        return not self.children


@dataclass(frozen=True)
class DsmpeMetrics:
    """Observed metrics for one encoded model."""

    depth: int
    visited_nodes: int
    total_possible_nodes: int
    leaf_count: int
    boundary_leaves: int
    compression_ratio: float
    complexity_ratio: float
    reconstruction_rmse: float
    depth_entropy: float
    partition_volume_error: float
    loss: float
    digest_sha256: str


@dataclass(frozen=True)
class EncodedModel:
    """Encoded tree plus its measured evidence."""

    config: DsmpeConfig
    root: OctreeNode
    leaves: tuple[OctreeNode, ...]
    metrics: DsmpeMetrics

    def decode(self, point: Vec3) -> float:
        """Decode a point using its containing leaf's piecewise-constant SDF value."""

        x, y, z = point
        half = self.config.root_size / 2.0
        if not (-half <= x <= half and -half <= y <= half and -half <= z <= half):
            raise ValueError("point lies outside the encoded root cube")

        node = self.root
        while node.children:
            cx, cy, cz = node.center
            child_index = (1 if x >= cx else 0) | (2 if y >= cy else 0) | (4 if z >= cz else 0)
            node = node.children[child_index]
        return node.sdf_value


@dataclass(frozen=True)
class ModelSelection:
    """Depth candidates and the model minimizing the observed objective."""

    candidates: tuple[EncodedModel, ...]
    selected: EncodedModel


def target_sdf(x: float, y: float, z: float, field_time: float = 0.0) -> float:
    """Displaced torus signed-distance field used by the reference and browser demo."""

    radial = math.hypot(x, y) - 1.1
    torus = math.hypot(radial, z) - 0.38
    displacement = (
        0.05
        * math.sin(4.0 * x + field_time)
        * math.cos(4.0 * y - 0.7 * field_time)
        * math.sin(4.0 * z + 0.5 * field_time)
    )
    return torus + displacement


def complete_octree_nodes(depth: int) -> int:
    """Return the node count of a complete eight-way tree through ``depth``."""

    if depth < 0:
        raise ValueError("depth must be non-negative")
    total = 0
    level_nodes = 1
    for _ in range(depth + 1):
        total += level_nodes
        level_nodes *= 8
    return total


def _build_node(
    center: Vec3,
    size: float,
    depth: int,
    path: int,
    config: DsmpeConfig,
) -> OctreeNode:
    x, y, z = center
    sdf_value = target_sdf(x, y, z, config.field_time)
    half_diagonal = math.sqrt(3.0) * size / 2.0
    safely_away_from_surface = (
        abs(sdf_value) > config.lipschitz_bound * half_diagonal + config.collapse_margin
    )

    if safely_away_from_surface:
        return OctreeNode(center, size, depth, path, sdf_value, False)
    if depth >= config.max_depth:
        return OctreeNode(center, size, depth, path, sdf_value, True)

    child_size = size / 2.0
    offset = size / 4.0
    children: list[OctreeNode] = []
    for child_index in range(8):
        child_center = (
            x + (offset if child_index & 1 else -offset),
            y + (offset if child_index & 2 else -offset),
            z + (offset if child_index & 4 else -offset),
        )
        children.append(
            _build_node(
                child_center,
                child_size,
                depth + 1,
                (path << 3) | child_index,
                config,
            )
        )
    return OctreeNode(center, size, depth, path, sdf_value, False, tuple(children))


def _walk(root: OctreeNode) -> tuple[tuple[OctreeNode, ...], int]:
    leaves: list[OctreeNode] = []
    visited = 0
    stack = [root]
    while stack:
        node = stack.pop()
        visited += 1
        if node.children:
            stack.extend(reversed(node.children))
        else:
            leaves.append(node)
    return tuple(leaves), visited


def _sample_points(size: float, resolution: int) -> tuple[Vec3, ...]:
    step = size / resolution
    first = -size / 2.0 + step / 2.0
    axis = tuple(first + index * step for index in range(resolution))
    return tuple((x, y, z) for z in axis for y in axis for x in axis)


def _depth_entropy(leaves: tuple[OctreeNode, ...], max_depth: int) -> float:
    counts = [0] * (max_depth + 1)
    for leaf in leaves:
        counts[leaf.depth] += 1
    total = len(leaves)
    if total == 0 or max_depth == 0:
        return 0.0
    entropy = 0.0
    for count in counts:
        if count:
            probability = count / total
            entropy -= probability * math.log2(probability)
    normalizer = math.log2(max_depth + 1)
    return entropy / normalizer if normalizer else 0.0


def _canonical_digest(root: OctreeNode, metrics_payload: dict[str, object]) -> str:
    leaves, _ = _walk(root)
    payload = {
        "metrics": metrics_payload,
        "leaves": [
            {
                "path": leaf.path,
                "depth": leaf.depth,
                "boundary": leaf.boundary,
                "sdf": round(leaf.sdf_value, 12),
            }
            for leaf in leaves
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def encode_model(config: DsmpeConfig) -> EncodedModel:
    """Encode, decode and measure one bounded 3D-DSMPE-Ω model."""

    root = _build_node((0.0, 0.0, 0.0), config.root_size, 0, 0, config)
    leaves, visited_nodes = _walk(root)
    total_possible_nodes = complete_octree_nodes(config.max_depth)
    complexity_ratio = visited_nodes / total_possible_nodes
    compression_ratio = 1.0 - complexity_ratio

    provisional = EncodedModel(
        config=config,
        root=root,
        leaves=leaves,
        metrics=DsmpeMetrics(
            depth=config.max_depth,
            visited_nodes=visited_nodes,
            total_possible_nodes=total_possible_nodes,
            leaf_count=len(leaves),
            boundary_leaves=sum(leaf.boundary for leaf in leaves),
            compression_ratio=compression_ratio,
            complexity_ratio=complexity_ratio,
            reconstruction_rmse=0.0,
            depth_entropy=0.0,
            partition_volume_error=0.0,
            loss=0.0,
            digest_sha256="",
        ),
    )

    squared_error = 0.0
    points = _sample_points(config.root_size, config.sample_resolution)
    for point in points:
        original = target_sdf(*point, config.field_time)
        reconstructed = provisional.decode(point)
        squared_error += (original - reconstructed) ** 2
    reconstruction_rmse = math.sqrt(squared_error / len(points))

    represented_volume = sum(leaf.size**3 for leaf in leaves)
    partition_volume_error = abs(represented_volume - config.root_size**3)
    depth_entropy = _depth_entropy(leaves, config.max_depth)
    loss = reconstruction_rmse + config.gamma * complexity_ratio

    metrics_payload: dict[str, object] = {
        "depth": config.max_depth,
        "visited_nodes": visited_nodes,
        "total_possible_nodes": total_possible_nodes,
        "leaf_count": len(leaves),
        "boundary_leaves": sum(leaf.boundary for leaf in leaves),
        "compression_ratio": round(compression_ratio, 12),
        "complexity_ratio": round(complexity_ratio, 12),
        "reconstruction_rmse": round(reconstruction_rmse, 12),
        "depth_entropy": round(depth_entropy, 12),
        "partition_volume_error": round(partition_volume_error, 12),
        "loss": round(loss, 12),
    }
    digest = _canonical_digest(root, metrics_payload)
    metrics = DsmpeMetrics(
        depth=config.max_depth,
        visited_nodes=visited_nodes,
        total_possible_nodes=total_possible_nodes,
        leaf_count=len(leaves),
        boundary_leaves=sum(leaf.boundary for leaf in leaves),
        compression_ratio=compression_ratio,
        complexity_ratio=complexity_ratio,
        reconstruction_rmse=reconstruction_rmse,
        depth_entropy=depth_entropy,
        partition_volume_error=partition_volume_error,
        loss=loss,
        digest_sha256=digest,
    )
    return EncodedModel(config, root, leaves, metrics)


def select_model(config: DsmpeConfig) -> ModelSelection:
    """Evaluate depths 1..``max_depth`` and select the minimum observed loss."""

    candidates = tuple(
        encode_model(
            DsmpeConfig(
                root_size=config.root_size,
                max_depth=depth,
                gamma=config.gamma,
                field_time=config.field_time,
                sample_resolution=config.sample_resolution,
                lipschitz_bound=config.lipschitz_bound,
                collapse_margin=config.collapse_margin,
            )
        )
        for depth in range(1, config.max_depth + 1)
    )
    selected = min(candidates, key=lambda candidate: (candidate.metrics.loss, candidate.metrics.depth))
    return ModelSelection(candidates, selected)
