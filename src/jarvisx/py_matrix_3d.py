"""Deterministic reference mathematics for the PY-MATRIX 3D 1M-LOC surface.

The module maps one million logical source lines onto 250 procedural 3D clusters.
It does not load or execute a million-line program and has no external side effects.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

TOTAL_LOC = 1_000_000
CLUSTER_COUNT = 250
LOC_PER_CLUSTER = TOTAL_LOC // CLUSTER_COUNT
GRID_X = 20
GRID_Y = 20
GRID_Z = 10
PULSE_HZ = 1.0
PULSE_PERIOD_S = 1.0 / PULSE_HZ
COHERENCE_TARGET = 0.9998
TARGET_FPS = 60.0
FRAME_BUDGET_MS = 1000.0 / TARGET_FPS
GOLDEN_ANGLE = math.pi * (3.0 - math.sqrt(5.0))

if GRID_X * GRID_Y * GRID_Z != LOC_PER_CLUSTER:
    raise RuntimeError("local grid must exactly cover one cluster")


@dataclass(frozen=True)
class SpatialAddress:
    """O(1) procedural address for one logical source line."""

    line: int
    cluster: int
    local_index: int
    cell_x: int
    cell_y: int
    cell_z: int
    radius: float
    theta: float
    world_x: float
    world_y: float
    world_z: float


@dataclass(frozen=True)
class ClusterPose:
    """Procedural cluster center used by the browser instancing surface."""

    cluster: int
    radius: float
    theta: float
    x: float
    y: float
    z: float


def _line_number(line: int) -> int:
    if isinstance(line, bool) or not isinstance(line, int):
        raise TypeError("line must be an integer")
    if not 1 <= line <= TOTAL_LOC:
        raise ValueError(f"line must be within [1, {TOTAL_LOC}]")
    return line


def cluster_pose(cluster: int) -> ClusterPose:
    """Return the deterministic center of one spiral cluster."""

    if isinstance(cluster, bool) or not isinstance(cluster, int):
        raise TypeError("cluster must be an integer")
    if not 0 <= cluster < CLUSTER_COUNT:
        raise ValueError(f"cluster must be within [0, {CLUSTER_COUNT - 1}]")

    theta = cluster * GOLDEN_ANGLE
    radius = 7.0 + 0.035 * cluster
    z = (cluster - (CLUSTER_COUNT - 1) / 2.0) * 0.08
    return ClusterPose(
        cluster=cluster,
        radius=radius,
        theta=theta,
        x=radius * math.cos(theta),
        y=radius * math.sin(theta),
        z=z,
    )


def locate_line(line: int) -> SpatialAddress:
    """Map a 1-based LOC index to cluster, local cell and world position in O(1)."""

    line = _line_number(line)
    zero = line - 1
    cluster = zero // LOC_PER_CLUSTER
    local = zero % LOC_PER_CLUSTER
    cell_x = local % GRID_X
    cell_y = (local // GRID_X) % GRID_Y
    cell_z = local // (GRID_X * GRID_Y)

    pose = cluster_pose(cluster)
    # Small deterministic local offsets encode the 20 x 20 x 10 cluster lattice.
    ox = (cell_x - (GRID_X - 1) / 2.0) * 0.045
    oy = (cell_y - (GRID_Y - 1) / 2.0) * 0.045
    oz = (cell_z - (GRID_Z - 1) / 2.0) * 0.060

    # Rotate the local XY lattice with the parent cluster to preserve spiral locality.
    c = math.cos(pose.theta)
    s = math.sin(pose.theta)
    world_x = pose.x + ox * c - oy * s
    world_y = pose.y + ox * s + oy * c
    world_z = pose.z + oz

    return SpatialAddress(
        line=line,
        cluster=cluster,
        local_index=local,
        cell_x=cell_x,
        cell_y=cell_y,
        cell_z=cell_z,
        radius=pose.radius,
        theta=pose.theta,
        world_x=world_x,
        world_y=world_y,
        world_z=world_z,
    )


def pulse(time_s: float, *, phase_cycles: float = 0.0) -> float:
    """Return the normalized logical execution pulse at exactly 1 Hz."""

    time_s = float(time_s)
    phase_cycles = float(phase_cycles)
    if not math.isfinite(time_s) or not math.isfinite(phase_cycles):
        raise ValueError("pulse inputs must be finite")
    return 0.5 + 0.5 * math.cos(2.0 * math.pi * (PULSE_HZ * time_s - phase_cycles))


def travelling_pulse(time_s: float, cluster: int, *, wavelength_clusters: float = 32.0) -> float:
    """Propagate the 1-Hz wave across cluster index space."""

    cluster_pose(cluster)  # validates cluster without allocating cluster state
    wavelength_clusters = float(wavelength_clusters)
    if not math.isfinite(wavelength_clusters) or wavelength_clusters <= 0.0:
        raise ValueError("wavelength_clusters must be finite and positive")
    return pulse(time_s, phase_cycles=cluster / wavelength_clusters)


def coherence_pass(value: float, *, target: float = COHERENCE_TARGET) -> bool:
    """Evaluate the declared semantic-coherence target as an explicit gate."""

    value = float(value)
    target = float(target)
    if not math.isfinite(value) or not math.isfinite(target):
        return False
    if not 0.0 <= target <= 1.0:
        raise ValueError("target must be within [0, 1]")
    return value >= target


def frame_budget_pass(frame_ms: float, *, target_fps: float = TARGET_FPS) -> bool:
    """Check a measured frame time against a target; this is not a real-time guarantee."""

    frame_ms = float(frame_ms)
    target_fps = float(target_fps)
    if not math.isfinite(frame_ms) or frame_ms < 0.0:
        return False
    if not math.isfinite(target_fps) or target_fps <= 0.0:
        raise ValueError("target_fps must be finite and positive")
    return frame_ms <= 1000.0 / target_fps


def cluster_transforms() -> tuple[ClusterPose, ...]:
    """Materialize only 250 cluster transforms, never one million line objects."""

    return tuple(cluster_pose(index) for index in range(CLUSTER_COUNT))
