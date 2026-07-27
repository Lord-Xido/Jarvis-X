"""Deterministic geometric primitives and architectural predicates."""

from dataclasses import dataclass
from math import sqrt


@dataclass(frozen=True)
class Vector3:
    """Three-dimensional Euclidean vector."""

    x: float
    y: float
    z: float

    def __add__(self, other: "Vector3") -> "Vector3":
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vector3") -> "Vector3":
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def scale(self, factor: float) -> "Vector3":
        return Vector3(self.x * factor, self.y * factor, self.z * factor)

    def norm(self) -> float:
        return sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)

    def as_tuple(self):
        return (self.x, self.y, self.z)


@dataclass(frozen=True)
class AABB:
    """Axis-aligned bounding box used by the deterministic spatial kernel."""

    minimum: Vector3
    maximum: Vector3

    def __post_init__(self) -> None:
        if (
            self.minimum.x > self.maximum.x
            or self.minimum.y > self.maximum.y
            or self.minimum.z > self.maximum.z
        ):
            raise ValueError("minimum corner must not exceed maximum corner")

    @property
    def center(self) -> Vector3:
        return Vector3(
            (self.minimum.x + self.maximum.x) / 2.0,
            (self.minimum.y + self.maximum.y) / 2.0,
            (self.minimum.z + self.maximum.z) / 2.0,
        )

    @property
    def size(self) -> Vector3:
        return self.maximum - self.minimum

    @property
    def volume(self) -> float:
        size = self.size
        return size.x * size.y * size.z

    @property
    def footprint_area(self) -> float:
        size = self.size
        return size.x * size.y

    def translated(self, delta: Vector3) -> "AABB":
        return AABB(self.minimum + delta, self.maximum + delta)

    def contains_point(self, point: Vector3, tolerance: float = 0.0) -> bool:
        return (
            self.minimum.x - tolerance <= point.x <= self.maximum.x + tolerance
            and self.minimum.y - tolerance <= point.y <= self.maximum.y + tolerance
            and self.minimum.z - tolerance <= point.z <= self.maximum.z + tolerance
        )

    def contains_box(self, other: "AABB", tolerance: float = 0.0) -> bool:
        return self.contains_point(other.minimum, tolerance) and self.contains_point(
            other.maximum, tolerance
        )

    def horizontal_overlap_area(self, other: "AABB") -> float:
        x_overlap = max(
            0.0,
            min(self.maximum.x, other.maximum.x)
            - max(self.minimum.x, other.minimum.x),
        )
        y_overlap = max(
            0.0,
            min(self.maximum.y, other.maximum.y)
            - max(self.minimum.y, other.minimum.y),
        )
        return x_overlap * y_overlap


def distance(first: AABB, second: AABB) -> float:
    """Euclidean center-to-center distance."""

    return (first.center - second.center).norm()


def intersects(first: AABB, second: AABB, tolerance: float = 0.0) -> bool:
    """Return whether two boxes overlap or touch within a tolerance."""

    return not (
        first.maximum.x < second.minimum.x - tolerance
        or second.maximum.x < first.minimum.x - tolerance
        or first.maximum.y < second.minimum.y - tolerance
        or second.maximum.y < first.minimum.y - tolerance
        or first.maximum.z < second.minimum.z - tolerance
        or second.maximum.z < first.minimum.z - tolerance
    )


def inside(inner: AABB, outer: AABB, tolerance: float = 0.0) -> bool:
    """Return whether ``inner`` is contained by ``outer``."""

    return outer.contains_box(inner, tolerance=tolerance)


def above(upper: AABB, lower: AABB, tolerance: float = 0.0) -> bool:
    """Return whether the upper object's bottom is above the lower object's top."""

    return upper.minimum.z >= lower.maximum.z - tolerance


def supports(
    supporter: AABB,
    supported: AABB,
    tolerance: float = 0.05,
    minimum_overlap_ratio: float = 0.2,
) -> bool:
    """Test contact and horizontal overlap for a support relationship.

    The overlap ratio is measured against the supported object's footprint,
    because a large floor supporting a small object should score as complete
    support rather than as a tiny fraction of the floor.
    """

    if not 0.0 <= minimum_overlap_ratio <= 1.0:
        raise ValueError("minimum_overlap_ratio must be in [0, 1]")

    vertical_gap = abs(supported.minimum.z - supporter.maximum.z)
    footprint = supported.footprint_area
    if footprint <= 0.0:
        return False

    overlap_ratio = supporter.horizontal_overlap_area(supported) / footprint
    return vertical_gap <= tolerance and overlap_ratio >= minimum_overlap_ratio
