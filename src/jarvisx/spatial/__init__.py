"""Operational spatial-world kernel for the Dr Moagi framework.

The package keeps perception models separate from the auditable world state.
External encoders can populate :class:`ArchitecturalWorldModel`; the geometry,
verification, echo, and provenance layers remain deterministic and testable.
"""

from .echo import EchoCandidate, EchoController, ObjectiveTerms, VerificationReport
from .geometry import AABB, Vector3, above, inside, intersects, supports
from .world import (
    ArchitecturalWorldModel,
    Entity,
    EntityKind,
    Relation,
    RelationKind,
)

__all__ = [
    "AABB",
    "ArchitecturalWorldModel",
    "EchoCandidate",
    "EchoController",
    "Entity",
    "EntityKind",
    "ObjectiveTerms",
    "Relation",
    "RelationKind",
    "Vector3",
    "VerificationReport",
    "above",
    "inside",
    "intersects",
    "supports",
]
