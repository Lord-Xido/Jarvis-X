"""Canonical hierarchical world state for Dr Moagi spatial intelligence."""

import copy
import hashlib
import json
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Optional

from .geometry import AABB, Vector3, inside, supports


class EntityKind(str, Enum):
    """Architectural entity hierarchy from local geometry to whole systems."""

    SURFACE = "surface"
    PART = "part"
    OBJECT = "object"
    OPENING = "opening"
    BOUNDARY = "boundary"
    ROOM = "room"
    ZONE = "zone"
    BUILDING = "building"


class RelationKind(str, Enum):
    """Geometric, constructive, spatial, and functional predicates."""

    ABOVE = "above"
    BELOW = "below"
    INSIDE = "inside"
    INTERSECTS = "intersects"
    ALIGNED = "aligned"
    PARALLEL = "parallel"
    ORTHOGONAL = "orthogonal"
    SUPPORTS = "supports"
    ATTACHED_TO = "attached_to"
    EMBEDDED_IN = "embedded_in"
    SPANS = "spans"
    ENCLOSES = "encloses"
    ADJACENT = "adjacent"
    CONNECTED_THROUGH = "connected_through"
    VISIBLE_FROM = "visible_from"
    ACCESSIBLE_FROM = "accessible_from"
    USED_FOR = "used_for"
    SERVES = "serves"
    CONTROLS = "controls"
    PERMITS = "permits"
    OBSTRUCTS = "obstructs"


@dataclass(frozen=True)
class Entity:
    """One identified architectural or scene entity."""

    identifier: str
    kind: EntityKind
    bounds: AABB
    semantic_label: str = "unknown"
    confidence: float = 1.0
    uncertainty: float = 0.0
    parent_id: Optional[str] = None
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.identifier:
            raise ValueError("entity identifier must not be empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("entity confidence must be in [0, 1]")
        if not 0.0 <= self.uncertainty <= 1.0:
            raise ValueError("entity uncertainty must be in [0, 1]")

    def translated(self, delta: Vector3) -> "Entity":
        return replace(self, bounds=self.bounds.translated(delta))


@dataclass(frozen=True)
class Relation:
    """Directed typed relation between two world entities."""

    source_id: str
    target_id: str
    kind: RelationKind
    confidence: float = 1.0
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.source_id or not self.target_id:
            raise ValueError("relation endpoints must not be empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("relation confidence must be in [0, 1]")


@dataclass
class ArchitecturalWorldModel:
    """Auditable world hypothesis shared by rendering and reasoning layers."""

    entities: Dict[str, Entity] = field(default_factory=dict)
    relations: List[Relation] = field(default_factory=list)
    revision: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def clone(self) -> "ArchitecturalWorldModel":
        return copy.deepcopy(self)

    def add_entity(self, entity: Entity) -> None:
        if entity.identifier in self.entities:
            raise ValueError("duplicate entity identifier: {}".format(entity.identifier))
        if entity.parent_id is not None and entity.parent_id not in self.entities:
            raise ValueError("unknown parent entity: {}".format(entity.parent_id))
        self.entities[entity.identifier] = entity

    def replace_entity(self, entity: Entity) -> None:
        if entity.identifier not in self.entities:
            raise KeyError(entity.identifier)
        if entity.parent_id is not None and entity.parent_id not in self.entities:
            raise ValueError("unknown parent entity: {}".format(entity.parent_id))
        self.entities[entity.identifier] = entity

    def move_entity(self, identifier: str, delta: Vector3) -> None:
        self.replace_entity(self.entities[identifier].translated(delta))

    def add_relation(self, relation: Relation) -> None:
        if relation.source_id not in self.entities:
            raise ValueError("unknown relation source: {}".format(relation.source_id))
        if relation.target_id not in self.entities:
            raise ValueError("unknown relation target: {}".format(relation.target_id))
        if relation in self.relations:
            raise ValueError("duplicate relation")
        self.relations.append(relation)

    def iter_relations(self, kind: Optional[RelationKind] = None) -> Iterable[Relation]:
        for relation in self.relations:
            if kind is None or relation.kind == kind:
                yield relation

    def validate(self, support_tolerance: float = 0.05) -> List[str]:
        """Return deterministic hard-constraint violations."""

        violations = []
        for entity in self.entities.values():
            if entity.parent_id is not None and entity.parent_id not in self.entities:
                violations.append(
                    "entity {} references missing parent {}".format(
                        entity.identifier, entity.parent_id
                    )
                )

        for relation in self.relations:
            source = self.entities.get(relation.source_id)
            target = self.entities.get(relation.target_id)
            if source is None or target is None:
                violations.append(
                    "relation {}->{} has missing endpoint".format(
                        relation.source_id, relation.target_id
                    )
                )
                continue

            if relation.kind == RelationKind.SUPPORTS and not supports(
                source.bounds,
                target.bounds,
                tolerance=support_tolerance,
            ):
                violations.append(
                    "{} does not geometrically support {}".format(
                        relation.source_id, relation.target_id
                    )
                )

            if relation.kind == RelationKind.INSIDE and not inside(
                source.bounds, target.bounds, tolerance=support_tolerance
            ):
                violations.append(
                    "{} is not geometrically inside {}".format(
                        relation.source_id, relation.target_id
                    )
                )

        return violations

    def canonical_dict(self) -> Dict[str, Any]:
        """Return a stable, serialization-ready representation."""

        entities = []
        for identifier in sorted(self.entities):
            entity = self.entities[identifier]
            entities.append(
                {
                    "identifier": entity.identifier,
                    "kind": entity.kind.value,
                    "semantic_label": entity.semantic_label,
                    "confidence": entity.confidence,
                    "uncertainty": entity.uncertainty,
                    "parent_id": entity.parent_id,
                    "bounds": {
                        "minimum": entity.bounds.minimum.as_tuple(),
                        "maximum": entity.bounds.maximum.as_tuple(),
                    },
                    "attributes": dict(sorted(entity.attributes.items())),
                }
            )

        relations = []
        for relation in sorted(
            self.relations,
            key=lambda value: (
                value.source_id,
                value.kind.value,
                value.target_id,
            ),
        ):
            relations.append(
                {
                    "source_id": relation.source_id,
                    "target_id": relation.target_id,
                    "kind": relation.kind.value,
                    "confidence": relation.confidence,
                    "attributes": dict(sorted(relation.attributes.items())),
                }
            )

        return {
            "revision": self.revision,
            "entities": entities,
            "relations": relations,
            "metadata": dict(sorted(self.metadata.items())),
        }

    def canonical_json(self) -> str:
        return json.dumps(
            self.canonical_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )

    def fingerprint(self) -> str:
        """Compute a real SHA-256 digest over the canonical world state."""

        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()
