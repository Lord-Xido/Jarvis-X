"""Verified propose-shadow-commit loop for architectural world states."""

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Tuple

from .geometry import Vector3, inside, supports
from .world import ArchitecturalWorldModel, Relation, RelationKind


DEFAULT_WEIGHTS = {
    "observation": 0.00,
    "geometry": 0.10,
    "semantics": 0.10,
    "relations": 0.10,
    "hierarchy": 0.15,
    "architecture": 0.20,
    "physics": 0.30,
    "uncertainty": 0.10,
    "mdl": 0.05,
    "query": 0.00,
}


@dataclass(frozen=True)
class ObjectiveTerms:
    """Normalized components of the operational Dr Moagi objective."""

    observation: float = 0.0
    geometry: float = 0.0
    semantics: float = 0.0
    relations: float = 0.0
    hierarchy: float = 0.0
    architecture: float = 0.0
    physics: float = 0.0
    uncertainty: float = 0.0
    mdl: float = 0.0
    query: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        return {
            "observation": self.observation,
            "geometry": self.geometry,
            "semantics": self.semantics,
            "relations": self.relations,
            "hierarchy": self.hierarchy,
            "architecture": self.architecture,
            "physics": self.physics,
            "uncertainty": self.uncertainty,
            "mdl": self.mdl,
            "query": self.query,
        }

    def total(self, weights: Optional[Mapping[str, float]] = None) -> float:
        active_weights = DEFAULT_WEIGHTS if weights is None else weights
        values = self.as_dict()
        unknown = set(active_weights) - set(values)
        if unknown:
            raise ValueError("unknown objective terms: {}".format(sorted(unknown)))
        return sum(values[name] * weight for name, weight in active_weights.items())


@dataclass(frozen=True)
class EchoCandidate:
    """A shadow world-state proposal that has not yet been committed."""

    world: ArchitecturalWorldModel
    operation: str
    rationale: str = ""


@dataclass(frozen=True)
class VerificationReport:
    """Complete decision record for one candidate world state."""

    accepted: bool
    baseline_score: float
    candidate_score: float
    violations: Tuple[str, ...]
    baseline_fingerprint: str
    candidate_fingerprint: str
    operation: str

    @property
    def improvement(self) -> float:
        return self.baseline_score - self.candidate_score


@dataclass
class EchoController:
    """Deterministic world-state optimizer with shadow verification and rollback."""

    initial_world: ArchitecturalWorldModel
    description_budget: int = 100
    minimum_improvement: float = 1e-9
    support_tolerance: float = 0.05
    weights: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))

    def __post_init__(self) -> None:
        if self.description_budget <= 0:
            raise ValueError("description_budget must be positive")
        if self.minimum_improvement < 0.0:
            raise ValueError("minimum_improvement must be non-negative")
        self._world = self.initial_world.clone()
        self._history = [self._world.clone()]
        self.journal = []

    @property
    def world(self) -> ArchitecturalWorldModel:
        return self._world.clone()

    def evaluate(self, world: ArchitecturalWorldModel) -> ObjectiveTerms:
        entities = list(world.entities.values())
        relations = list(world.relations)

        entity_count = max(1, len(entities))
        relation_count = max(1, len(relations))

        geometry_errors = sum(
            1 for entity in entities if entity.bounds.volume <= 0.0
        ) / float(entity_count)
        semantic_error = sum(1.0 - entity.confidence for entity in entities) / float(
            entity_count
        )
        relation_error = sum(
            1.0 - relation.confidence for relation in relations
        ) / float(relation_count)
        uncertainty_error = sum(entity.uncertainty for entity in entities) / float(
            entity_count
        )

        parented = [entity for entity in entities if entity.parent_id is not None]
        hierarchy_errors = 0
        for entity in parented:
            parent = world.entities.get(entity.parent_id)
            if parent is None or not inside(
                entity.bounds, parent.bounds, tolerance=self.support_tolerance
            ):
                hierarchy_errors += 1
        hierarchy_error = hierarchy_errors / float(max(1, len(parented)))

        support_relations = [
            relation
            for relation in relations
            if relation.kind == RelationKind.SUPPORTS
        ]
        support_errors = 0
        for relation in support_relations:
            source = world.entities[relation.source_id]
            target = world.entities[relation.target_id]
            if not supports(
                source.bounds,
                target.bounds,
                tolerance=self.support_tolerance,
            ):
                support_errors += 1
        physics_error = support_errors / float(max(1, len(support_relations)))

        inside_relations = [
            relation for relation in relations if relation.kind == RelationKind.INSIDE
        ]
        inside_errors = 0
        for relation in inside_relations:
            source = world.entities[relation.source_id]
            target = world.entities[relation.target_id]
            if not inside(
                source.bounds,
                target.bounds,
                tolerance=self.support_tolerance,
            ):
                inside_errors += 1
        architecture_error = inside_errors / float(max(1, len(inside_relations)))

        mdl = (len(entities) + len(relations)) / float(self.description_budget)

        return ObjectiveTerms(
            geometry=geometry_errors,
            semantics=semantic_error,
            relations=relation_error,
            hierarchy=hierarchy_error,
            architecture=architecture_error,
            physics=physics_error,
            uncertainty=uncertainty_error,
            mdl=mdl,
        )

    def score(self, world: ArchitecturalWorldModel) -> float:
        return self.evaluate(world).total(self.weights)

    def propose_move(
        self,
        identifier: str,
        delta: Vector3,
        rationale: str = "",
    ) -> EchoCandidate:
        shadow = self._world.clone()
        shadow.move_entity(identifier, delta)
        return EchoCandidate(
            world=shadow,
            operation="move:{}:{},{},{}".format(
                identifier, delta.x, delta.y, delta.z
            ),
            rationale=rationale,
        )

    def propose_relation(
        self,
        relation: Relation,
        rationale: str = "",
    ) -> EchoCandidate:
        shadow = self._world.clone()
        shadow.add_relation(relation)
        return EchoCandidate(
            world=shadow,
            operation="relate:{}:{}:{}".format(
                relation.source_id,
                relation.kind.value,
                relation.target_id,
            ),
            rationale=rationale,
        )

    def verify(self, candidate: EchoCandidate) -> VerificationReport:
        baseline_score = self.score(self._world)
        candidate_score = self.score(candidate.world)
        violations = tuple(
            candidate.world.validate(support_tolerance=self.support_tolerance)
        )
        accepted = (
            not violations
            and candidate_score
            <= baseline_score - self.minimum_improvement
        )
        return VerificationReport(
            accepted=accepted,
            baseline_score=baseline_score,
            candidate_score=candidate_score,
            violations=violations,
            baseline_fingerprint=self._world.fingerprint(),
            candidate_fingerprint=candidate.world.fingerprint(),
            operation=candidate.operation,
        )

    def commit(
        self,
        candidate: EchoCandidate,
        report: Optional[VerificationReport] = None,
    ) -> ArchitecturalWorldModel:
        active_report = self.verify(candidate) if report is None else report
        if not active_report.accepted:
            raise ValueError("candidate failed verification")
        if active_report.candidate_fingerprint != candidate.world.fingerprint():
            raise ValueError("candidate changed after verification")

        committed = candidate.world.clone()
        committed.revision = self._world.revision + 1
        committed.metadata["previous_fingerprint"] = self._world.fingerprint()
        committed.metadata["operation"] = candidate.operation
        committed.metadata["rationale"] = candidate.rationale

        self._world = committed
        self._history.append(committed.clone())
        self.journal.append(
            {
                "event": "commit",
                "revision": committed.revision,
                "operation": candidate.operation,
                "baseline_score": active_report.baseline_score,
                "candidate_score": active_report.candidate_score,
                "improvement": active_report.improvement,
                "fingerprint": committed.fingerprint(),
            }
        )
        return committed.clone()

    def rollback(self) -> ArchitecturalWorldModel:
        if len(self._history) <= 1:
            raise RuntimeError("no committed revision available for rollback")
        removed = self._history.pop()
        self._world = self._history[-1].clone()
        self.journal.append(
            {
                "event": "rollback",
                "removed_revision": removed.revision,
                "restored_revision": self._world.revision,
                "fingerprint": self._world.fingerprint(),
            }
        )
        return self._world.clone()

    def auto_repair_supports(self, max_steps: int = 16) -> List[VerificationReport]:
        """Repair vertical support gaps using verified, reversible moves.

        This is intentionally bounded. It performs no unconstrained source-code
        rewriting and no opaque neural mutation; every candidate is evaluated,
        verified, fingerprinted, and journaled.
        """

        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")

        reports = []
        for _ in range(max_steps):
            invalid_relation = None
            for relation in self._world.iter_relations(RelationKind.SUPPORTS):
                supporter = self._world.entities[relation.source_id]
                supported = self._world.entities[relation.target_id]
                if not supports(
                    supporter.bounds,
                    supported.bounds,
                    tolerance=self.support_tolerance,
                ):
                    invalid_relation = relation
                    break

            if invalid_relation is None:
                break

            supporter = self._world.entities[invalid_relation.source_id]
            supported = self._world.entities[invalid_relation.target_id]
            delta_z = supporter.bounds.maximum.z - supported.bounds.minimum.z
            candidate = self.propose_move(
                supported.identifier,
                Vector3(0.0, 0.0, delta_z),
                rationale="close support gap declared by {}->{}".format(
                    supporter.identifier, supported.identifier
                ),
            )
            report = self.verify(candidate)
            reports.append(report)
            if not report.accepted:
                break
            self.commit(candidate, report)

        return reports
