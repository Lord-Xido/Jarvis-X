from jarvisx.spatial import (
    AABB,
    ArchitecturalWorldModel,
    EchoController,
    Entity,
    EntityKind,
    Relation,
    RelationKind,
    Vector3,
    supports,
)


def box(x0, y0, z0, x1, y1, z1):
    return AABB(Vector3(x0, y0, z0), Vector3(x1, y1, z1))


def floating_lamp_world():
    world = ArchitecturalWorldModel()
    world.add_entity(
        Entity(
            identifier="table",
            kind=EntityKind.OBJECT,
            semantic_label="table",
            bounds=box(-1.0, -0.5, 0.0, 1.0, 0.5, 1.0),
        )
    )
    world.add_entity(
        Entity(
            identifier="lamp",
            kind=EntityKind.OBJECT,
            semantic_label="lamp",
            bounds=box(-0.2, -0.2, 1.2, 0.2, 0.2, 2.2),
        )
    )
    world.add_relation(
        Relation(
            source_id="table",
            target_id="lamp",
            kind=RelationKind.SUPPORTS,
        )
    )
    return world


def test_support_predicate_detects_gap_and_contact():
    table = box(-1.0, -0.5, 0.0, 1.0, 0.5, 1.0)
    floating = box(-0.2, -0.2, 1.2, 0.2, 0.2, 2.2)
    resting = box(-0.2, -0.2, 1.0, 0.2, 0.2, 2.0)

    assert not supports(table, floating)
    assert supports(table, resting)


def test_auto_repair_closes_loop_and_records_provenance():
    controller = EchoController(floating_lamp_world())
    baseline_score = controller.score(controller.world)

    reports = controller.auto_repair_supports()

    assert len(reports) == 1
    assert reports[0].accepted
    assert reports[0].candidate_score < reports[0].baseline_score

    committed = controller.world
    assert committed.revision == 1
    assert committed.metadata["operation"].startswith("move:lamp")
    assert len(committed.fingerprint()) == 64
    assert supports(
        committed.entities["table"].bounds,
        committed.entities["lamp"].bounds,
    )
    assert controller.score(committed) < baseline_score
    assert controller.journal[-1]["event"] == "commit"


def test_failed_candidate_does_not_mutate_committed_world():
    controller = EchoController(floating_lamp_world())
    baseline_fingerprint = controller.world.fingerprint()

    candidate = controller.propose_move(
        "lamp",
        Vector3(0.0, 0.0, 1.0),
        rationale="intentionally worsen the support gap",
    )
    report = controller.verify(candidate)

    assert not report.accepted
    assert report.violations
    assert controller.world.fingerprint() == baseline_fingerprint


def test_rollback_restores_previous_verified_revision():
    controller = EchoController(floating_lamp_world())
    reports = controller.auto_repair_supports()
    assert reports[0].accepted

    restored = controller.rollback()

    assert restored.revision == 0
    assert not supports(
        restored.entities["table"].bounds,
        restored.entities["lamp"].bounds,
    )
    assert controller.journal[-1]["event"] == "rollback"


def test_fingerprint_is_independent_of_insertion_order():
    first = ArchitecturalWorldModel()
    second = ArchitecturalWorldModel()

    table = Entity(
        identifier="table",
        kind=EntityKind.OBJECT,
        semantic_label="table",
        bounds=box(-1.0, -0.5, 0.0, 1.0, 0.5, 1.0),
    )
    lamp = Entity(
        identifier="lamp",
        kind=EntityKind.OBJECT,
        semantic_label="lamp",
        bounds=box(-0.2, -0.2, 1.0, 0.2, 0.2, 2.0),
    )

    first.add_entity(table)
    first.add_entity(lamp)
    second.add_entity(lamp)
    second.add_entity(table)

    relation = Relation("table", "lamp", RelationKind.SUPPORTS)
    first.add_relation(relation)
    second.add_relation(relation)

    assert first.canonical_json() == second.canonical_json()
    assert first.fingerprint() == second.fingerprint()
