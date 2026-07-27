"""Minimal executable Dr Moagi spatial echo demonstration."""

from jarvisx.spatial import (
    AABB,
    ArchitecturalWorldModel,
    EchoController,
    Entity,
    EntityKind,
    Relation,
    RelationKind,
    Vector3,
)


def main() -> None:
    world = ArchitecturalWorldModel(metadata={"source": "synthetic-demo"})
    world.add_entity(
        Entity(
            "table",
            EntityKind.OBJECT,
            AABB(Vector3(-1.0, -0.5, 0.0), Vector3(1.0, 0.5, 1.0)),
            semantic_label="table",
        )
    )
    world.add_entity(
        Entity(
            "lamp",
            EntityKind.OBJECT,
            AABB(Vector3(-0.2, -0.2, 1.2), Vector3(0.2, 0.2, 2.2)),
            semantic_label="lamp",
        )
    )
    world.add_relation(Relation("table", "lamp", RelationKind.SUPPORTS))

    controller = EchoController(world)
    print("baseline score:", controller.score(controller.world))
    print("baseline fingerprint:", controller.world.fingerprint())

    reports = controller.auto_repair_supports()
    for report in reports:
        print("operation:", report.operation)
        print("accepted:", report.accepted)
        print("improvement:", report.improvement)
        print("violations:", report.violations)

    print("committed revision:", controller.world.revision)
    print("committed score:", controller.score(controller.world))
    print("committed fingerprint:", controller.world.fingerprint())
    print("journal:", controller.journal)


if __name__ == "__main__":
    main()
