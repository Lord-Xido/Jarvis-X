from jarvisx.operational_field import OperationalTetrationFieldAutomaton
from jarvisx.tetration_field import (
    FieldMechanics,
    TetrationAddress,
    make_brick_pulse,
)


def origin():
    return TetrationAddress(2, "origin", 0, 0, 0)


def make_engine(seed=1337):
    return OperationalTetrationFieldAutomaton(
        mechanics=FieldMechanics(max_active_bricks=32),
        latent_dim=8,
        expert_count=3,
        seed=seed,
        bucket_count=17,
    )


def test_operational_commit_materialises_matching_latent_repository():
    engine = make_engine()
    metrics = engine.step({origin(): make_brick_pulse(20.0)})

    assert metrics.committed
    assert len(engine.latent_repository) == metrics.materialised_bricks
    assert set(engine.latent_repository) == set(engine.directory.to_dict())
    assert all(len(latent) == 8 for latent in engine.latent_repository.values())

    snapshot = engine.snapshot()
    assert snapshot["physical_state"] == ["active_frontier", "B", "Z", "Omega"]
    assert snapshot["latent_repository_entries"] == snapshot["materialised_bricks"]
    assert snapshot["journal_hash"] == metrics.journal_hash


def test_operational_replay_includes_latent_state_and_sealed_journal():
    left = make_engine(seed=91)
    right = make_engine(seed=91)
    workload = [{origin(): make_brick_pulse(18.0)}, None, None]

    for injections in workload:
        assert left.step(injections) == right.step(injections)

    assert left.directory.to_dict() == right.directory.to_dict()
    assert left.latent_repository == right.latent_repository
    assert left.journal_hash == right.journal_hash


def test_latent_failure_rolls_back_B_Z_and_omega_atomically(monkeypatch):
    engine = make_engine(seed=17)
    assert engine.step({origin(): make_brick_pulse(16.0)}).committed

    previous_cycle = engine.cycle
    previous_hash = engine.journal_hash
    previous_field = engine.directory.to_dict()
    previous_latents = dict(engine.latent_repository)

    def reject_latents(_states):
        raise ValueError("forced latent verification failure")

    monkeypatch.setattr(engine, "_build_latent_repository", reject_latents)
    metrics = engine.step()

    assert not metrics.committed
    assert metrics.rollback_reason.startswith("latent commit failed")
    assert engine.cycle == previous_cycle
    assert engine.journal_hash == previous_hash
    assert engine.directory.to_dict() == previous_field
    assert engine.latent_repository == previous_latents
