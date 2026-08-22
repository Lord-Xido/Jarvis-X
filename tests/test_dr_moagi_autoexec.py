from __future__ import annotations

import json

import pytest

from jarvisx.dr_moagi_autoexec import (
    AutoExecPolicy,
    DrMoagiAutoExecConfig,
    DrMoagiAutoExecutionEngine,
    HashChainJournal,
    SparseBlockCodec3D,
    SparseParser3D,
)
from jarvisx.dr_moagi_field_runtime import DrMoagiFieldConfig


def test_sparse_block_codec_compresses_and_materializes_only_requested_support():
    codec = SparseBlockCodec3D(
        AutoExecPolicy(block_size=2, quantization=0.01, prune_epsilon=0.0)
    )
    field = {(0, 0, 0): 1.0, (1, 0, 0): 0.0, (5, 5, 5): 0.8}

    latent = codec.encode(field)
    decoded = codec.decode(latent, [(0, 0, 0), (1, 0, 0)])

    assert latent.latent_cells == 2
    assert decoded == {(0, 0, 0): pytest.approx(0.5), (1, 0, 0): pytest.approx(0.5)}
    assert (5, 5, 5) not in decoded


def test_sparse_parser_prunes_without_allocating_dense_lattice():
    parser = SparseParser3D(
        side=1000,
        max_active_cells=4,
        value_min=-1.0,
        value_max=1.0,
        prune_epsilon=0.05,
    )

    parsed = parser.parse(
        [
            ((10, 20, 30), 0.9),
            ((999, 999, 999), 0.01),
            ((4, 5, 6), 5.0),
        ]
    )

    assert parsed == {(10, 20, 30): 0.9, (4, 5, 6): 1.0}
    with pytest.raises(ValueError, match="outside logical 3D lattice"):
        parser.parse([((1000, 0, 0), 1.0)])


def _engine_config(**overrides):
    values = dict(
        field_config=DrMoagiFieldConfig(
            side=8,
            alpha=0.5,
            lambda_residual=0.0,
            eta=0.0,
            dt=0.05,
            max_active_cells=256,
            expand_halo=False,
            prune_epsilon=0.0,
        ),
        policy=AutoExecPolicy(block_size=2, quantization=0.01, prune_epsilon=0.0),
        cycles=2,
        auto_optimize=False,
        max_reconstruction_mse=1.0,
    )
    values.update(overrides)
    return DrMoagiAutoExecConfig(**values)


def test_engine_runs_end_to_end_and_journal_verifies(tmp_path):
    journal_path = tmp_path / "omega-autoexec.jsonl"
    engine = DrMoagiAutoExecutionEngine(_engine_config(), journal_path=journal_path)
    engine.load({(2, 2, 2): 1.0, (3, 2, 2): 0.5})

    reports = engine.run()
    status = engine.status()

    assert len(reports) == 2
    assert all(report.committed for report in reports)
    assert all(len(report.journal_hash) == 64 for report in reports)
    assert status["cycle"] == 2
    assert status["journal_valid"] is True
    assert journal_path.exists()

    restored = HashChainJournal(journal_path)
    assert restored.verify()
    assert restored.head == engine.journal.head


def test_verification_rejects_lossy_transition_and_rolls_back():
    engine = DrMoagiAutoExecutionEngine(
        _engine_config(max_reconstruction_mse=0.0, cycles=1)
    )
    initial = {(2, 2, 2): 1.0, (3, 2, 2): 0.1}
    engine.load(initial)

    report = engine.step()

    assert not report.committed
    assert report.rejection_reason == "validator rejected candidate"
    assert engine.runtime.snapshot() == initial
    assert engine.journal.verify()


def test_bounded_optimizer_promotes_only_measured_improvement():
    field_config = DrMoagiFieldConfig(
        side=8,
        alpha=0.5,
        lambda_residual=0.0,
        eta=0.0,
        dt=0.05,
        max_active_cells=256,
        expand_halo=False,
        prune_epsilon=0.0,
    )
    config = DrMoagiAutoExecConfig(
        field_config=field_config,
        policy=AutoExecPolicy(block_size=1, quantization=0.01, prune_epsilon=0.0),
        cycles=1,
        auto_optimize=True,
        fidelity_weight=0.39,
        compression_weight=0.60,
        execution_weight=0.01,
    )
    engine = DrMoagiAutoExecutionEngine(config)
    engine.load(
        {
            (2, 2, 2): 1.0,
            (3, 2, 2): 1.0,
            (2, 3, 2): 1.0,
            (3, 3, 2): 1.0,
        }
    )

    report = engine.step()

    assert report.committed
    assert report.policy_promoted
    assert report.objective_after > report.objective_before
    assert engine.policy.block_size == 2
    assert engine.permeation_generation == 1
    assert engine.codec.policy == engine.policy
    assert engine.parser.prune_epsilon == engine.policy.prune_epsilon
    assert engine.runtime.config.prune_epsilon == engine.policy.prune_epsilon


def test_journal_detects_tampering(tmp_path):
    path = tmp_path / "journal.jsonl"
    journal = HashChainJournal(path)
    journal.append({"cycle": 1, "committed": True})

    envelope = json.loads(path.read_text(encoding="utf-8"))
    envelope["record"]["committed"] = False
    path.write_text(json.dumps(envelope) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="verification failed"):
        HashChainJournal(path)
