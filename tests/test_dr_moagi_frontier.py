from __future__ import annotations

import pytest

from jarvisx.dr_moagi_autoexec import AutoExecPolicy
from jarvisx.dr_moagi_frontier import (
    BenchmarkEvidence,
    DrMoagiFrontierRuntime,
    FrontierConfig,
    HierarchicalSparseGrid3D,
    SOTAClaimGate,
    SolverConfig,
    SparseAndersonSolver3D,
    SparseEntropyCodec3D,
    morton3_decode,
    morton3_encode,
)


def test_morton_roundtrip_and_hierarchy_profile() -> None:
    assert morton3_decode(morton3_encode(7, 4, 9)) == (7, 4, 9)
    grid = HierarchicalSparseGrid3D.from_field(
        {
            (0, 0, 0): 1.0,
            (1, 0, 0): 0.5,
            (7, 7, 7): -0.25,
        },
        side=8,
    )

    profile = grid.occupancy_profile()

    assert grid.active_cells == 3
    assert profile[0] == 3
    assert profile[-1] == 1
    assert all(left >= right for left, right in zip(profile, profile[1:]))


def test_entropy_packet_roundtrips_quantized_sparse_field() -> None:
    codec = SparseEntropyCodec3D()
    source = {
        (1, 2, 0): 0.91,
        (1, 2, 64): -0.39,
        (8, 8, 8): 0.123,
    }

    packet = codec.encode(source, side=130, quantization=0.01)
    decoded = codec.decode(packet)

    assert packet.encoded_bytes > 0
    assert packet.active_cells == 3
    assert len(packet.checksum_sha256) == 64
    assert decoded[(1, 2, 0)] == pytest.approx(0.91)
    assert decoded[(1, 2, 64)] == pytest.approx(-0.39)
    assert decoded[(8, 8, 8)] == pytest.approx(0.12)


def test_anderson_accelerates_affine_fixed_point() -> None:
    solver = SparseAndersonSolver3D(
        SolverConfig(
            tolerance=1.0e-8,
            max_iterations=30,
            depth=4,
            damping=1.0,
        )
    )

    def operator(field: dict[tuple[int, int, int], float]) -> dict[tuple[int, int, int], float]:
        value = float(field.get((0, 0, 0), 0.0))
        return {(0, 0, 0): 0.5 * value + 0.5}

    initial = {(0, 0, 0): 0.0}
    plain = solver.solve_plain(operator, initial)
    accelerated = solver.solve(operator, initial)

    assert plain.converged
    assert accelerated.converged
    assert accelerated.iterations < plain.iterations
    assert accelerated.state[(0, 0, 0)] == pytest.approx(1.0, abs=1.0e-6)


def test_frontier_runtime_transaction_selects_no_worse_internal_baseline(tmp_path) -> None:
    runtime = DrMoagiFrontierRuntime(
        FrontierConfig(
            side=16,
            max_active_cells=4_096,
            policy=AutoExecPolicy(block_size=2, quantization=0.01, prune_epsilon=0.0),
            max_iterations=12,
        ),
        journal_path=tmp_path / "frontier.jsonl",
    )
    runtime.load(
        {
            (4, 4, 4): 1.0,
            (5, 4, 4): 0.7,
            (8, 8, 8): 0.5,
        }
    )

    report = runtime.step()

    assert report.committed
    assert report.active_cells_after > 0
    assert report.selected_objective <= report.plain_objective + 1.0e-15
    assert report.selected_objective <= report.accelerated_objective + 1.0e-15
    assert report.selected_solver in {"plain", "anderson"}
    assert runtime.journal.verify()
    assert runtime.status()["claim_status"] == (
        "frontier-candidate-until-external-benchmark-gate-passes"
    )


def test_sota_claim_gate_requires_provenance_and_margin() -> None:
    gate = SOTAClaimGate()

    missing_source = gate.evaluate(
        [
            BenchmarkEvidence(
                metric="latency_ms",
                candidate_value=8.0,
                reference_value=10.0,
                higher_is_better=False,
                minimum_relative_gain=0.10,
            )
        ]
    )
    assert not missing_source.passed

    verified = gate.evaluate(
        [
            BenchmarkEvidence(
                metric="latency_ms",
                candidate_value=8.0,
                reference_value=10.0,
                higher_is_better=False,
                minimum_relative_gain=0.10,
                source="reproducible-workload://reference",
            ),
            BenchmarkEvidence(
                metric="throughput",
                candidate_value=120.0,
                reference_value=100.0,
                higher_is_better=True,
                minimum_relative_gain=0.10,
                source="reproducible-workload://reference",
            ),
        ]
    )

    assert verified.passed
    assert all(check[1] for check in verified.checks)
