import math
import struct

import pytest

from jarvisx.multiparallel import (
    BranchSnapshot,
    CodeGeometry,
    Compression,
    DataKind,
    EncodedChunk,
    EvolutionConfig,
    FramedArtifact,
    JarvisX3DEngine,
    Mesh,
    PackageReceipt,
    ParallelPipeline,
    PipelineNode,
    PipelineTopology,
    RuntimeLimits,
    SpatialProcessor,
    StageKind,
    TopologyEvolution,
    Vector3,
    VertexBatch,
    default_topology,
    topology_from_stages,
)


def test_vector_batch_and_mesh_are_finite_and_index_safe() -> None:
    batch = VertexBatch((Vector3(1, 2, 3), Vector3(-1, -2, -3)))
    mesh = Mesh(batch, ((0, 1, 0),))

    assert len(batch) == 2
    assert mesh.vertices == batch

    with pytest.raises(ValueError, match="finite"):
        Vector3(math.inf, 0, 0)
    with pytest.raises(ValueError, match="outside"):
        Mesh(batch, ((0, 1, 2),))


def test_topology_rejects_cycles_disconnected_nodes_and_duplicate_paths() -> None:
    nodes = (
        PipelineNode("a", StageKind.LOAD),
        PipelineNode("b", StageKind.ENCODE),
        PipelineNode("c", StageKind.COMPRESS),
    )

    with pytest.raises(ValueError, match="cycle|one source"):
        PipelineTopology(nodes, (("a", "b"), ("b", "a")))
    with pytest.raises(ValueError, match="one source|one path"):
        PipelineTopology(nodes, (("a", "b"),))
    with pytest.raises(ValueError, match="single linear DAG"):
        PipelineTopology(nodes, (("a", "b"), ("a", "c")))


def test_default_pipeline_preserves_text_exactly_and_frames_chunks() -> None:
    source = "def kinetic(value):\n    return value * 3\n" * 12
    topology = default_topology(parallelism=4, batch_size=32, compression_level=6)

    run = ParallelPipeline(topology).run(source, workers=4)

    assert run.success
    assert isinstance(run.output, FramedArtifact)
    assert run.output.decode() == source
    assert run.stats.package_count == 4
    assert run.stats.codec_runtime_version
    assert [receipt.sequence for receipt in run.receipts] == [0, 1, 2, 3]
    assert run.stats.compression_ratio > 0.0


def test_process_and_sequential_backends_reconcile_to_same_artifact() -> None:
    source = "parallel deterministic payload\n" * 80
    topology = default_topology(parallelism=3, batch_size=64, compression_level=4)
    pipeline = ParallelPipeline(topology)

    sequential = pipeline.run(source, workers=3, backend="sequential")
    process = pipeline.run(source, workers=3, backend="process")

    assert sequential.success and process.success
    assert isinstance(sequential.output, FramedArtifact)
    assert isinstance(process.output, FramedArtifact)
    assert sequential.output.to_bytes() == process.output.to_bytes()
    assert sequential.run_id == process.run_id


def test_merge_order_is_source_order_not_completion_order() -> None:
    receipts = (
        PackageReceipt("third", 2, True, "C", "c" * 64, (), 1),
        PackageReceipt("first", 0, True, "A", "a" * 64, (), 100),
        PackageReceipt("second", 1, True, "B", "b" * 64, (), 10),
    )

    assert ParallelPipeline.merge_receipts(receipts) == "ABC"


def test_vertex_transform_is_typed_and_round_trips_through_frame() -> None:
    batch = VertexBatch((Vector3(1, 2, 3), Vector3(4, 5, 6)))
    nodes = (
        PipelineNode("load", StageKind.LOAD),
        PipelineNode(
            "rotate",
            StageKind.TRANSFORM,
            (("axis_order", "yzx"), ("scale", 2.0)),
        ),
        PipelineNode("encode", StageKind.ENCODE),
        PipelineNode("compress", StageKind.COMPRESS),
    )
    topology = PipelineTopology(
        nodes,
        (("load", "rotate"), ("rotate", "encode"), ("encode", "compress")),
        parallelism=2,
        batch_size=1,
    )

    run = ParallelPipeline(topology).run(batch, workers=2)

    assert run.success
    assert isinstance(run.output, FramedArtifact)
    assert run.output.decode() == VertexBatch(
        (Vector3(4, 6, 2), Vector3(10, 12, 8))
    )


def test_mesh_is_never_split_and_face_indices_survive_round_trip() -> None:
    mesh = Mesh(
        VertexBatch((Vector3(0, 0, 0), Vector3(1, 0, 0), Vector3(0, 1, 0))),
        ((0, 1, 2),),
    )

    run = ParallelPipeline(default_topology(batch_size=1)).run(mesh, workers=4)

    assert run.success
    assert run.stats.package_count == 1
    assert isinstance(run.output, FramedArtifact)
    assert run.output.decode() == mesh


def test_framed_artifact_round_trip_and_tamper_detection() -> None:
    artifact = FramedArtifact(
        DataKind.TEXT,
        (
            EncodedChunk.encode("alpha").compress(6),
            EncodedChunk.encode("beta").compress(6),
        ),
    )
    frame = artifact.to_bytes()

    restored = FramedArtifact.from_bytes(frame)

    assert restored == artifact
    assert restored.decode() == "alphabeta"
    tampered = frame[:-1] + bytes((frame[-1] ^ 0x01,))
    with pytest.raises(ValueError, match="invalid|integrity|match"):
        FramedArtifact.from_bytes(tampered)


def test_framed_artifact_rejects_declared_zip_bomb_size() -> None:
    chunk = EncodedChunk.encode("bounded").compress(6)
    frame = bytearray(FramedArtifact(DataKind.TEXT, (chunk,)).to_bytes())
    raw_size_offset = 4 + 6 + 1
    struct.pack_into(">Q", frame, raw_size_offset, 1_000_000)

    with pytest.raises(ValueError, match="decoded-output limit"):
        FramedArtifact.from_bytes(bytes(frame), max_output_bytes=32)


def test_code_geometry_is_observational_and_does_not_rewrite_source() -> None:
    source = "def add(left, right):\n    total = left + right\n    return total\n"
    geometry = SpatialProcessor.map_code(source)
    rotated = geometry.transform("yzx", 2.0)

    assert isinstance(geometry, CodeGeometry)
    assert geometry.source_line_count == 3
    assert geometry.points[0].coordinate == Vector3(1, 0, 3)
    assert geometry.points[1].coordinate.y == 4
    assert rotated.source_sha256 == geometry.source_sha256
    assert rotated.points[0].coordinate == Vector3(0, 6, 2)
    compile(source, "<test>", "exec")

    with pytest.raises(ValueError, match="syntactically valid"):
        SpatialProcessor.map_code("def broken(:\n")


def test_stage_type_failure_returns_receipt_and_never_commits_engine_run() -> None:
    topology = topology_from_stages((StageKind.LOAD, StageKind.COMPRESS))
    engine = JarvisX3DEngine(topology=topology)

    run = engine.process_parallel("not encoded")

    assert not run.success
    assert run.output is None
    assert run.receipts[0].error_type == "TypeError"
    assert engine.last_committed_run is None
    assert engine.stats.failed_runs == 1


def test_decode_pipeline_merges_raw_text_without_inserting_newlines() -> None:
    topology = topology_from_stages(
        (
            StageKind.LOAD,
            StageKind.ENCODE,
            StageKind.COMPRESS,
            StageKind.DECODE,
        ),
        parallelism=3,
        batch_size=3,
    )
    source = "abcdefghi"

    run = ParallelPipeline(topology).run(source, workers=3)

    assert run.success
    assert run.output == source


def test_branches_are_immutable_isolated_and_merge_in_requested_order() -> None:
    engine = JarvisX3DEngine(topology=default_topology(batch_size=2))
    alpha_id = engine.create_branch("alpha", "alpha")
    beta_id = engine.create_branch("beta", "beta")
    before = engine.branch(alpha_id)

    alpha_run = engine.process_branch(alpha_id, num_workers=2)
    beta_run = engine.process_branch(beta_id, num_workers=2)
    merged = engine.merge_branches((beta_id, alpha_id))

    assert isinstance(before, BranchSnapshot)
    assert before.payload == "alpha"
    assert before.run is None
    assert alpha_run.success and beta_run.success
    assert isinstance(merged, FramedArtifact)
    assert merged.decode() == "betaalpha"
    assert engine.branch(alpha_id).payload == before.payload


def test_branch_merge_rejects_mixed_data_kinds() -> None:
    engine = JarvisX3DEngine()
    text_id = engine.create_branch("text", "same bytes")
    bytes_id = engine.create_branch("bytes", b"same bytes")
    engine.process_branch(text_id)
    engine.process_branch(bytes_id)

    with pytest.raises(TypeError, match="incompatible data kinds"):
        engine.merge_branches((text_id, bytes_id))


def test_seeded_evolution_is_reproducible_and_ignores_wall_clock_in_fitness() -> None:
    limits = RuntimeLimits(max_workers=4, max_population=12, max_generations=8)
    base = default_topology(parallelism=2, batch_size=16)
    data = "evolution payload with repeated structure\n" * 30
    config = EvolutionConfig(generations=3, population_size=8, seed=1234)

    left = TopologyEvolution(limits).evolve(base, data, config)
    right = TopologyEvolution(limits).evolve(base, data, config)

    assert left.selected_topology.digest_sha256 == right.selected_topology.digest_sha256
    assert left.best_score.fitness == right.best_score.fitness
    assert left.best_score.estimated_work_units == right.best_score.estimated_work_units
    assert [report.best_topology_digest for report in left.history] == [
        report.best_topology_digest for report in right.history
    ]


def test_engine_promotes_only_a_verified_candidate_topology() -> None:
    limits = RuntimeLimits(max_workers=4, max_population=8, max_generations=4)
    engine = JarvisX3DEngine(limits=limits)
    data = "candidate-first topology search\n" * 20

    result = engine.auto_evolve(
        data,
        EvolutionConfig(generations=2, population_size=6, seed=11),
    )

    assert result.promoted
    assert engine.topology == result.selected_topology
    assert engine.stats.evolution_generations == 2
    assert engine.stats.best_fitness == result.best_score.fitness


def test_failed_evolution_configuration_preserves_active_topology() -> None:
    limits = RuntimeLimits(max_population=4, max_generations=2)
    engine = JarvisX3DEngine(limits=limits)
    before = engine.topology

    with pytest.raises(ValueError, match="population exceeds"):
        engine.auto_evolve(
            "bounded",
            EvolutionConfig(generations=1, population_size=5),
        )

    assert engine.topology == before


def test_resource_bounds_reject_workers_input_and_branch_overflow() -> None:
    limits = RuntimeLimits(max_workers=2, max_input_bytes=16, max_branches=1)
    engine = JarvisX3DEngine(limits=limits)

    with pytest.raises(ValueError, match="worker limit"):
        engine.process_parallel("small", num_workers=3)
    with pytest.raises(ValueError, match="byte limit"):
        engine.process_parallel("x" * 17)

    engine.create_branch("one", "small")
    with pytest.raises(ValueError, match="branch limit"):
        engine.create_branch("two", "small")


def test_encoded_chunk_validates_integrity_and_compression_level() -> None:
    chunk = EncodedChunk.encode(b"payload")

    assert chunk.compression is Compression.NONE
    assert chunk.decode() == b"payload"
    with pytest.raises(ValueError, match="compression level"):
        chunk.compress(10)
    with pytest.raises(ValueError, match="integrity"):
        EncodedChunk(DataKind.BYTES, b"wrong", 5, "0" * 64)
