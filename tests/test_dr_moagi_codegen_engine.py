from __future__ import annotations

import json
from pathlib import Path

import pytest

from jarvisx.dr_moagi_3d_animation_codec import EXPECTED_SHA256
from jarvisx.dr_moagi_codegen_engine import (
    GenerationConfig,
    HighThroughputCodeGenerator,
    autotune,
    generate_sharded,
)


def test_repeat_generation_is_exact_and_deterministic() -> None:
    config = GenerationConfig(lines=10_000, chunk_lines=1_024, template="pass  # deterministic")
    generator = HighThroughputCodeGenerator(config)

    first, first_metrics = generator.generate_to_memory()
    second, second_metrics = generator.generate_to_memory()

    assert first.count(b"\n") == 10_000
    assert first == second
    assert first_metrics.sha256 == second_metrics.sha256
    assert first_metrics.lines == 10_000
    assert first_metrics.bytes_emitted == len(first)
    assert first_metrics.provenance_sha256 == EXPECTED_SHA256
    assert first_metrics.strategy == "repeat"
    assert first_metrics.lines_per_second > 0


def test_indexed_generation_expands_global_indices() -> None:
    config = GenerationConfig(
        lines=5,
        chunk_lines=2,
        template="dm_generated_{index} = {index}",
    )
    payload, metrics = HighThroughputCodeGenerator(config).generate_to_memory()
    source = payload.decode("utf-8")

    assert source.splitlines() == [
        "dm_generated_0 = 0",
        "dm_generated_1 = 1",
        "dm_generated_2 = 2",
        "dm_generated_3 = 3",
        "dm_generated_4 = 4",
    ]
    compile(source, "<generated>", "exec")
    assert metrics.strategy == "indexed"


def test_invalid_multiline_template_is_rejected() -> None:
    with pytest.raises(ValueError, match="one physical source line"):
        GenerationConfig(lines=10, template="pass\npass").validate()


def test_autotune_selects_one_of_the_candidates() -> None:
    selected, trials = autotune(
        lines=20_000,
        template="pass  # autotune",
        candidates=(256, 1_024, 4_096),
    )

    assert selected.chunk_lines in {256, 1_024, 4_096}
    assert len(trials) == 3
    assert all(item.lines == 20_000 for item in trials)
    assert max(item.lines_per_second for item in trials) == max(
        trial.lines_per_second for trial in trials
    )


def test_sharded_generation_preserves_exact_line_count(tmp_path: Path) -> None:
    config = GenerationConfig(
        lines=1_003,
        chunk_lines=128,
        template="dm_generated_{index} = {index}",
    )
    manifest = generate_sharded(tmp_path, config, workers=4)

    assert manifest["lines"] == 1_003
    assert manifest["workers"] == 4
    assert manifest["provenance_sha256"] == EXPECTED_SHA256
    assert sum(int(shard["lines"]) for shard in manifest["shards"]) == 1_003

    generated_lines: list[str] = []
    for shard in manifest["shards"]:
        generated_lines.extend((tmp_path / str(shard["path"])).read_text().splitlines())
    assert len(generated_lines) == 1_003
    assert generated_lines[0] == "dm_generated_0 = 0"
    assert generated_lines[-1] == "dm_generated_1002 = 1002"

    stored_manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert stored_manifest["lines"] == 1_003
