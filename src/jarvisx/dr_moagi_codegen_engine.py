"""High-throughput deterministic code-generation plane for Jarvis-X.

This module turns the Dr Moagi runtime into a measurable source-emission engine.
The 1,000,000 lines/second figure is treated as an empirical throughput target,
not a hardware-independent guarantee. Semantic synthesis and mechanical source
emission are deliberately reported as separate concerns.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import BinaryIO, Sequence, cast

from jarvisx.dr_moagi_3d_animation_codec import EXPECTED_SHA256

TARGET_LINES_PER_SECOND = 1_000_000.0
DEFAULT_LINES = 1_000_000
DEFAULT_CHUNK_LINES = 65_536
DEFAULT_TEMPLATE = "pass  # DM-vOmegaXi generated scaffold"
INDEX_TOKEN = "{index}"
AUTOTUNE_CHUNKS = (4_096, 16_384, 65_536, 262_144)


@dataclass(frozen=True)
class GenerationConfig:
    """Configuration for one deterministic generation run."""

    lines: int = DEFAULT_LINES
    target_lps: float = TARGET_LINES_PER_SECOND
    chunk_lines: int = DEFAULT_CHUNK_LINES
    template: str = DEFAULT_TEMPLATE
    digest: bool = True

    def validate(self) -> None:
        if self.lines <= 0:
            raise ValueError("lines must be positive")
        if self.target_lps <= 0:
            raise ValueError("target_lps must be positive")
        if self.chunk_lines <= 0:
            raise ValueError("chunk_lines must be positive")
        if "\n" in self.template or "\r" in self.template:
            raise ValueError("template must describe exactly one physical source line")
        if not self.template.strip():
            raise ValueError("template must not be blank")


@dataclass(frozen=True)
class GenerationMetrics:
    """Measured output properties for a generation run."""

    lines: int
    bytes_emitted: int
    elapsed_seconds: float
    lines_per_second: float
    megabytes_per_second: float
    target_lps: float
    target_ratio: float
    target_met: bool
    strategy: str
    chunk_lines: int
    sha256: str | None
    provenance_sha256: str
    unique_line_semantics: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


class NullBinaryWriter:
    """Binary sink that counts bytes without retaining them."""

    def __init__(self) -> None:
        self.bytes_written = 0

    def write(self, data: bytes) -> int:
        size = len(data)
        self.bytes_written += size
        return size

    def flush(self) -> None:
        return None


class CodeTemplate:
    """Compile a one-line template into repeat or globally indexed shards."""

    def __init__(self, template: str) -> None:
        self.template = template
        self.strategy = "indexed" if INDEX_TOKEN in template else "repeat"
        self._repeat_line = None
        if self.strategy == "repeat":
            self._repeat_line = (template + "\n").encode("utf-8")

    def validate_syntax(self) -> None:
        if self.strategy == "repeat":
            sample = self.template + "\n"
        else:
            sample = self.template.format(index=0) + "\n"
        compile(sample, "<dr-moagi-codegen-template>", "exec")

    def build(self, start_index: int, line_count: int) -> bytes:
        if line_count <= 0:
            return b""
        if self._repeat_line is not None:
            return self._repeat_line * line_count
        stop = start_index + line_count
        return "".join(
            self.template.format(index=index) + "\n" for index in range(start_index, stop)
        ).encode("utf-8")


class HighThroughputCodeGenerator:
    """Bounded, deterministic source generator with runtime throughput telemetry."""

    def __init__(self, config: GenerationConfig) -> None:
        config.validate()
        self.config = config
        self.program = CodeTemplate(config.template)
        self.program.validate_syntax()

    def _metrics(
        self,
        *,
        bytes_emitted: int,
        elapsed: float,
        digest: str | None,
    ) -> GenerationMetrics:
        elapsed = max(elapsed, 1.0e-12)
        lps = self.config.lines / elapsed
        mbps = bytes_emitted / elapsed / 1_000_000.0
        ratio = lps / self.config.target_lps
        semantics = "one precompiled line repeated" if self.program.strategy == "repeat" else (
            "globally indexed template expansion"
        )
        return GenerationMetrics(
            lines=self.config.lines,
            bytes_emitted=bytes_emitted,
            elapsed_seconds=elapsed,
            lines_per_second=lps,
            megabytes_per_second=mbps,
            target_lps=self.config.target_lps,
            target_ratio=ratio,
            target_met=ratio >= 1.0,
            strategy=self.program.strategy,
            chunk_lines=self.config.chunk_lines,
            sha256=digest,
            provenance_sha256=EXPECTED_SHA256,
            unique_line_semantics=semantics,
        )

    def generate(self, writer: BinaryIO | NullBinaryWriter) -> GenerationMetrics:
        """Generate exactly ``config.lines`` physical lines into a binary writer."""

        hasher = hashlib.sha256() if self.config.digest else None
        bytes_emitted = 0
        produced = 0
        cached_repeat = None
        if self.program.strategy == "repeat":
            cached_repeat = self.program.build(0, self.config.chunk_lines)

        started = time.perf_counter()
        while produced < self.config.lines:
            count = min(self.config.chunk_lines, self.config.lines - produced)
            if cached_repeat is not None and count == self.config.chunk_lines:
                payload = cached_repeat
            else:
                payload = self.program.build(produced, count)
            written = writer.write(payload)
            if written != len(payload):
                raise OSError(f"short write: expected {len(payload)} bytes, wrote {written}")
            if hasher is not None:
                hasher.update(payload)
            bytes_emitted += written
            produced += count
        writer.flush()
        elapsed = time.perf_counter() - started
        digest = hasher.hexdigest() if hasher is not None else None
        return self._metrics(bytes_emitted=bytes_emitted, elapsed=elapsed, digest=digest)

    def generate_to_memory(self) -> tuple[bytes, GenerationMetrics]:
        buffer = io.BytesIO()
        metrics = self.generate(buffer)
        return buffer.getvalue(), metrics

    def generate_to_file(self, output: Path) -> GenerationMetrics:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("wb", buffering=1024 * 1024) as stream:
            return self.generate(stream)

    def benchmark(self) -> GenerationMetrics:
        return self.generate(NullBinaryWriter())


def autotune(
    *,
    lines: int = DEFAULT_LINES,
    target_lps: float = TARGET_LINES_PER_SECOND,
    template: str = DEFAULT_TEMPLATE,
    candidates: Sequence[int] = AUTOTUNE_CHUNKS,
) -> tuple[GenerationConfig, list[GenerationMetrics]]:
    """Select the fastest chunk size on the current machine using a null sink."""

    if not candidates:
        raise ValueError("at least one chunk candidate is required")
    trials: list[GenerationMetrics] = []
    for chunk_lines in candidates:
        config = GenerationConfig(
            lines=lines,
            target_lps=target_lps,
            chunk_lines=int(chunk_lines),
            template=template,
            digest=True,
        )
        trials.append(HighThroughputCodeGenerator(config).benchmark())
    best = max(trials, key=lambda item: item.lines_per_second)
    selected = GenerationConfig(
        lines=lines,
        target_lps=target_lps,
        chunk_lines=best.chunk_lines,
        template=template,
        digest=True,
    )
    return selected, trials


def _shard_plan(total_lines: int, workers: int) -> list[tuple[int, int, int]]:
    workers = max(1, min(workers, total_lines))
    base, remainder = divmod(total_lines, workers)
    plan: list[tuple[int, int, int]] = []
    start = 0
    for shard_id in range(workers):
        count = base + (1 if shard_id < remainder else 0)
        plan.append((shard_id, start, count))
        start += count
    return plan


def generate_sharded(
    directory: Path,
    config: GenerationConfig,
    *,
    workers: int | None = None,
) -> dict[str, object]:
    """Generate independent source shards concurrently with a manifest."""

    config.validate()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    worker_count = workers or max(1, min(os.cpu_count() or 1, 8))
    plan = _shard_plan(config.lines, worker_count)
    started = time.perf_counter()

    def emit(item: tuple[int, int, int]) -> dict[str, object]:
        shard_id, start, count = item
        shard_template = config.template
        if INDEX_TOKEN in shard_template:
            program = CodeTemplate(shard_template)
            program.validate_syntax()
            path = directory / f"shard-{shard_id:04d}.py"
            hasher = hashlib.sha256()
            written = 0
            produced = 0
            with path.open("wb", buffering=1024 * 1024) as stream:
                while produced < count:
                    take = min(config.chunk_lines, count - produced)
                    payload = program.build(start + produced, take)
                    stream.write(payload)
                    hasher.update(payload)
                    written += len(payload)
                    produced += take
            return {
                "id": shard_id,
                "path": path.name,
                "start": start,
                "lines": count,
                "bytes": written,
                "sha256": hasher.hexdigest(),
            }

        shard_config = GenerationConfig(
            lines=count,
            target_lps=config.target_lps,
            chunk_lines=config.chunk_lines,
            template=shard_template,
            digest=config.digest,
        )
        path = directory / f"shard-{shard_id:04d}.py"
        metrics = HighThroughputCodeGenerator(shard_config).generate_to_file(path)
        return {
            "id": shard_id,
            "path": path.name,
            "start": start,
            "lines": count,
            "bytes": metrics.bytes_emitted,
            "sha256": metrics.sha256,
        }

    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        shards = list(pool.map(emit, plan))
    elapsed = max(time.perf_counter() - started, 1.0e-12)
    bytes_emitted = sum(cast(int, item["bytes"]) for item in shards)
    lps = config.lines / elapsed
    manifest: dict[str, object] = {
        "schema": "jarvisx.dr-moagi-codegen.v1",
        "provenance_sha256": EXPECTED_SHA256,
        "lines": config.lines,
        "bytes": bytes_emitted,
        "elapsed_seconds": elapsed,
        "lines_per_second": lps,
        "target_lps": config.target_lps,
        "target_ratio": lps / config.target_lps,
        "target_met": lps >= config.target_lps,
        "workers": worker_count,
        "chunk_lines": config.chunk_lines,
        "strategy": "indexed" if INDEX_TOKEN in config.template else "repeat",
        "shards": shards,
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _config_from_args(args: argparse.Namespace, *, chunk_lines: int | None = None) -> GenerationConfig:
    return GenerationConfig(
        lines=args.lines,
        target_lps=args.target_lps,
        chunk_lines=chunk_lines or args.chunk_lines,
        template=args.template,
        digest=not args.no_digest,
    )


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--lines", type=int, default=DEFAULT_LINES)
    parser.add_argument("--target-lps", type=float, default=TARGET_LINES_PER_SECOND)
    parser.add_argument("--chunk-lines", type=int, default=DEFAULT_CHUNK_LINES)
    parser.add_argument("--template", default=DEFAULT_TEMPLATE)
    parser.add_argument("--no-digest", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jarvisx-dr-moagi-codegen")
    sub = parser.add_subparsers(dest="command", required=True)

    benchmark_parser = sub.add_parser("benchmark", help="benchmark source emission")
    _add_common(benchmark_parser)
    benchmark_parser.add_argument("--require-target", action="store_true")

    generate_parser = sub.add_parser("generate", help="generate one source file")
    _add_common(generate_parser)
    generate_parser.add_argument("--output", type=Path, required=True)
    generate_parser.add_argument("--require-target", action="store_true")

    tune_parser = sub.add_parser("autotune", help="select the fastest chunk size")
    _add_common(tune_parser)

    shard_parser = sub.add_parser("shard", help="generate multiple source shards")
    _add_common(shard_parser)
    shard_parser.add_argument("--directory", type=Path, required=True)
    shard_parser.add_argument("--workers", type=int, default=None)
    shard_parser.add_argument("--require-target", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "benchmark":
        metrics = HighThroughputCodeGenerator(_config_from_args(args)).benchmark()
        print(metrics.to_json())
        return 2 if args.require_target and not metrics.target_met else 0

    if args.command == "generate":
        metrics = HighThroughputCodeGenerator(_config_from_args(args)).generate_to_file(args.output)
        print(metrics.to_json())
        return 2 if args.require_target and not metrics.target_met else 0

    if args.command == "autotune":
        selected, trials = autotune(
            lines=args.lines,
            target_lps=args.target_lps,
            template=args.template,
        )
        payload = {
            "selected_chunk_lines": selected.chunk_lines,
            "target_lps": selected.target_lps,
            "provenance_sha256": EXPECTED_SHA256,
            "trials": [asdict(item) for item in trials],
        }
        print(json.dumps(payload, sort_keys=True))
        return 0

    if args.command == "shard":
        manifest = generate_sharded(
            args.directory,
            _config_from_args(args),
            workers=args.workers,
        )
        print(json.dumps(manifest, sort_keys=True))
        return 2 if args.require_target and not bool(manifest["target_met"]) else 0

    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
