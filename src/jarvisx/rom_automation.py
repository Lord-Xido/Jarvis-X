"""Deterministic batch compilation and verification for polyglot ROM builds."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Tuple

from .polyglot import PolyglotCompiler


@dataclass(frozen=True)
class SourceUnit:
    name: str
    language: str
    source: str
    metadata: Optional[Mapping[str, Any]] = None


@dataclass(frozen=True)
class AutomationResult:
    name: str
    language: str
    instruction_count: int
    rom_sha256: str
    source_sha256: str
    rom: bytes
    assembly: str


class PolyglotRomAutomation:
    """Compile, decode, replay, and verify source units deterministically."""

    def __init__(self, compiler: Optional[PolyglotCompiler] = None) -> None:
        self.compiler = compiler or PolyglotCompiler()

    def run(self, units: Iterable[SourceUnit]) -> Tuple[AutomationResult, ...]:
        results = []
        for unit in units:
            compilation = self.compiler.compile(
                unit.source,
                unit.language,
                metadata=dict(unit.metadata or {}, unit=unit.name),
            )
            image = self.compiler.decode(compilation.rom)
            assembly = self.compiler.decompile_words(image.words)
            replay = self.compiler.compile(
                assembly,
                "jarvis-asm",
                metadata={"unit": unit.name, "replay_of": compilation.language},
            )
            if replay.words != compilation.words:
                raise RuntimeError("round-trip word verification failed for %s" % unit.name)
            results.append(
                AutomationResult(
                    name=unit.name,
                    language=compilation.language,
                    instruction_count=len(compilation.words),
                    rom_sha256=hashlib.sha256(compilation.rom).hexdigest(),
                    source_sha256=image.source_sha256,
                    rom=compilation.rom,
                    assembly=assembly,
                )
            )
        return tuple(results)
