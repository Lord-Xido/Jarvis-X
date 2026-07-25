import hashlib
import json

import pytest

from jarvisx.polyglot import PolyglotCompiler
from jarvisx.rom import RomFormatError
from jarvisx.rom_automation import PolyglotRomAutomation, SourceUnit


EXPECTED_ASSEMBLY = "SET A 7\nSET B 5\nADD C A B\nSUB D C B\nHALT"


@pytest.mark.parametrize(
    "language,source",
    [
        ("jarvis-asm", EXPECTED_ASSEMBLY),
        (
            "json-ir",
            json.dumps(
                [
                    {"op": "SET", "args": ["A", 7]},
                    ["SET", "B", 5],
                    ["ADD", "C", "A", "B"],
                    ["SUB", "D", "C", "B"],
                    ["HALT"],
                ]
            ),
        ),
        ("python", "A = 7\nB = 5\nC = A + B\nD = C - B\nhalt()\n"),
    ],
)
def test_polyglot_sources_compile_to_identical_words(language, source):
    compiler = PolyglotCompiler()
    compilation = compiler.compile(source, language)
    assert compiler.decode_to_assembly(compilation.rom) == EXPECTED_ASSEMBLY
    assert len(compilation.words) == 5


def test_rom_is_deterministic():
    compiler = PolyglotCompiler()
    first = compiler.compile(EXPECTED_ASSEMBLY, "asm", {"build": 1}).rom
    second = compiler.compile(EXPECTED_ASSEMBLY, "assembly", {"build": 1}).rom
    assert first == second
    assert hashlib.sha256(first).hexdigest() == hashlib.sha256(second).hexdigest()


def test_rom_detects_payload_tampering():
    compiler = PolyglotCompiler()
    rom = bytearray(compiler.compile(EXPECTED_ASSEMBLY, "asm").rom)
    rom[-1] ^= 0x01
    with pytest.raises(RomFormatError, match="digest mismatch"):
        compiler.decode(bytes(rom))


def test_automation_verifies_round_trip():
    result = PolyglotRomAutomation().run(
        [SourceUnit(name="python-example", language="python", source="A = 1\nB = 2\nC = A + B")]
    )[0]
    assert result.instruction_count == 4
    assert result.assembly.endswith("HALT")
    assert len(result.rom_sha256) == 64
