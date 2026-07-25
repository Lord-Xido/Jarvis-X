"""Polyglot front ends for the canonical Jarvis-X 64-bit bytecode ISA."""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .assembler import Assembler, OPCODES, REG_MAP
from .parser import Parser
from .rom import BytecodeAutodecoder, BytecodeAutoencoder, RomImage

REGISTER_BY_INDEX = {index: name for name, index in REG_MAP.items()}
OPCODE_BY_VALUE = {value: name for name, value in OPCODES.items()}

LANGUAGE_ALIASES = {
    "asm": "jarvis-asm",
    "assembly": "jarvis-asm",
    "jarvis": "jarvis-asm",
    "jarvis-asm": "jarvis-asm",
    "json": "json-ir",
    "json-ir": "json-ir",
    "python": "python",
    "py": "python",
}


class PolyglotCompileError(ValueError):
    """Raised when a source cannot be lowered to the Jarvis-X ISA."""


def _normalize_register(name: str) -> str:
    aliases = {
        "XI": "Ξ",
        "PSI": "Ψ",
        "PHI": "Φ",
        "LAMBDA": "Λ",
        "OMEGA": "Ω",
        "THETA": "Θ",
        "SIGMA": "𝒮",
        "PI": "Π",
    }
    normalized = aliases.get(name.upper(), name)
    if normalized not in REG_MAP:
        raise PolyglotCompileError("unknown register: %s" % name)
    return normalized


def _validate_ast(nodes: Sequence[Sequence[str]]) -> List[List[str]]:
    normalized: List[List[str]] = []
    for raw_node in nodes:
        node = [str(part) for part in raw_node]
        if not node:
            continue
        op = node[0].upper()
        if op == "SET" and len(node) == 3:
            register = _normalize_register(node[1])
            try:
                immediate = int(node[2], 0)
            except ValueError as exc:
                raise PolyglotCompileError("SET immediate must be an integer") from exc
            if not 0 <= immediate <= 0xFFFF:
                raise PolyglotCompileError("SET immediate must be between 0 and 65535")
            normalized.append([op, register, str(immediate)])
        elif op in ("ADD", "SUB") and len(node) == 4:
            normalized.append([op] + [_normalize_register(register) for register in node[1:4]])
        elif op == "HALT" and len(node) == 1:
            normalized.append([op])
        else:
            raise PolyglotCompileError("invalid instruction: %s" % " ".join(node))
    if not normalized or normalized[-1][0] != "HALT":
        normalized.append(["HALT"])
    return normalized


def _from_assembly(source: str) -> List[List[str]]:
    return _validate_ast(Parser().parse(source))


def _from_json(source: str) -> List[List[str]]:
    try:
        payload = json.loads(source)
    except json.JSONDecodeError as exc:
        raise PolyglotCompileError("invalid JSON IR") from exc
    if not isinstance(payload, list):
        raise PolyglotCompileError("JSON IR root must be an array")
    nodes: List[List[str]] = []
    for item in payload:
        if isinstance(item, list):
            nodes.append([str(part) for part in item])
        elif isinstance(item, dict):
            op = str(item.get("op", ""))
            args = item.get("args", [])
            if not isinstance(args, list):
                raise PolyglotCompileError("JSON IR args must be an array")
            nodes.append([op] + [str(part) for part in args])
        else:
            raise PolyglotCompileError("JSON IR instructions must be arrays or objects")
    return _validate_ast(nodes)


def _python_register(node: ast.AST) -> str:
    if not isinstance(node, ast.Name):
        raise PolyglotCompileError("Python operands must be register names")
    return _normalize_register(node.id)


def _from_python(source: str) -> List[List[str]]:
    try:
        module = ast.parse(source)
    except SyntaxError as exc:
        raise PolyglotCompileError("invalid Python source") from exc
    nodes: List[List[str]] = []
    for statement in module.body:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            destination = _python_register(statement.targets[0])
            value = statement.value
            if isinstance(value, ast.Constant) and isinstance(value.value, int):
                nodes.append(["SET", destination, str(value.value)])
            elif isinstance(value, ast.BinOp) and isinstance(value.op, (ast.Add, ast.Sub)):
                op = "ADD" if isinstance(value.op, ast.Add) else "SUB"
                nodes.append(
                    [op, destination, _python_register(value.left), _python_register(value.right)]
                )
            else:
                raise PolyglotCompileError(
                    "Python assignments support integer literals or register +/- register"
                )
        elif (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Call)
            and isinstance(statement.value.func, ast.Name)
            and statement.value.func.id.lower() == "halt"
            and not statement.value.args
            and not statement.value.keywords
        ):
            nodes.append(["HALT"])
        else:
            raise PolyglotCompileError(
                "unsupported Python statement: %s" % statement.__class__.__name__
            )
    return _validate_ast(nodes)


@dataclass(frozen=True)
class Compilation:
    language: str
    ast: Tuple[Tuple[str, ...], ...]
    words: Tuple[int, ...]
    rom: bytes


class PolyglotCompiler:
    """Lowers supported source languages into one deterministic ROM format."""

    def __init__(self) -> None:
        self._encoder = BytecodeAutoencoder()
        self._decoder = BytecodeAutodecoder()

    @property
    def languages(self) -> Tuple[str, ...]:
        return ("jarvis-asm", "json-ir", "python")

    def normalize_language(self, language: str) -> str:
        try:
            return LANGUAGE_ALIASES[language.strip().lower()]
        except KeyError as exc:
            raise PolyglotCompileError("unsupported language: %s" % language) from exc

    def parse(self, source: str, language: str) -> List[List[str]]:
        normalized = self.normalize_language(language)
        if normalized == "jarvis-asm":
            return _from_assembly(source)
        if normalized == "json-ir":
            return _from_json(source)
        if normalized == "python":
            return _from_python(source)
        raise AssertionError("unreachable language adapter")

    def compile(
        self,
        source: str,
        language: str,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Compilation:
        normalized = self.normalize_language(language)
        nodes = self.parse(source, normalized)
        words = tuple(Assembler().assemble(nodes))
        enriched: Dict[str, Any] = dict(metadata or {})
        enriched.update(
            {
                "compiler": "jarvisx-polyglot",
                "format": "JXROM/1",
                "instruction_count": len(words),
            }
        )
        rom = self._encoder.encode(normalized, source, words, enriched)
        decoded = self._decoder.decode(rom)
        if decoded.words != words:
            raise PolyglotCompileError("ROM verification failed after compilation")
        return Compilation(
            language=normalized,
            ast=tuple(tuple(node) for node in nodes),
            words=words,
            rom=rom,
        )

    def decode(self, rom: bytes) -> RomImage:
        return self._decoder.decode(rom)

    def decompile_words(self, words: Iterable[int]) -> str:
        lines: List[str] = []
        for word in words:
            opcode = (word >> 56) & 0xFF
            dst = (word >> 40) & 0xFF
            src1 = (word >> 32) & 0xFF
            src2 = (word >> 24) & 0xFF
            immediate = (word >> 8) & 0xFFFF
            op = OPCODE_BY_VALUE.get(opcode)
            if op == "SET":
                lines.append("SET %s %d" % (REGISTER_BY_INDEX[dst], immediate))
            elif op in ("ADD", "SUB"):
                lines.append(
                    "%s %s %s %s"
                    % (
                        op,
                        REGISTER_BY_INDEX[dst],
                        REGISTER_BY_INDEX[src1],
                        REGISTER_BY_INDEX[src2],
                    )
                )
            elif op == "HALT":
                lines.append("HALT")
            else:
                raise PolyglotCompileError("unknown opcode in ROM: 0x%02X" % opcode)
        return "\n".join(lines)

    def decode_to_assembly(self, rom: bytes) -> str:
        return self.decompile_words(self.decode(rom).words)
