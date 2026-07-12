from __future__ import annotations

import shlex
from dataclasses import dataclass

from .isa import Instruction, NumericFormat, Opcode, Phase


@dataclass(slots=True)
class AssembledProgram:
    instructions: list[Instruction]
    labels: dict[str, int]


class Assembler:
    """Assembler for a compact human-readable VANN bytecode syntax."""

    def __init__(self) -> None:
        self.default_format = NumericFormat.FP32

    def assemble(self, source: str) -> AssembledProgram:
        labels: dict[str, int] = {}
        parsed: list[tuple[int, list[str]]] = []

        for line_no, raw in enumerate(source.splitlines(), 1):
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            if line.endswith(":"):
                label = line[:-1].strip()
                if not label or label in labels:
                    raise SyntaxError(f"line {line_no}: invalid or duplicate label")
                labels[label] = len(parsed)
                continue
            parsed.append((line_no, shlex.split(line.replace(",", " "))))

        instructions: list[Instruction] = []
        for line_no, tokens in parsed:
            if not tokens:
                continue
            try:
                opcode = Opcode[tokens[0].upper()]
            except KeyError as exc:
                raise SyntaxError(f"line {line_no}: unknown opcode {tokens[0]!r}") from exc

            kwargs: dict[str, int | Opcode | NumericFormat | Phase] = {
                "opcode": opcode,
                "numeric_format": self.default_format,
                "phase": self._phase_for(opcode),
            }

            positional: list[str] = []
            for token in tokens[1:]:
                if "=" in token:
                    key, value = token.split("=", 1)
                    key = key.lower()
                    if key == "format":
                        kwargs["numeric_format"] = NumericFormat[value.upper()]
                    elif key == "phase":
                        kwargs["phase"] = Phase[value.upper()]
                    elif key in {"dst", "src_a", "src_b", "lambda_mask", "geo", "length", "immediate"}:
                        kwargs[key] = self._number_or_label(value, labels)
                    else:
                        raise SyntaxError(f"line {line_no}: unknown field {key}")
                else:
                    positional.append(token)

            if positional:
                if opcode == Opcode.JMP3D:
                    kwargs["immediate"] = self._number_or_label(positional[0], labels)
                else:
                    fields = ("dst", "src_a", "src_b", "immediate")
                    for field, value in zip(fields, positional, strict=False):
                        kwargs[field] = self._number_or_label(value, labels)

            instructions.append(Instruction(**kwargs))

        return AssembledProgram(instructions=instructions, labels=labels)

    @staticmethod
    def _number_or_label(value: str, labels: dict[str, int]) -> int:
        if value in labels:
            return labels[value]
        return int(value, 0)

    @staticmethod
    def _phase_for(opcode: Opcode) -> Phase:
        mapping = {
            Opcode.ENCODE3D: Phase.ENCODE,
            Opcode.PREDICT: Phase.PREDICT,
            Opcode.COMPARE: Phase.RESIDUAL,
            Opcode.UPDATE_OMEGA: Phase.UPDATE,
            Opcode.DECODE3D: Phase.DECODE,
            Opcode.COMMIT: Phase.COMMIT,
            Opcode.OPTIMIZE_POLICY: Phase.OPTIMIZE,
            Opcode.HALT: Phase.HALT,
        }
        return mapping.get(opcode, Phase.FETCH)
