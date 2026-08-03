from __future__ import annotations

from collections.abc import Mapping

REGISTER_NAMES = (
    "Ξ",
    "Ψ",
    "Φ",
    "Λ",
    "Ω",
    "Θ",
    "𝒮",
    "Π",
    "A",
    "B",
    "C",
    "D",
    "IP",
    "SP",
    "FLAGS",
    "TMP",
)


class Registers:
    def __init__(self) -> None:
        self._regs = {name: 0 for name in REGISTER_NAMES}

    def __getitem__(self, key: str) -> int:
        return self._regs[key]

    def __setitem__(self, key: str, value: int) -> None:
        self._regs[key] = int(value)

    def snapshot(self) -> dict[str, int]:
        return dict(self._regs)

    def restore(self, snapshot: Mapping[str, int]) -> None:
        if set(snapshot) != set(REGISTER_NAMES):
            raise ValueError("register snapshot schema mismatch")
        self._regs = {name: int(snapshot[name]) for name in REGISTER_NAMES}

    def reset(self) -> None:
        self._regs = {name: 0 for name in REGISTER_NAMES}
