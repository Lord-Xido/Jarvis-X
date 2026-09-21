"""Symbolic parameter-space contracts for the HyperCloud runtime."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SymbolicParameterExtent:
    """Describe a logical parameter count without expanding it into an integer.

    The default represents::

        1_000_000 ** (1_000_000 ** 1_000_000)

    The value is intentionally symbolic. Constructing the corresponding dense
    array, enumerating every address, or claiming that many physical weights is
    outside the runtime contract.
    """

    base: int = 1_000_000
    exponent_base: int = 1_000_000
    exponent_exponent: int = 1_000_000

    def __post_init__(self) -> None:
        if self.base < 2 or self.exponent_base < 2 or self.exponent_exponent < 1:
            raise ValueError("symbolic extent terms must be positive and non-trivial")

    @property
    def expression(self) -> str:
        return f"{self.base}^({self.exponent_base}^{self.exponent_exponent})"

    @property
    def address_radix(self) -> int:
        return self.base

    def metadata(self) -> dict[str, int | str]:
        return {
            "expression": self.expression,
            "base": self.base,
            "exponent_base": self.exponent_base,
            "exponent_exponent": self.exponent_exponent,
            "allocation_semantics": "symbolic-sparse",
        }


@dataclass(frozen=True)
class HierarchicalAddress:
    """Finite sparse path into the symbolic virtual parameter space.

    A client transmits only the path it actually touches. Each digit lies in
    ``[0, radix)``. The runtime therefore never needs to materialize the full
    logical index width implied by the symbolic extent.
    """

    digits: tuple[int, ...]

    def validate(self, *, radix: int) -> None:
        if radix < 2:
            raise ValueError("radix must be at least 2")
        if not self.digits:
            raise ValueError("address must contain at least one digit")
        for digit in self.digits:
            if digit < 0 or digit >= radix:
                raise ValueError(f"address digit {digit} is outside radix {radix}")

    def canonical(self) -> str:
        return ".".join(str(digit) for digit in self.digits)
