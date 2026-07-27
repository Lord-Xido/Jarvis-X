import pytest

from jarvisx.reflex import ReflexEngine
from jarvisx.registers import Registers


def test_reflex_is_non_mutating_by_default():
    registers = Registers()
    registers["Ψ"] = 10
    registers["Φ"] = 20

    correction = ReflexEngine().stabilize(registers)

    assert correction == 0
    assert registers["Φ"] == 20


def test_reflex_can_be_enabled_explicitly():
    registers = Registers()
    registers["Ψ"] = 10
    registers["Φ"] = 20

    correction = ReflexEngine(gain=0.1, enabled=True).stabilize(registers)

    assert correction == -1
    assert registers["Φ"] == 19


def test_reflex_rejects_negative_gain():
    with pytest.raises(ValueError):
        ReflexEngine(gain=-0.1)
