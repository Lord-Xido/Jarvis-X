from jarvisx.reflex import ReflexEngine
from jarvisx.registers import Registers


def test_reflex_can_observe_without_mutating_registers():
    regs = Registers()
    regs["Ψ"] = 10
    regs["Φ"] = 20
    reflex = ReflexEngine()

    delta = reflex.stabilize(regs, apply=False)

    assert delta == -1
    assert reflex.last_delta == -1
    assert regs["Φ"] == 20


def test_reflex_mutation_requires_explicit_active_mode():
    regs = Registers()
    regs["Ψ"] = 10
    regs["Φ"] = 20
    reflex = ReflexEngine()

    reflex.stabilize(regs, apply=True)

    assert regs["Φ"] == 19
