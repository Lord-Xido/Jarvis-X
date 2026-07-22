import json

from jarvisx.ledger import OmegaLedger
from jarvisx.reflex import REFLEX_ENABLE_FLAG, ReflexEngine
from jarvisx.registers import Registers


def test_omega_ledger_entries_are_json_serializable():
    ledger = OmegaLedger()
    ledger.log({"A": 1}, 0x01)
    assert isinstance(ledger.chain[0]["payload"], str)
    json.dumps(ledger.chain)


def test_reflex_mutation_requires_explicit_flag():
    regs = Registers()
    regs["Ψ"] = 10
    regs["Φ"] = 20
    reflex = ReflexEngine()

    reflex.stabilize(regs)
    assert regs["Φ"] == 20

    regs["FLAGS"] = REFLEX_ENABLE_FLAG
    reflex.stabilize(regs)
    assert regs["Φ"] == 19
