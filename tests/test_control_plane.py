from __future__ import annotations

from dataclasses import replace

import pytest

from jarvisx.assembler import Assembler
from jarvisx.control_plane import OmegaEvidenceChain, StateEnvelope, TransactionReceipt
from jarvisx.core import CodexVM
from jarvisx.dr_moagi_field_runtime import DrMoagiFieldConfig, DrMoagiFieldRuntime
from jarvisx.parser import Parser


def envelope(value: int, *, authoritative: bool) -> StateEnvelope:
    return StateEnvelope.from_payload(
        state_type="test-state",
        state_version=1,
        dimensions=(1,),
        payload={"value": value},
        authoritative=authoritative,
    )


def test_commit_receipt_is_hash_chained_and_deterministic() -> None:
    chain = OmegaEvidenceChain()
    before = envelope(1, authoritative=True)
    candidate = envelope(2, authoritative=False)
    after = envelope(2, authoritative=True)

    first = chain.append(
        subsystem="test",
        operation="advance",
        decision="commit",
        before=before,
        candidate=candidate,
        after=after,
        metrics={"error": 0.25},
    )
    second = chain.append(
        subsystem="test",
        operation="advance",
        decision="commit",
        before=after,
        candidate=envelope(3, authoritative=False),
        after=envelope(3, authoritative=True),
        metrics={"error": 0.125},
    )

    assert first.verify()
    assert second.previous_hash == first.receipt_hash
    assert chain.verify()


def test_rollback_receipt_requires_authoritative_state_preservation() -> None:
    chain = OmegaEvidenceChain()
    before = envelope(1, authoritative=True)
    candidate = envelope(2, authoritative=False)

    with pytest.raises(ValueError, match="rollback"):
        chain.append(
            subsystem="test",
            operation="advance",
            decision="rollback",
            before=before,
            candidate=candidate,
            after=envelope(3, authoritative=True),
        )

    receipt = chain.append(
        subsystem="test",
        operation="advance",
        decision="rollback",
        reason="validator rejected candidate",
        before=before,
        candidate=candidate,
        after=envelope(1, authoritative=True),
    )

    assert receipt.decision == "rollback"
    assert chain.verify()


def test_tampered_receipt_breaks_chain_verification() -> None:
    chain = OmegaEvidenceChain()
    receipt = chain.append(
        subsystem="test",
        operation="advance",
        decision="commit",
        before=envelope(1, authoritative=True),
        candidate=envelope(2, authoritative=False),
        after=envelope(2, authoritative=True),
    )
    chain.chain[0] = replace(receipt, receipt_hash="f" * 64)

    assert not chain.verify()


def test_transaction_receipt_rejects_mismatched_state_types() -> None:
    before = envelope(1, authoritative=True)
    candidate = StateEnvelope.from_payload(
        state_type="other-state",
        state_version=1,
        dimensions=(1,),
        payload={"value": 2},
        authoritative=False,
    )

    with pytest.raises(ValueError, match="state types"):
        TransactionReceipt.build(
            sequence=0,
            subsystem="test",
            operation="advance",
            decision="commit",
            reason=None,
            before=before,
            candidate=candidate,
            after=envelope(2, authoritative=True),
            metrics={},
            previous_hash="0" * 64,
        )


def test_codex_vm_emits_common_control_receipts() -> None:
    program = Assembler().assemble(Parser().parse("SET A 7\nHALT"))
    vm = CodexVM()
    vm.load(program)

    vm.run()

    assert len(vm.control_plane.chain) == 2
    assert vm.control_plane.verify()
    assert all(receipt.subsystem == "codex-vm" for receipt in vm.control_plane.chain)
    assert vm.control_plane.chain[-1].after.authoritative


def test_vm_control_receipt_is_rolled_back_with_failed_instruction_receipt(monkeypatch) -> None:
    program = Assembler().assemble(Parser().parse("SET A 7\nHALT"))
    vm = CodexVM()
    vm.load(program)

    def fail_log(state: object, opcode: int) -> None:
        raise OSError("receipt unavailable")

    monkeypatch.setattr(vm.ledger, "log", fail_log)

    with pytest.raises(OSError, match="receipt unavailable"):
        vm.step()

    assert not vm.control_plane.chain


def test_field_runtime_emits_commit_and_rollback_receipts() -> None:
    class ZeroCodec:
        def encode(self, field):
            return None

        def decode(self, latent, support):
            return {coordinate: 0.0 for coordinate in support}

    runtime = DrMoagiFieldRuntime(
        ZeroCodec(),
        DrMoagiFieldConfig(
            side=5,
            alpha=1.0,
            lambda_residual=0.0,
            eta=0.0,
            dt=0.1,
            expand_halo=False,
        ),
    )
    runtime.load({(2, 2, 2): 1.0})

    committed = runtime.step()
    rejected = runtime.step(validator=lambda candidate, metrics: False)

    assert committed.committed
    assert not rejected.committed
    assert [receipt.decision for receipt in runtime.control_plane.chain] == [
        "commit",
        "rollback",
    ]
    assert runtime.control_plane.verify()
    assert (
        runtime.control_plane.chain[-1].before.payload_digest
        == runtime.control_plane.chain[-1].after.payload_digest
    )
