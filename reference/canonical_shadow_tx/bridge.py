"""Canonical *shadow-only* multi-surface transaction reference (ADR-016).

Real CodexVM, Dr Moagi OS, trained 3D ANN and ENV1 outputs are acquired from
fresh, isolated instances. Only this coordinator's in-memory byte snapshots
are atomically promoted. NO existing backend state, checkpoint, service,
trained weight or host resource is replaced by a bridge commit.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import threading
from typing import Iterable

DOMAINS = ("ann", "geometry", "os", "vm")
PROFILES = {
    "ann": "Conv3D-24/XOR/v1",
    "geometry": "DM3D-ENV1/U8",
    "os": "DMOS2/sparse-f64",
    "vm": "CodexVM-64/v1",
}
MAX_FRAME_BYTES = 262_228
MAX_BUNDLE_BYTES = 524_288
ROOT = Path(__file__).resolve().parents[2]


def encoded(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class DomainFrame:
    domain: str
    profile: str
    payload: bytes


@dataclass(frozen=True)
class Proposal:
    epoch: int
    parent_hash: str
    frames: tuple[DomainFrame, ...]
    candidate_hash: str
    seal: str


@dataclass(frozen=True)
class CommitReceipt:
    epoch: int
    parent_hash: str
    candidate_hash: str
    journal_parent: str
    journal_head: str
    decision: str
    gates: tuple[tuple[str, str], ...]


def _payload(frame: DomainFrame) -> dict:
    try:
        value = json.loads(frame.payload)
    except (ValueError, UnicodeDecodeError, TypeError) as exc:
        raise ValueError("invalid canonical JSON") from exc
    if not isinstance(value, dict) or encoded(value) != frame.payload:
        raise ValueError("noncanonical JSON")
    return value


def _verify_frame(frame: DomainFrame) -> None:
    """Independent structural/reconstruction checks; not external-world proof."""
    if type(frame) is not DomainFrame or frame.domain not in PROFILES:
        raise ValueError("unsupported domain")
    if frame.profile != PROFILES[frame.domain]:
        raise ValueError("unsupported geometry or backend profile")
    if type(frame.payload) is not bytes or not 1 <= len(frame.payload) <= MAX_FRAME_BYTES:
        raise ValueError("frame resource budget exceeded")

    if frame.domain == "geometry":
        from reference.dm3d_env1.envelope import Geometry, from_morton, to_morton, unpack
        shape, raw = unpack(frame.payload)
        if shape != Geometry(4) or len(raw) != 64:
            raise ValueError("unexpected geometric profile")
        if from_morton(to_morton(raw, shape), shape) != raw:
            raise ValueError("Morton roundtrip failed")
        return

    data = _payload(frame)
    if frame.domain == "vm":
        registers = data.get("registers")
        if not isinstance(registers, dict):
            raise ValueError("VM registers missing")
        if any(type(v) is not int for v in registers.values()):
            raise ValueError("VM non-integer register")
        if (registers.get("A"), registers.get("B"), registers.get("Ψ")) != (19, 23, 42):
            raise ValueError("independent VM arithmetic oracle rejected")
    elif frame.domain == "os":
        from jarvisx.dr_moagi_os_store import SparseStateCodec3D, SparseStatePacket3D
        rows = data.get("cells")
        if not isinstance(rows, list) or not 1 <= len(rows) <= 512:
            raise ValueError("OS cell budget or shape failed")
        if data.get("cycle") != 1 or data.get("journal_valid") is not True:
            raise ValueError("OS reference cycle or journal not verified")
        if not isinstance(data.get("journal_head"), str) or len(data["journal_head"]) != 64:
            raise ValueError("OS journal head missing")
        packet = SparseStatePacket3D(
            side=8, active_cells=len(rows),
            payload=bytes.fromhex(data["packet_hex"]),
            checksum_sha256=data["packet_sha256"],
        )
        reconstructed = SparseStateCodec3D().decode(packet)
        expected = {}
        for cell in rows:
            if not isinstance(cell, dict) or set(cell) != {"x", "y", "z", "value"}:
                raise ValueError("OS coordinate schema invalid")
            key = (cell["x"], cell["y"], cell["z"])
            if key in expected or any(type(axis) is not int or not 0 <= axis < 8 for axis in key):
                raise ValueError("OS duplicate or out-of-bounds coordinate")
            expected[key] = float(cell["value"])
        if reconstructed != expected:
            raise ValueError("OS independent DMOS2 reconstruction mismatch")
    elif frame.domain == "ann":
        try:
            source = bytes.fromhex(data["input_hex"])
            decoded = bytes.fromhex(data["decoded_hex"])
            residual = bytes.fromhex(data["residual_hex"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("ANN residual representation malformed") from exc
        if len(source) != 64 or len(decoded) != 64 or len(residual) != 64:
            raise ValueError("ANN token lattice must contain exactly 64 bytes")
        if bytes(d ^ r for d, r in zip(decoded, residual)) != source:
            raise ValueError("ANN exact residual recovery failed")
        if data.get("reference_match") is not True or data.get("finite_latent") is not True:
            raise ValueError("ANN original reference/finite gate failed")
        if any(v > 26 for v in source + decoded):
            raise ValueError("ANN vocabulary boundary exceeded")


def _checked(frames: Iterable[DomainFrame]) -> tuple[DomainFrame, ...]:
    result = tuple(frames)
    if len(result) != len(DOMAINS) or tuple(f.domain for f in result) != DOMAINS:
        raise ValueError("exactly four sorted unique domains are mandatory")
    if sum(len(f.payload) for f in result) > MAX_BUNDLE_BYTES:
        raise ValueError("aggregate memory budget exceeded")
    for frame in result:
        _verify_frame(frame)
    return result


def _state_hash(frames: tuple[DomainFrame, ...]) -> str:
    return sha(encoded([(f.domain, f.profile, sha(f.payload)) for f in frames]))


def _seal(epoch: int, parent: str, candidate: str) -> str:
    return sha(encoded({"schema": "CanonicalShadowTx/1", "epoch": epoch,
                        "parent": parent, "candidate": candidate}))


class ShadowCoordinator:
    """Atomic commit of this object's owned snapshots; not cross-service ACID."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._frames: tuple[DomainFrame, ...] = ()
        self.epoch = 0
        self.journal_head = "0" * 64
        self._history: tuple[CommitReceipt, ...] = ()

    def snapshot(self) -> tuple[DomainFrame, ...]:
        with self._lock:
            return self._frames

    def state_hash(self) -> str:
        with self._lock:
            return _state_hash(self._frames)

    def stage(self, frames: Iterable[DomainFrame]) -> Proposal:
        with self._lock:
            checked = _checked(frames)
            target = _state_hash(checked)
            parent = _state_hash(self._frames)
            return Proposal(self.epoch, parent, checked, target,
                            _seal(self.epoch, parent, target))

    def commit(self, proposal: Proposal, *, approve: bool = False) -> CommitReceipt:
        with self._lock:
            if approve is not True:
                raise PermissionError("explicit local approval required")
            if type(proposal) is not Proposal:
                raise ValueError("invalid candidate type")
            if proposal.epoch != self.epoch or proposal.parent_hash != _state_hash(self._frames):
                raise ValueError("stale candidate")
            verified = _checked(proposal.frames)
            actual = _state_hash(verified)
            if actual != proposal.candidate_hash or proposal.seal != _seal(
                proposal.epoch, proposal.parent_hash, actual
            ):
                raise ValueError("candidate identity or seal mismatch")
            parent_journal = self.journal_head
            next_epoch = self.epoch + 1
            next_journal = sha(
                bytes.fromhex(parent_journal)
                + encoded({"epoch": next_epoch, "before": proposal.parent_hash,
                           "after": actual, "decision": "COMMITTED"})
            )
            receipt = CommitReceipt(
                next_epoch, proposal.parent_hash, actual,
                parent_journal, next_journal, "COMMITTED",
                tuple((name, "PASS") for name in
                      ("type", "geometry", "resource", "reconstruction", "internal"))
                + (("external_evidence", "NOT_APPLICABLE"),),
            )
            # One assignment boundary for this owned in-memory authority namespace.
            self._frames, self.epoch, self.journal_head, self._history = (
                verified, next_epoch, next_journal, self._history + (receipt,)
            )
            return receipt

    def verify(self) -> bool:
        with self._lock:
            try:
                if self._frames:
                    _checked(self._frames)
                head = "0" * 64
                before = _state_hash(())
                for index, receipt in enumerate(self._history, start=1):
                    if (receipt.epoch != index or receipt.journal_parent != head
                            or receipt.parent_hash != before
                            or receipt.decision != "COMMITTED"):
                        return False
                    head = sha(bytes.fromhex(head) + encoded({
                        "epoch": index, "before": before,
                        "after": receipt.candidate_hash, "decision": "COMMITTED",
                    }))
                    if receipt.journal_head != head:
                        return False
                    before = receipt.candidate_hash
                return (head == self.journal_head and len(self._history) == self.epoch
                        and before == _state_hash(self._frames))
            except (ValueError, TypeError, KeyError):
                return False


def observe_vm() -> DomainFrame:
    """Use a fresh, in-memory CodexVM; no live VM state is touched."""
    from jarvisx.assembler import Assembler
    from jarvisx.core import CodexVM
    words = Assembler().assemble((
        ("SET", "A", "19"), ("SET", "B", "23"),
        ("ADD", "Ψ", "A", "B"), ("HALT",),
    ))
    vm = CodexVM(max_cycles=16)
    vm.load(words)
    registers = vm.run()
    if not vm.ledger.verify() or vm.cycles != 4:
        raise ArithmeticError("isolated VM ledger/cycle verification failed")
    return DomainFrame("vm", PROFILES["vm"], encoded({"registers": registers}))


def observe_os() -> DomainFrame:
    """Run a fresh OS kernel with persistence and networking disabled."""
    from jarvisx.dr_moagi_os import DrMoagiOSConfig, DrMoagiOSKernel, demo_field
    os = DrMoagiOSKernel(DrMoagiOSConfig(
        side=8, max_active_cells=512, deep_distiller_max_latent_cells=256,
        fixed_point_passes=1, state_dir=None,
    ))
    os.boot(restore=False)
    os.load(demo_field(8))
    report = os.step()
    if not report.committed:
        raise ArithmeticError(report.rejection_reason or "OS shadow cycle failed")
    snap, status, packet = os.snapshot(limit=512), os.status(), os.export_state_packet()
    if snap["truncated"] or not status["journal_valid"]:
        raise ArithmeticError("OS shadow snapshot incomplete or journal invalid")
    return DomainFrame("os", PROFILES["os"], encoded({
        "cells": snap["cells"], "cycle": snap["cycle"],
        "journal_valid": status["journal_valid"],
        "journal_head": status["journal_head"],
        "packet_hex": packet.payload.hex(),
        "packet_sha256": packet.checksum_sha256,
    }))


def observe_ann() -> DomainFrame:
    """Execute published, trained weights locally; no training or parameter writes."""
    import torch
    path = ROOT / "apps" / "recursive-3d-ann" / "runtime.py"
    spec = importlib.util.spec_from_file_location("canonical_shadow_ann", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("ANN reference not available")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _, model, vocabulary, reference = module.load_models(path.with_name("index.html"))
    source = reference["selected_streams"][0]["token_ids"]
    if len(source) != 64 or len(vocabulary) != 27:
        raise ValueError("wrong ANN reference dimensions")
    with torch.no_grad():
        logits, latent = model(torch.tensor([source], dtype=torch.long))
    decoded = logits.argmax(-1)[0].tolist()
    if decoded != reference["volume_example"]["decoded_ids"]:
        raise ArithmeticError("trained ANN reference output changed")
    if not bool(torch.isfinite(latent).all()):
        raise ArithmeticError("ANN latent state nonfinite")
    original_bytes, decoded_bytes = bytes(source), bytes(decoded)
    residual = bytes(a ^ b for a, b in zip(original_bytes, decoded_bytes))
    return DomainFrame("ann", PROFILES["ann"], encoded({
        "input_hex": original_bytes.hex(),
        "decoded_hex": decoded_bytes.hex(),
        "residual_hex": residual.hex(),
        "finite_latent": True, "reference_match": True,
    }))


def observe_geometry() -> DomainFrame:
    from reference.dm3d_env1.envelope import Geometry, pack
    raw = bytes((i * 19 + 7) % 256 for i in range(64))
    return DomainFrame("geometry", PROFILES["geometry"], pack(raw, Geometry(4)))


def evidence_cycle() -> dict:
    """Four real isolated backends -> one provisional bundle -> verified commit."""
    frames = tuple(sorted((observe_vm(), observe_os(), observe_ann(),
                           observe_geometry()), key=lambda x: x.domain))
    coordinator = ShadowCoordinator()
    stage = coordinator.stage(frames)
    if coordinator.epoch != 0 or coordinator.snapshot():
        raise AssertionError("stage mutated authoritative shadow state")
    receipt = coordinator.commit(stage, approve=True)
    if not coordinator.verify():
        raise AssertionError("cross-surface in-memory journal invalid")
    return {
        "authority": "shadow-only/in-memory",
        "schema": "CanonicalShadowTx/1",
        "backend_domains": [f.domain for f in frames],
        "backend_sha256": {f.domain: sha(f.payload) for f in frames},
        "epoch": receipt.epoch,
        "state_sha256": receipt.candidate_hash,
        "journal_head": receipt.journal_head,
        "gates": dict(receipt.gates),
        "receipt": "verified",
        "limit": "No live VM/OS/ANN or cross-process atomic commit",
    }


if __name__ == "__main__":
    print(json.dumps(evidence_cycle(), indent=2, sort_keys=True))
