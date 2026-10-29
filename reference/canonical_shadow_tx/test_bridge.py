"""Independently reproducible multi-surface candidate/commit acceptance tests."""
from __future__ import annotations

from dataclasses import replace
import json
import unittest

from reference.canonical_shadow_tx.bridge import (
    DOMAINS, MAX_FRAME_BYTES, PROFILES, DomainFrame, ShadowCoordinator,
    _checked, encoded, evidence_cycle, observe_ann, observe_geometry,
    observe_os, observe_vm, sha,
)


class TestShadowTxn(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(2)
        cls.frames = tuple(sorted((observe_ann(), observe_geometry(),
                                   observe_os(), observe_vm()),
                                  key=lambda f: f.domain))

    def setUp(self):
        self.coordinator = ShadowCoordinator()

    def test_all_four_real_backends_are_materialised(self):
        self.assertEqual(tuple(f.domain for f in self.frames), DOMAINS)
        self.assertTrue(all(len(f.payload) > 0 for f in self.frames))
        self.assertEqual(_checked(self.frames), self.frames)

    def test_stage_is_not_a_commit(self):
        proposal = self.coordinator.stage(self.frames)
        self.assertEqual(self.coordinator.epoch, 0)
        self.assertEqual(self.coordinator.snapshot(), ())
        self.assertEqual(self.coordinator.journal_head, "0" * 64)
        self.assertEqual(proposal.parent_hash, self.coordinator.state_hash())

    def test_requires_local_explicit_approval(self):
        proposal = self.coordinator.stage(self.frames)
        with self.assertRaises(PermissionError):
            self.coordinator.commit(proposal)
        self.assertEqual(self.coordinator.snapshot(), ())
        self.assertTrue(self.coordinator.verify())

    def test_commit_all_four_domains_and_verify_lineage(self):
        candidate = self.coordinator.stage(self.frames)
        receipt = self.coordinator.commit(candidate, approve=True)
        self.assertEqual(receipt.decision, "COMMITTED")
        self.assertEqual(receipt.epoch, 1)
        self.assertEqual(self.coordinator.snapshot(), self.frames)
        self.assertEqual(self.coordinator.state_hash(), candidate.candidate_hash)
        self.assertEqual(receipt.gates[-1], ("external_evidence", "NOT_APPLICABLE"))
        self.assertTrue(self.coordinator.verify())

    def test_replay_is_deterministic_across_coordinator_instances(self):
        other = ShadowCoordinator()
        left = self.coordinator.commit(self.coordinator.stage(self.frames), approve=True)
        right = other.commit(other.stage(self.frames), approve=True)
        self.assertEqual(left, right)
        self.assertEqual(self.coordinator.journal_head, other.journal_head)

    def test_second_commit_and_stale_rejection(self):
        stale = self.coordinator.stage(self.frames)
        first = self.coordinator.commit(stale, approve=True)
        self.assertEqual(first.epoch, 1)
        with self.assertRaisesRegex(ValueError, "stale"):
            self.coordinator.commit(stale, approve=True)
        next_receipt = self.coordinator.commit(
            self.coordinator.stage(self.frames), approve=True
        )
        self.assertEqual(next_receipt.epoch, 2)
        self.assertTrue(self.coordinator.verify())

    def test_single_domain_corruption_rolls_back_entire_owned_bundle(self):
        self.coordinator.commit(self.coordinator.stage(self.frames), approve=True)
        before = (self.coordinator.snapshot(), self.coordinator.epoch,
                  self.coordinator.state_hash(), self.coordinator.journal_head)
        candidate = self.coordinator.stage(self.frames)
        corrupt = bytearray(self.frames[1].payload)
        corrupt[-1] ^= 1
        forged = replace(candidate, frames=(
            candidate.frames[0], replace(candidate.frames[1], payload=bytes(corrupt)),
            candidate.frames[2], candidate.frames[3],
        ))
        with self.assertRaises(ValueError):
            self.coordinator.commit(forged, approve=True)
        self.assertEqual(before, (self.coordinator.snapshot(), self.coordinator.epoch,
                                  self.coordinator.state_hash(), self.coordinator.journal_head))
        self.assertTrue(self.coordinator.verify())

    def test_missing_or_duplicate_domain_is_rejected(self):
        for bad in (self.frames[:3], self.frames[:3] + (self.frames[0],),
                    self.frames[::-1]):
            with self.subTest(bad=tuple(f.domain for f in bad)):
                with self.assertRaises(ValueError):
                    self.coordinator.stage(bad)
        self.assertEqual(self.coordinator.epoch, 0)

    def test_wrong_profile_and_budget_are_rejected(self):
        ann = self.frames[0]
        for bad in (replace(ann, profile="unknown"),
                    replace(ann, payload=b"A" * (MAX_FRAME_BYTES + 1))):
            with self.assertRaises(ValueError):
                self.coordinator.stage((bad,) + self.frames[1:])

    def test_independent_arithmetic_oracle_catches_invalid_vm_data(self):
        vm = self.frames[-1]
        value = json.loads(vm.payload)
        value["registers"]["Ψ"] = 43
        with self.assertRaisesRegex(ValueError, "arithmetic oracle"):
            self.coordinator.stage(self.frames[:-1] + (
                replace(vm, payload=encoded(value)),))

    def test_ann_corruption_rejected_even_if_payload_is_recanonicalised(self):
        ann = self.frames[0]
        obj = json.loads(ann.payload)
        obj["residual_hex"] = "00" * 64
        with self.assertRaisesRegex(ValueError, "residual recovery"):
            self.coordinator.stage((replace(ann, payload=encoded(obj)),) + self.frames[1:])

    def test_os_transport_integrity_is_independently_checked(self):
        target = self.frames[2]
        obj = json.loads(target.payload)
        original = bytes.fromhex(obj["packet_hex"])
        obj["packet_hex"] = (original[:-1] + bytes([original[-1] ^ 1])).hex()
        with self.assertRaises(ValueError):
            self.coordinator.stage(
                self.frames[:2] + (replace(target, payload=encoded(obj)), self.frames[3])
            )

    def test_manifest_seal_and_hash_prevent_silent_candidate_replacement(self):
        p = self.coordinator.stage(self.frames)
        for changed in (replace(p, candidate_hash="f" * 64),
                        replace(p, seal="0" * 64),
                        replace(p, parent_hash="f" * 64)):
            with self.assertRaises(ValueError):
                self.coordinator.commit(changed, approve=True)
        self.assertEqual(self.coordinator.snapshot(), ())

    def test_full_reference_receipt(self):
        report = evidence_cycle()
        self.assertEqual(report["backend_domains"], list(DOMAINS))
        self.assertEqual(report["epoch"], 1)
        self.assertEqual(report["gates"]["external_evidence"], "NOT_APPLICABLE")
        self.assertEqual(report["receipt"], "verified")
        self.assertEqual(len(report["state_sha256"]), 64)
        self.assertNotEqual(report["state_sha256"], sha(b""))


if __name__ == "__main__":
    unittest.main()
