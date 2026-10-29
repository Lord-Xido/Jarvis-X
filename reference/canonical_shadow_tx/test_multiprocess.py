"""Acceptance tests for independently persisted participants and coordinator replay.

All authority is SHADOW ONLY. Test subprocesses kill the coordinator driver,
not an existing live backend. Tests must not be interpreted as distributed ACID.
"""
from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest

from reference.canonical_shadow_tx.bridge import (
    DOMAINS, observe_ann, observe_geometry, observe_os, observe_vm, _state_hash,
)
from reference.canonical_shadow_tx.multiprocess import Coordinator, Participant


class TestSeparateParticipants(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(2)
        cls.frames = tuple(sorted(
            (observe_ann(), observe_geometry(), observe_os(), observe_vm()),
            key=lambda f: f.domain,
        ))

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "participants"

    def _prepare(self, txid="t1"):
        with Coordinator(self.root) as coordinator:
            return coordinator.prepare(txid, self.frames)

    def _kill(self, txid: str, phase: str, point: str):
        script = """
import os,sys
from reference.canonical_shadow_tx.multiprocess import Coordinator
c = Coordinator(sys.argv[1])
def stop(point):
    if point == sys.argv[3]:
        os._exit(73)
if sys.argv[2] == 'approve':
    c.approve(sys.argv[4], approve=True, fault=stop)
else:
    raise AssertionError('unsupported subprocess phase')
"""
        result = subprocess.run(
            [sys.executable, "-c", script, str(self.root), phase, point, txid],
            check=False, capture_output=True, text=True, timeout=20,
        )
        self.assertEqual(result.returncode, 73, result.stderr)

    def test_four_files_are_independently_persistent(self):
        self._prepare()
        for domain, frame in zip(DOMAINS, self.frames):
            with self.subTest(domain=domain), Participant(self.root, domain, create=False) as part:
                self.assertEqual(part.status("t1"), "PREPARED")
                self.assertEqual(part.state()[0], 0)
                self.assertIsNone(part.state()[2])
                self.assertTrue(part.verify())
        self.assertEqual(
            {path.name for path in self.root.glob("*.sqlite3")},
            {"coordinator.sqlite3"} | {domain + ".sqlite3" for domain in DOMAINS},
        )

    def test_prepared_never_auto_applied_on_recovery(self):
        self._prepare()
        with Coordinator(self.root) as coordinator:
            self.assertEqual(coordinator.recover(), {"t1": "ABORTED"})
            self.assertEqual(coordinator.decision_status("t1"), "ABORTED")
            self.assertEqual(coordinator._meta()[0], 0)
            with self.assertRaises(RuntimeError):
                coordinator.approve("t1", approve=True)
        for domain in DOMAINS:
            with Participant(self.root, domain, create=False) as part:
                self.assertIsNone(part.state()[2])
                self.assertEqual(part.status("t1"), "ABORTED")

    def test_approval_is_required(self):
        self._prepare()
        with Coordinator(self.root) as coordinator:
            with self.assertRaises(PermissionError):
                coordinator.approve("t1")
            self.assertEqual(coordinator.decision_status("t1"), "PREPARED")

    def test_all_four_shadow_participants_commit_and_read_gate(self):
        expected_hash = self._prepare()
        with Coordinator(self.root) as coordinator:
            with self.assertRaises(RuntimeError):
                coordinator.coordinated_read("t1")
            result = coordinator.approve("t1", approve=True)
            self.assertEqual(result.status, "COMPLETE")
            self.assertEqual(result.state_hash, expected_hash)
            self.assertEqual(coordinator.coordinated_read("t1"), self.frames)
            self.assertEqual(coordinator.receipt("t1")["acked_domains"], list(DOMAINS))
            self.assertEqual(coordinator.receipt("t1")["complete"]["epoch"], 1)
        with Coordinator(self.root) as reopened:
            self.assertEqual(reopened.coordinated_read("t1"), self.frames)
            self.assertEqual(reopened.approve("t1", approve=True), result)
        for domain in DOMAINS:
            with Participant(self.root, domain) as part:
                self.assertEqual(part.state()[0], 1)
                self.assertEqual(part.status("t1"), "COMMITTED")

    def test_subprocess_crash_after_decision_and_every_applied_domain(self):
        points = ["after_decision"] + ["after_apply_" + d for d in DOMAINS] + ["after_complete"]
        for index, point in enumerate(points):
            with self.subTest(point=point):
                root = Path(self.temp.name) / ("crash-" + str(index))
                with Coordinator(root) as coordinator:
                    coordinator.prepare("crash", self.frames)
                self.root = root
                self._kill("crash", "approve", point)
                with Coordinator(root) as reopened:
                    if point == "after_complete":
                        self.assertEqual(reopened.decision_status("crash"), "COMPLETE")
                        self.assertEqual(reopened.recover(), {})
                    else:
                        self.assertEqual(reopened.decision_status("crash"), "COMMIT_DECIDED")
                        with self.assertRaises(RuntimeError):
                            reopened.coordinated_read("crash")
                        self.assertEqual(reopened.recover(), {"crash": "COMPLETE"})
                    self.assertEqual(reopened.coordinated_read("crash"), self.frames)
                    self.assertEqual(reopened._meta()[0], 1)
                    self.assertEqual(reopened.receipt("crash")["acked_domains"], list(DOMAINS))

    def test_crash_before_durable_decision_aborts_all(self):
        self._prepare()
        self._kill("t1", "approve", "before_decision")
        with Coordinator(self.root) as coordinator:
            self.assertEqual(coordinator.decision_status("t1"), "PREPARED")
            self.assertEqual(coordinator.recover(), {"t1": "ABORTED"})
            self.assertEqual(coordinator._meta()[0], 0)
        for domain in DOMAINS:
            with Participant(self.root, domain) as participant:
                self.assertIsNone(participant.state()[2])

    def test_preparing_crash_after_partial_participant_persist(self):
        for cut in DOMAINS:
            with self.subTest(point=cut):
                root = Path(self.temp.name) / ("prepare-" + cut)
                with Coordinator(root) as coordinator:
                    def crash_when(point):
                        if point == "after_prepare_" + cut:
                            raise OSError("injected prepare interruption")
                    with self.assertRaises(OSError):
                        coordinator.prepare("abort-me", self.frames, fault=crash_when)
                    self.assertEqual(coordinator.decision_status("abort-me"), "PREPARING")
                with Coordinator(root) as reopened:
                    self.assertEqual(reopened.recover(), {"abort-me": "ABORTED"})
                    self.assertEqual(reopened._meta()[0], 0)
                for domain in DOMAINS:
                    path = root / (domain + ".sqlite3")
                    if path.exists():
                        with Participant(root, domain) as participant:
                            self.assertNotEqual(participant.status("abort-me"), "COMMITTED")

    def test_optimistic_conflict_rejects_second_unresolved_transaction(self):
        self._prepare()
        with Coordinator(self.root) as coordinator:
            with self.assertRaisesRegex(RuntimeError, "unresolved"):
                coordinator.prepare("t2", self.frames)
            coordinator.approve("t1", approve=True)
            new = coordinator.prepare("t2", self.frames)
            self.assertEqual(new, _state_hash(self.frames))
            complete = coordinator.approve("t2", approve=True)
            self.assertEqual(complete.epoch, 2)
            with self.assertRaises(ValueError):
                coordinator.prepare("t2", self.frames)
            self.assertEqual(coordinator.coordinated_read("t2"), self.frames)

    def test_stale_fence_cannot_apply_unknown_prepared_tx(self):
        self._prepare()
        with Participant(self.root, "ann", create=False) as ann:
            with self.assertRaises(RuntimeError):
                ann.commit("t1", -1)
            with self.assertRaises(RuntimeError):
                ann.commit("unapproved", 1)
        with Coordinator(self.root) as coordinator:
            self.assertEqual(coordinator.decision_status("t1"), "PREPARED")
            self.assertEqual(coordinator.recover(), {"t1": "ABORTED"})

    def test_reject_corrupted_participant_before_durable_decision(self):
        self._prepare()
        with Participant(self.root, "os") as participant:
            participant.db.execute(
                "UPDATE prepared SET payload=? WHERE txid=?", (b"corrupt", "t1")
            )
        with Coordinator(self.root) as coordinator:
            with self.assertRaises(RuntimeError):
                coordinator.approve("t1", approve=True)
            self.assertEqual(coordinator.decision_status("t1"), "PREPARED")
            self.assertEqual(coordinator.recover(), {"t1": "ABORTED"})

    def test_raw_participant_reads_are_not_atomically_visible(self):
        self._prepare()
        with Coordinator(self.root) as coordinator:
            def halt_after_first(point):
                if point == "after_apply_ann":
                    raise OSError("interrupted after first participant commit")
            with self.assertRaises(OSError):
                coordinator.approve("t1", approve=True, fault=halt_after_first)
            with Participant(self.root, "ann") as ann, Participant(self.root, "geometry") as geo:
                self.assertIsNotNone(ann.state()[2])
                self.assertIsNone(geo.state()[2])
            with self.assertRaises(RuntimeError):
                coordinator.coordinated_read("t1")
            self.assertEqual(coordinator.recover(), {"t1": "COMPLETE"})
            self.assertEqual(coordinator.coordinated_read("t1"), self.frames)

    def test_missing_participant_during_recovery_fails_closed(self):
        self._prepare()
        with Coordinator(self.root) as coordinator:
            def stop_after_decision(point):
                if point == "after_decision":
                    raise RuntimeError("stopped intentionally")
            with self.assertRaises(RuntimeError):
                coordinator.approve("t1", approve=True, fault=stop_after_decision)
        path = self.root / "os.sqlite3"
        for suffix in ("", "-wal", "-shm"):
            file = Path(str(path) + suffix)
            if file.exists():
                file.unlink()
        with Coordinator(self.root) as coordinator:
            with self.assertRaises(RuntimeError):
                coordinator.recover()
            self.assertEqual(coordinator.decision_status("t1"), "COMMIT_DECIDED")

    def test_full_receipt_is_bounded_and_discloses_visibility_limit(self):
        self._prepare()
        with Coordinator(self.root) as coordinator:
            result = coordinator.approve("t1", approve=True).as_dict()
            self.assertEqual(result["external_evidence"], "NOT_APPLICABLE")
            self.assertIn("raw participant reads can be partial", result["visibility"])
            self.assertEqual(result["domains"], list(DOMAINS))
            self.assertEqual(result["authority"],
                             "four independent shadow SQLite files + decision SQLite")


if __name__ == "__main__":
    unittest.main()
