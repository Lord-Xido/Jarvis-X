"""Crash/restart acceptance tests for the single-database durable shadow reference."""
from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest

from reference.canonical_shadow_tx.bridge import (
    DOMAINS, DomainFrame, _state_hash, encoded, observe_ann,
    observe_geometry, observe_os, observe_vm,
)
from reference.canonical_shadow_tx.durable import (
    DurableShadowStore, MAX_PENDING, recovery_receipt,
)


class TestDurableShadow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(2)
        cls.frames = tuple(sorted(
            (observe_ann(), observe_geometry(), observe_os(), observe_vm()),
            key=lambda frame: frame.domain,
        ))

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "state.sqlite3"

    def store(self):
        return DurableShadowStore(self.path)

    def _prepared(self, txid="test-1"):
        with self.store() as store:
            candidate = store.prepare(txid, self.frames)
            self.assertEqual(store.snapshot(), ())
            self.assertEqual(store.epoch(), 0)
            return candidate

    def test_real_4_domain_snapshots_are_valid(self):
        self.assertEqual(tuple(frame.domain for frame in self.frames), DOMAINS)
        with self.store() as store:
            self.assertTrue(store.verify())
            self.assertEqual(store.epoch(), 0)
            self.assertEqual(store.snapshot(), ())

    def test_prepare_survives_process_reopen_without_commit(self):
        candidate = self._prepared()
        with self.store() as store:
            self.assertEqual(store.pending(), ("test-1",))
            self.assertEqual(store.epoch(), 0)
            self.assertEqual(store.snapshot(), ())
            self.assertEqual(store.prepare("test-1", self.frames), candidate)

    def test_approval_required_and_committed_state_survives_reopen(self):
        self._prepared()
        with self.store() as store:
            with self.assertRaises(PermissionError):
                store.commit("test-1")
            receipt = store.commit("test-1", approve=True)
            self.assertEqual(receipt.epoch, 1)
            self.assertEqual(receipt.decision, "COMMITTED")
            self.assertEqual(receipt.external_evidence, "NOT_APPLICABLE")
            self.assertEqual(store.snapshot(), self.frames)
            self.assertEqual(store.pending(), ())
            self.assertTrue(store.verify())
        with self.store() as reopened:
            self.assertEqual(reopened.epoch(), 1)
            self.assertEqual(reopened.snapshot(), self.frames)
            self.assertTrue(reopened.verify())
            again = reopened.commit("test-1", approve=True)
            self.assertEqual(again.as_dict(), receipt.as_dict())
        evidence = recovery_receipt(self.path, "test-1")
        self.assertEqual(evidence["epoch"], 1)
        self.assertTrue(evidence["verified"])

    def test_failed_candidate_rejected_before_prepare(self):
        invalid = replace(self.frames[1], profile="invalid-geometry")
        with self.store() as store:
            with self.assertRaises(ValueError):
                store.prepare("reject", (
                    self.frames[0], invalid, self.frames[2], self.frames[3]
                ))
            self.assertEqual(store.pending(), ())
            self.assertEqual(store.epoch(), 0)

    def test_id_collision_is_fail_closed(self):
        self._prepared()
        changed = replace(self.frames[0], profile="wrong-profile")
        with self.store() as store:
            with self.assertRaises(ValueError):
                store.prepare("test-1", (
                    changed, self.frames[1], self.frames[2], self.frames[3]
                ))
            self.assertEqual(store.pending(), ("test-1",))

    def test_aborted_candidate_survives_recovery(self):
        self._prepared()
        with self.store() as store:
            store.abort("test-1")
            store.abort("test-1")
            self.assertEqual(store.pending(), ())
            with self.assertRaises(ValueError):
                store.commit("test-1", approve=True)
        with self.store() as store:
            self.assertEqual(store.snapshot(), ())
            self.assertTrue(store.verify())

    def test_prepared_transaction_becomes_stale(self):
        with self.store() as store:
            store.prepare("left", self.frames)
            store.prepare("right", self.frames)
            store.commit("left", approve=True)
            before = store.snapshot()
            with self.assertRaisesRegex(ValueError, "stale"):
                store.commit("right", approve=True)
            self.assertEqual(store.snapshot(), before)
            self.assertEqual(store.pending(), ("right",))
            store.abort("right")
            self.assertTrue(store.verify())

    def test_simulated_disk_error_rolls_back_all_staged_domain_writes(self):
        self._prepared()
        def fail_at_os(point):
            if point == "after_domain_os":
                raise OSError("simulated disk-full / I/O fault")
        with self.store() as store:
            before = (store.epoch(), store.snapshot(), store._meta())
            with self.assertRaises(OSError):
                store.commit("test-1", approve=True, fault=fail_at_os)
            self.assertEqual(before, (store.epoch(), store.snapshot(), store._meta()))
            self.assertEqual(store.pending(), ("test-1",))
            self.assertTrue(store.verify())
            store.commit("test-1", approve=True)
            self.assertTrue(store.verify())

    def test_crash_at_every_commit_boundary_and_restart(self):
        points = (
            "after_domain_ann", "after_domain_geometry", "after_domain_os",
            "after_domain_vm", "before_sql_commit", "after_sql_commit",
        )
        for index, point in enumerate(points):
            with self.subTest(point=point):
                path = Path(self.temp.name) / f"crash-{index}.sqlite3"
                with DurableShadowStore(path) as store:
                    store.prepare("crash-tx", self.frames)
                command = (
                    "import os,sys;"
                    "from reference.canonical_shadow_tx.durable import DurableShadowStore;"
                    "store=DurableShadowStore(sys.argv[1]);"
                    "store.commit('crash-tx',approve=True,"
                    "fault=lambda event: os._exit(73) if event==sys.argv[2] else None)"
                )
                result = subprocess.run(
                    [sys.executable, "-c", command, str(path), point],
                    check=False, capture_output=True, text=True, timeout=20,
                )
                self.assertEqual(result.returncode, 73, result.stderr)
                with DurableShadowStore(path) as reopened:
                    self.assertTrue(reopened.verify())
                    if point == "after_sql_commit":
                        self.assertEqual(reopened.epoch(), 1)
                        self.assertEqual(reopened.snapshot(), self.frames)
                        self.assertEqual(reopened.pending(), ())
                    else:
                        self.assertEqual(reopened.epoch(), 0)
                        self.assertEqual(reopened.snapshot(), ())
                        self.assertEqual(reopened.pending(), ("crash-tx",))
                    receipt = reopened.commit("crash-tx", approve=True)
                    self.assertEqual(receipt.epoch, 1)
                    self.assertEqual(reopened.snapshot(), self.frames)
                    self.assertTrue(reopened.verify())

    def test_prepared_survives_restart_after_preparation_commit(self):
        with self.store() as store:
            def report_durable(point):
                if point == "after_prepare_durable":
                    raise SystemExit(71)
            with self.assertRaises(SystemExit):
                store.prepare("prepared-crash", self.frames, fault=report_durable)
            self.assertEqual(store.pending(), ("prepared-crash",))
        with self.store() as reopened:
            self.assertEqual(reopened.snapshot(), ())
            self.assertEqual(reopened.pending(), ("prepared-crash",))
            self.assertTrue(reopened.verify())

    def test_crash_before_prepare_commit_has_no_pending_candidate(self):
        with self.store() as store:
            def fail(point):
                if point == "before_prepare_commit":
                    raise OSError("injected pre-prepare commit failure")
            with self.assertRaises(OSError):
                store.prepare("not-durable", self.frames, fault=fail)
            self.assertEqual(store.pending(), ())
        with self.store() as reopened:
            self.assertEqual(reopened.pending(), ())
            self.assertTrue(reopened.verify())

    def test_concurrent_connections_serialize_and_reject_stale(self):
        with self.store() as first, self.store() as second:
            first.prepare("one", self.frames)
            second.prepare("two", self.frames)
            first.commit("one", approve=True)
            with self.assertRaisesRegex(ValueError, "stale"):
                second.commit("two", approve=True)
            self.assertEqual(first.epoch(), 1)
            self.assertEqual(second.epoch(), 1)
            self.assertEqual(first.snapshot(), second.snapshot())
            self.assertTrue(second.verify())

    def test_tampered_current_state_fails_closed(self):
        self._prepared()
        with self.store() as store:
            store.commit("test-1", approve=True)
        with sqlite3.connect(self.path) as unsafe:
            unsafe.execute("UPDATE current SET payload=? WHERE domain='vm'", (b"corrupt",))
        with self.assertRaisesRegex(RuntimeError, "integrity"):
            self.store()

    def test_tampered_journal_fails_closed(self):
        self._prepared()
        with self.store() as store:
            store.commit("test-1", approve=True)
        with sqlite3.connect(self.path) as unsafe:
            unsafe.execute("UPDATE commits SET after_hash=? WHERE epoch=1", ("f" * 64,))
        with self.assertRaisesRegex(RuntimeError, "integrity"):
            self.store()

    def test_identifier_and_approval_validation(self):
        with self.store() as store:
            for txid in ("", "bad/path", "x" * 65, "../", "�"):
                with self.subTest(txid=txid):
                    with self.assertRaises(ValueError):
                        store.prepare(txid, self.frames)
            store.prepare("okay", self.frames)
            with self.assertRaises(PermissionError):
                store.commit("okay", approve=1)
            self.assertEqual(store.epoch(), 0)

    def test_completes_second_epoch_and_validates_chain(self):
        with self.store() as store:
            store.prepare("first", self.frames)
            store.commit("first", approve=True)
            store.prepare("second", self.frames)
            r = store.commit("second", approve=True)
            self.assertEqual(r.epoch, 2)
            self.assertEqual(len(store.snapshot()), 4)
            self.assertTrue(store.verify())
        with self.store() as reopened:
            self.assertEqual(reopened.epoch(), 2)
            self.assertTrue(reopened.verify())


if __name__ == "__main__":
    unittest.main()
