"""Real subprocess + Unix socket HMAC authentication and recovery acceptance.

No fake service mocks, no actual VM/OS/ANN live-state writes. Four separately
running child processes own the shadow SQLite participants.
"""
from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import secrets
import subprocess
import sys
import tempfile
import time
import unittest

from reference.canonical_shadow_tx.bridge import (
    DOMAINS, observe_ann, observe_geometry, observe_os, observe_vm,
)
from reference.canonical_shadow_tx.ipc import (
    AuthenticationError, IPCCoordinator, MAX_SKEW_SECONDS, ParticipantService,
    PeerClient, make_request,
)

SECRET = b"TEST-ONLY-NOT-FOR-DEPLOYMENT-" + b"A" * 40
OTHER_SECRET = b"TEST-ONLY-NOT-FOR-DEPLOYMENT-" + b"B" * 40


class TestAuthenticatedProcessBoundary(unittest.TestCase):
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
        self.root = Path(self.temp.name)
        self.children: dict[str, subprocess.Popen] = {}
        self.addCleanup(self.stop_all)
        self.env = os.environ.copy()
        self.env["JARVISX_SHADOW_IPC_SECRET"] = SECRET.decode("ascii")
        for domain in DOMAINS:
            self.start(domain)

    def start(self, domain: str):
        self.cleanup_socket(domain)
        child = subprocess.Popen(
            [sys.executable, "-m", "reference.canonical_shadow_tx.ipc",
             str(self.root), "serve", "--domain", domain],
            env=self.env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
        self.children[domain] = child
        peer = PeerClient(self.root, domain, SECRET, timeout=0.2)
        ready = False
        for _ in range(120):
            if child.poll() is not None:
                raise AssertionError(f"participant {domain} terminated during start: "
                                     + child.stderr.read().decode(errors="replace"))
            try:
                if peer.call("verify")["verified"]:
                    ready = True
                    break
            except (ConnectionRefusedError, FileNotFoundError, TimeoutError,
                    BrokenPipeError, OSError):
                time.sleep(0.025)
        self.assertTrue(ready, f"participant {domain} failed to start")
        return child

    def cleanup_socket(self, domain: str):
        path = self.root / (domain + ".sock")
        if path.exists() or path.is_symlink():
            path.unlink()

    def stop(self, domain: str):
        proc = self.children.pop(domain, None)
        if proc:
            proc.terminate()
            try:
                proc.wait(timeout=4)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=4)
            if proc.stderr:
                proc.stderr.close()
        self.cleanup_socket(domain)

    def stop_all(self):
        for domain in tuple(self.children):
            self.stop(domain)

    def coordinator(self):
        return IPCCoordinator(self.root, secret=SECRET)

    def test_four_independent_service_pids_have_authenticated_state(self):
        self.assertEqual(len({proc.pid for proc in self.children.values()}), 4)
        self.assertNotIn(os.getpid(), {proc.pid for proc in self.children.values()})
        for domain in DOMAINS:
            with self.subTest(domain=domain):
                peer = PeerClient(self.root, domain, SECRET)
                data = peer.call("state")
                self.assertEqual(data["version"], 0)
                self.assertIsNone(data["frame"])
                self.assertTrue(peer.call("verify")["verified"])
                self.assertEqual((self.root / (domain + ".sock")).stat().st_mode & 0o777,
                                 0o600)

    def test_authenticated_prepare_complete_and_idempotent_receipt(self):
        with self.coordinator() as c:
            candidate = c.prepare("rpc-1", self.frames)
            self.assertEqual(c.decision_status("rpc-1"), "PREPARED")
            for domain in DOMAINS:
                self.assertIsNone(PeerClient(self.root, domain, SECRET).call("state")["frame"])
            receipt = c.approve("rpc-1", approve=True)
            self.assertEqual(receipt.status, "COMPLETE")
            self.assertEqual(receipt.state_hash, candidate)
            self.assertEqual(c.coordinated_read("rpc-1"), self.frames)
        with self.coordinator() as reopened:
            self.assertEqual(reopened.approve("rpc-1", approve=True), receipt)
            self.assertEqual(reopened.receipt("rpc-1")["acked_domains"], list(DOMAINS))

    def test_missing_approval_prevents_durable_decision(self):
        with self.coordinator() as c:
            c.prepare("approval", self.frames)
            with self.assertRaises(PermissionError):
                c.approve("approval")
            self.assertEqual(c.decision_status("approval"), "PREPARED")
            self.assertEqual(c.recover(), {"approval": "ABORTED"})
            self.assertEqual(c._meta()[0], 0)

    def test_forged_request_wrong_secret_and_wrong_domain_are_rejected(self):
        for domain in DOMAINS:
            with self.subTest(domain=domain):
                forged = PeerClient(self.root, domain, OTHER_SECRET)
                with self.assertRaises((RuntimeError, AuthenticationError)):
                    forged.call("state")
        mismatched = make_request(SECRET, "ann", "state", {})
        with self.assertRaises(RuntimeError):
            PeerClient(self.root, "geometry", SECRET).exchange(mismatched)
        for domain in DOMAINS:
            self.assertEqual(PeerClient(self.root, domain, SECRET).call("state")["version"], 0)

    def test_tampered_mac_or_timestamp_fails_closed(self):
        peer = PeerClient(self.root, "ann", SECRET)
        changed = make_request(SECRET, "ann", "state", {})
        changed["action"] = "commit"
        with self.assertRaises(RuntimeError):
            peer.exchange(changed)
        stale = make_request(SECRET, "ann", "state", {},
                             timestamp=int(time.time()) - MAX_SKEW_SECONDS - 10)
        with self.assertRaises(RuntimeError):
            peer.exchange(stale)
        self.assertIsNone(peer.call("state")["frame"])

    def test_nonce_replay_rejected_and_survives_process_restart(self):
        peer = PeerClient(self.root, "ann", SECRET)
        request = make_request(SECRET, "ann", "state", {})
        self.assertEqual(peer.exchange(request)["version"], 0)
        with self.assertRaises(RuntimeError):
            peer.exchange(request)
        self.stop("ann")
        self.start("ann")
        with self.assertRaises(RuntimeError):
            peer.exchange(request)
        self.assertTrue(peer.call("verify")["verified"])

    def test_authentication_is_required_even_for_read_only_rpcs(self):
        for name in ("state", "verify"):
            with self.subTest(name=name):
                wrong = PeerClient(self.root, "os", OTHER_SECRET)
                with self.assertRaises((RuntimeError, AuthenticationError)):
                    wrong.call(name)
        self.assertTrue(PeerClient(self.root, "os", SECRET).call("verify")["verified"])

    def test_coordinator_process_termination_after_durable_decision(self):
        with self.coordinator() as c:
            c.prepare("crash", self.frames)
        script = """
import os,sys
from reference.canonical_shadow_tx.ipc import IPCCoordinator
with IPCCoordinator(sys.argv[1]) as c:
    c.approve('crash', approve=True,
              fault=lambda p: os._exit(73) if p=='after_apply_ann' else None)
"""
        run = subprocess.run(
            [sys.executable, "-c", script, str(self.root)], env=self.env,
            check=False, capture_output=True, text=True, timeout=20,
        )
        self.assertEqual(run.returncode, 73, run.stderr)
        with self.coordinator() as c:
            self.assertEqual(c.decision_status("crash"), "COMMIT_DECIDED")
            with self.assertRaises(RuntimeError):
                c.coordinated_read("crash")
            self.assertEqual(c.recover(), {"crash": "COMPLETE"})
            self.assertEqual(c.coordinated_read("crash"), self.frames)

    def test_participant_down_during_replay_cannot_fake_completion(self):
        with self.coordinator() as c:
            c.prepare("down", self.frames)
            def stop_after_decision(point):
                if point == "after_decision":
                    raise OSError("deliberate interruption after durable decision")
            with self.assertRaises(OSError):
                c.approve("down", approve=True, fault=stop_after_decision)
            self.assertEqual(c.decision_status("down"), "COMMIT_DECIDED")
        self.stop("os")
        with self.coordinator() as c:
            with self.assertRaises(OSError):
                c.recover()
            self.assertEqual(c.decision_status("down"), "COMMIT_DECIDED")
        self.start("os")
        with self.coordinator() as c:
            self.assertEqual(c.recover(), {"down": "COMPLETE"})
            self.assertEqual(c.coordinated_read("down"), self.frames)

    def test_restart_after_prepared_aborts_without_applying(self):
        with self.coordinator() as c:
            c.prepare("undecided", self.frames)
        self.stop("geometry")
        self.start("geometry")
        with self.coordinator() as c:
            self.assertEqual(c.recover(), {"undecided": "ABORTED"})
            self.assertEqual(c._meta()[0], 0)
        for domain in DOMAINS:
            self.assertIsNone(PeerClient(self.root, domain, SECRET).call("state")["frame"])

    def test_raw_state_visibility_limit_is_preserved_across_processes(self):
        with self.coordinator() as c:
            c.prepare("partial", self.frames)
            def stop_after_ann(point):
                if point == "after_apply_ann":
                    raise OSError("simulate interruption between remote processes")
            with self.assertRaises(OSError):
                c.approve("partial", approve=True, fault=stop_after_ann)
            self.assertIsNotNone(PeerClient(self.root, "ann", SECRET).call("state")["frame"])
            self.assertIsNone(PeerClient(self.root, "geometry", SECRET).call("state")["frame"])
            with self.assertRaises(RuntimeError):
                c.coordinated_read("partial")
            self.assertEqual(c.recover(), {"partial": "COMPLETE"})
            self.assertEqual(c.coordinated_read("partial"), self.frames)

    def test_unconfigured_secret_fails_closed(self):
        environment = os.environ.copy()
        environment.pop("JARVISX_SHADOW_IPC_SECRET", None)
        result = subprocess.run(
            [sys.executable, "-m", "reference.canonical_shadow_tx.ipc",
             str(self.root), "status", "--txid", "missing"],
            env=environment, check=False, capture_output=True, timeout=10,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn(SECRET, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
