"""3D geometric, bitwise, fixed-point and transaction-observation regression."""
from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path
import tempfile
import unittest

from reference.canonical_shadow_tx.bridge import (
    DOMAINS, observe_ann, observe_geometry, observe_os, observe_vm,
)
from reference.canonical_shadow_tx.geometry_witness import (
    GEOMETRY, SIDE, STATUS, VERTICES, decode_lattice, encode_lattice,
    relaxation, signed_tetra_volume, svg_view, witness,
)
from reference.canonical_shadow_tx.multiprocess import Coordinator
from reference.dm3d_env1.envelope import (
    from_morton, pack, row_index, to_morton, unpack,
)


class TestGeometryWitness3D(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(2)
        cls.frames = tuple(sorted(
            (observe_ann(), observe_geometry(), observe_os(), observe_vm()),
            key=lambda f: f.domain,
        ))

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_exact_nondegenerate_tetrahedron(self):
        self.assertEqual(set(VERTICES), set(DOMAINS))
        self.assertEqual(len(set(VERTICES.values())), 4)
        self.assertAlmostEqual(abs(signed_tetra_volume()), 250 / 6, places=12)
        self.assertTrue(all(0 <= axis < SIDE for v in VERTICES.values() for axis in v))
        self.assertEqual(GEOMETRY.size, 512)

    def test_all_statuses_injectively_roundtrip_through_bytes(self):
        for status in STATUS:
            states = {name: status for name in DOMAINS}
            packet = encode_lattice(states)
            self.assertEqual(decode_lattice(packet), states)
            shape, raw = unpack(packet)
            self.assertEqual(shape, GEOMETRY)
            self.assertEqual(len(packet), 596)
            self.assertEqual(from_morton(to_morton(raw, shape), shape), raw)
            self.assertEqual(sum(v != 0 for v in raw), 4)

    def test_distinct_domain_state_code_no_collisions(self):
        for k, name in enumerate(DOMAINS):
            for index, status in enumerate(STATUS):
                states = {domain: "ABSENT" for domain in DOMAINS}
                states[name] = status
                packet = encode_lattice(states)
                _, raw = unpack(packet)
                x, y, z = VERTICES[name]
                self.assertEqual(raw[row_index(x, y, z, SIDE)], k * 8 + index + 1)

    def test_reject_integrity_and_unmapped_voxel_even_if_rehashed(self):
        states = {name: "PREPARED" for name in DOMAINS}
        packet = bytearray(encode_lattice(states))
        packet[-1] ^= 0x80
        with self.assertRaises(ValueError):
            decode_lattice(bytes(packet))
        packet = encode_lattice(states)
        _, raw = unpack(packet)
        edited = bytearray(raw)
        edited[row_index(0, 0, 0, SIDE)] = 0xff
        with self.assertRaisesRegex(ValueError, "unmapped"):
            decode_lattice(pack(bytes(edited), GEOMETRY))

    def test_reject_unknown_status_code_at_domain_vertex(self):
        states = {name: "PREPARED" for name in DOMAINS}
        _, raw = unpack(encode_lattice(states))
        edited = bytearray(raw)
        edited[row_index(*VERTICES["ann"], SIDE)] = 7
        with self.assertRaisesRegex(ValueError, "unknown"):
            decode_lattice(pack(bytes(edited), GEOMETRY))

    def test_contraction_proof_is_bounded_and_does_not_commit(self):
        states = {"ann": "COMMITTED", "geometry": "PREPARED",
                  "os": "ABSENT", "vm": "UNREACHABLE"}
        result = relaxation(states)
        self.assertEqual(result["iterations"], 1000)
        self.assertFalse(result["logical_commit_fixed_point"])
        self.assertEqual(result["initial"], [1.0, 0.5, 0.0, 0.0])
        e0 = result["initial_squared_error"]
        for item in result["samples"]:
            self.assertAlmostEqual(item["squared_error"], e0 * 4**(-item["k"]),
                                   delta=1e-275)
            self.assertTrue(all(math.isfinite(x) for x in item["z"]))
        self.assertEqual(states["os"], "ABSENT")
        self.assertEqual(relaxation({name: "COMMITTED" for name in DOMAINS})
                         ["initial_squared_error"], 0.0)

    def test_reject_unbounded_steps_and_unsorted_domains(self):
        with self.assertRaises(ValueError):
            relaxation({name: "ABSENT" for name in DOMAINS}, iterations=1001)
        with self.assertRaises(ValueError):
            relaxation({name: "ABSENT" for name in reversed(DOMAINS)})
        with self.assertRaises(ValueError):
            encode_lattice({name: "INVALID" for name in DOMAINS})

    def test_observation_from_prepared_to_decided_partial_to_complete(self):
        with Coordinator(self.root) as coordinator:
            coordinator.prepare("geometric", self.frames)
            before = witness(coordinator, "geometric", iterations=64)
            self.assertEqual(before.stage, "PREPARED")
            self.assertTrue(before.verifiable)
            self.assertFalse(before.complete)
            self.assertEqual(set(before.participant_statuses.values()), {"PREPARED"})
            self.assertEqual(before.gate_results["commit_complete"], "BLOCKED")
            self.assertEqual(before.gate_results["external_evidence"], "NOT_APPLICABLE")
            self.assertEqual(decode_lattice(before.packet), before.participant_statuses)

            def interrupt(point):
                if point == "after_apply_ann":
                    raise OSError("injected participant interruption")
            with self.assertRaises(OSError):
                coordinator.approve("geometric", approve=True, fault=interrupt)
            mid = witness(coordinator, "geometric")
            self.assertEqual(mid.stage, "COMMIT_DECIDED")
            self.assertEqual(mid.participant_statuses["ann"], "COMMITTED")
            self.assertEqual(mid.participant_statuses["geometry"], "PREPARED")
            self.assertTrue(mid.verifiable)
            self.assertFalse(mid.complete)
            self.assertEqual(mid.gate_results["distributed_atomic_visibility"], "NOT_PROVEN")
            self.assertEqual(decode_lattice(mid.packet), mid.participant_statuses)
            self.assertEqual(coordinator.recover(), {"geometric": "COMPLETE"})
            after = witness(coordinator, "geometric")
            self.assertTrue(after.verifiable)
            self.assertTrue(after.complete)
            self.assertTrue(after.contraction["logical_commit_fixed_point"])
            self.assertTrue(all(x == "PASS" for x in after.gate_results.values()
                                if x not in ("NOT_PROVEN", "NOT_APPLICABLE")))
            self.assertEqual(decode_lattice(after.packet), after.participant_statuses)
            self.assertEqual(coordinator.coordinated_read("geometric"), self.frames)
        with Coordinator(self.root) as reopened:
            restarted = witness(reopened, "geometric")
            self.assertTrue(restarted.complete)
            self.assertEqual(restarted.packet, after.packet)
            self.assertEqual(restarted.target_sha256, after.target_sha256)
            result = restarted.as_dict()
            self.assertEqual(result["lattice"]["envelope_size_bytes"], 596)
            self.assertEqual(result["lattice"]["side"], 8)
            self.assertEqual(result["authority"], "shadow-only/read-only geometric observation")
            self.assertEqual(result["external_evidence"], "NOT_APPLICABLE")
            self.assertTrue(svg_view(restarted).startswith('<svg'))
            self.assertIn("tetrahedral", svg_view(restarted).splitlines()[0])

    def test_aborted_candidate_is_not_mislabeled_committed(self):
        with Coordinator(self.root) as coordinator:
            coordinator.prepare("cancel", self.frames)
            self.assertEqual(coordinator.recover(), {"cancel": "ABORTED"})
            after = witness(coordinator, "cancel")
            self.assertEqual(after.stage, "ABORTED")
            self.assertTrue(after.verifiable)
            self.assertFalse(after.complete)
            self.assertEqual(set(after.participant_statuses.values()), {"ABORTED"})
            self.assertEqual(after.gate_results["commit_complete"], "BLOCKED")

    def test_missing_participant_invalidates_completed_witness(self):
        with Coordinator(self.root) as coordinator:
            coordinator.prepare("broken", self.frames)
            coordinator.approve("broken", approve=True)
        path = self.root / "os.sqlite3"
        for suffix in ("", "-wal", "-shm"):
            file = Path(str(path) + suffix)
            if file.exists():
                file.unlink()
        with Coordinator(self.root) as coordinator:
            observation = witness(coordinator, "broken")
            self.assertFalse(observation.complete)
            self.assertFalse(observation.verifiable)
            self.assertEqual(observation.gate_results["domain_os"], "MISSING")
            self.assertEqual(observation.participant_statuses["os"], "ABSENT")

    def test_packet_reconstruction_never_authorizes_mutation(self):
        with Coordinator(self.root) as coordinator:
            coordinator.prepare("immutable", self.frames)
            before = coordinator.receipt("immutable")
            result = witness(coordinator, "immutable")
            self.assertTrue(result.verifiable)
            self.assertFalse(result.complete)
            self.assertEqual(coordinator.receipt("immutable"), before)
            self.assertEqual(coordinator.decision_status("immutable"), "PREPARED")
            for domain in DOMAINS:
                with coordinator._participant(domain) as participant:
                    self.assertEqual(participant.status("immutable"), "PREPARED")


if __name__ == "__main__":
    unittest.main()
