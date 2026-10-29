"""Public DM3D-ENV1 Python/C++ bytewise and transaction conformance."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import shutil
import subprocess
import tempfile
import unittest

from envelope import (
    Geometry, HEADER, Proposal, ReferenceTransaction, digest, from_morton,
    morton_index, pack, row_index, to_morton, unpack,
)

HERE = Path(__file__).resolve().parent


def vector(side: int, salt: int = 0) -> bytes:
    rng = random.Random(0xD3D3 + side + salt)
    return bytes(rng.getrandbits(8) for _ in range(side**3))


class TestEnvelope(unittest.TestCase):
    def test_exact_header_size(self):
        self.assertEqual(HEADER.size, 84)

    def test_golden_magic_and_size(self):
        raw = vector(8)
        blob = pack(raw, Geometry(8))
        self.assertEqual(blob[:8], b"DM3DENV1")
        self.assertEqual(len(blob), 84 + 512)
        self.assertEqual(blob[52:84], hashlib.sha256(raw).digest())

    def test_roundtrip_multiple_sizes(self):
        for side in (2, 4, 8, 16, 32, 64):
            with self.subTest(side=side):
                raw = vector(side)
                self.assertEqual(unpack(pack(raw, Geometry(side))), (Geometry(side), raw))

    def test_zero_payload_is_full_cube(self):
        for side in (2, 8, 16):
            raw = bytes(side**3)
            self.assertEqual(unpack(pack(raw, Geometry(side)))[1], raw)

    def test_non_power_two_and_oversize_rejected(self):
        for side in (0, 1, 3, 7, 65, 128, -1, True):
            with self.subTest(side=side), self.assertRaises(ValueError):
                Geometry(side).check()

    def test_not_a_cube_rejected(self):
        with self.assertRaises(ValueError):
            pack(b"abc", Geometry(8))

    def test_corruption_detection(self):
        blob = bytearray(pack(vector(8), Geometry(8)))
        blob[-2] ^= 0x80
        with self.assertRaisesRegex(ValueError, "integrity"):
            unpack(bytes(blob))

    def test_trailing_and_truncated_data_rejected(self):
        blob = pack(vector(8), Geometry(8))
        for malformed in (blob + b"X", blob[:-1], blob[:10]):
            with self.assertRaises(ValueError):
                unpack(malformed)

    def test_noncanonical_header_fields_rejected(self):
        original = pack(vector(8), Geometry(8))
        for offset in (0, 9, 10, 12, 28, 29, 30, 31, 32, 36, 40, 51):
            mutated = bytearray(original)
            mutated[offset] ^= 1
            with self.subTest(offset=offset), self.assertRaises(ValueError):
                unpack(bytes(mutated))

    def test_morton_known_coordinate(self):
        self.assertEqual(row_index(1, 2, 3, 16), 801)
        self.assertEqual(morton_index(1, 2, 3, 16), 53)

    def test_morton_bijection(self):
        for side in (2, 4, 8, 16):
            with self.subTest(side=side):
                addresses = {
                    morton_index(x, y, z, side)
                    for z in range(side) for y in range(side) for x in range(side)
                }
                self.assertEqual(addresses, set(range(side**3)))

    def test_spatial_roundtrip(self):
        for side in (2, 4, 8, 16):
            with self.subTest(side=side):
                raw = vector(side)
                self.assertEqual(
                    from_morton(to_morton(raw, Geometry(side)), Geometry(side)), raw
                )

    def test_out_of_range_coordinates(self):
        for xyz in ((-1, 0, 0), (0, 16, 0), (0, 0, 16)):
            with self.assertRaises(ValueError):
                morton_index(*xyz, 16)

    def test_declared_digest(self):
        raw = vector(16)
        self.assertEqual(digest(raw), hashlib.sha256(raw).hexdigest())


class TestTransaction(unittest.TestCase):
    def setUp(self):
        self.geometry = Geometry(8)
        self.before = vector(8)
        self.after = vector(8, 1)
        self.tx = ReferenceTransaction(self.before, self.geometry)

    def test_staging_is_non_mutating(self):
        proposal = self.tx.stage(self.after)
        self.assertEqual(self.tx.authoritative, self.before)
        self.assertEqual(self.tx.epoch, 0)
        self.assertEqual(proposal.receipt()["decision"], "PENDING")
        self.assertEqual(proposal.receipt()["gates"]["external_evidence"], "NOT_APPLICABLE")

    def test_requires_explicit_approval(self):
        proposal = self.tx.stage(self.after)
        with self.assertRaises(PermissionError):
            self.tx.commit(proposal)
        self.assertEqual(self.tx.authoritative, self.before)

    def test_successful_commit_lineage(self):
        proposal = self.tx.stage(self.after)
        receipt = self.tx.commit(proposal, approve=True)
        self.assertEqual(self.tx.authoritative, self.after)
        self.assertEqual(self.tx.epoch, 1)
        self.assertEqual(receipt["decision"], "COMMITTED")
        self.assertEqual(receipt["journal_parent"], "0" * 64)
        self.assertEqual(receipt["journal_head"], self.tx.journal_head)

    def test_reject_stale_proposal(self):
        first = self.tx.stage(self.after)
        second = self.tx.stage(vector(8, 2))
        self.tx.commit(first, approve=True)
        with self.assertRaises(ValueError):
            self.tx.commit(second, approve=True)
        self.assertEqual(self.tx.authoritative, self.after)

    def test_reject_tampered_candidate_without_mutation(self):
        proposal = self.tx.stage(self.after)
        bad = bytearray(proposal.envelope)
        bad[-1] ^= 0x80
        tampered = Proposal(
            proposal.epoch, proposal.before_sha256, proposal.after_sha256,
            bytes(bad), proposal.gates
        )
        with self.assertRaises(ValueError):
            self.tx.commit(tampered, approve=True)
        self.assertEqual(self.tx.authoritative, self.before)
        self.assertEqual(self.tx.epoch, 0)

    def test_reject_unverified_gate(self):
        proposal = self.tx.stage(self.after)
        gates = tuple(
            (name, "FAIL" if name == "policy" else value)
            for name, value in proposal.gates
        )
        invalid = Proposal(
            proposal.epoch, proposal.before_sha256, proposal.after_sha256,
            proposal.envelope, gates
        )
        with self.assertRaises(ValueError):
            self.tx.commit(invalid, approve=True)
        self.assertEqual(self.tx.authoritative, self.before)

    def test_incorrect_geometry_staging_keeps_authoritative_state(self):
        with self.assertRaises(ValueError):
            self.tx.stage(b"truncated")
        self.assertEqual(self.tx.authoritative, self.before)

    def test_deterministic_journal(self):
        other = ReferenceTransaction(self.before, self.geometry)
        first = self.tx.commit(self.tx.stage(self.after), approve=True)
        second = other.commit(other.stage(self.after), approve=True)
        self.assertEqual(first["journal_head"], second["journal_head"])


class TestCrossLanguage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("g++")
        if compiler is None:
            raise unittest.SkipTest("g++ not installed; Python-only checks still run")
        cls.tmpdir = tempfile.TemporaryDirectory()
        cls.executable = str(Path(cls.tmpdir.name) / "dm3d_env1_cpp")
        subprocess.run(
            [compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
             str(HERE / "envelope.cpp"), "-o", cls.executable],
            check=True, timeout=45,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "tmpdir"):
            cls.tmpdir.cleanup()

    def run_cpp(self, *args, ok=True):
        result = subprocess.run(
            [self.executable, *map(str, args)],
            capture_output=True, text=True, timeout=15,
        )
        self.assertEqual(result.returncode == 0, ok, result.stderr)
        return result

    def test_python_to_cpp_full_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            for side in (2, 8, 16, 32, 64):
                with self.subTest(side=side):
                    raw = vector(side)
                    env = Path(directory) / "input.env"
                    result = Path(directory) / "output.bin"
                    env.write_bytes(pack(raw, Geometry(side)))
                    self.run_cpp("unpack", env, result)
                    self.assertEqual(result.read_bytes(), raw)
                    inspected = json.loads(self.run_cpp("inspect", env).stdout)
                    self.assertEqual(inspected["sha256"], digest(raw))

    def test_cpp_to_python_bit_identical_envelope(self):
        with tempfile.TemporaryDirectory() as directory:
            for side in (2, 8, 16, 32, 64):
                with self.subTest(side=side):
                    raw = vector(side, 33)
                    source = Path(directory) / "input.bin"
                    encoded = Path(directory) / "output.env"
                    source.write_bytes(raw)
                    self.run_cpp("pack", source, side, encoded)
                    self.assertEqual(encoded.read_bytes(), pack(raw, Geometry(side)))
                    self.assertEqual(unpack(encoded.read_bytes())[1], raw)

    def test_shared_morton_coordinates(self):
        for side in (2, 8, 16, 32, 64):
            for xyz in ((0, 0, 0), (1, 1, 1),
                        (side - 1, side - 1, side - 1), (1, 0, side - 1)):
                with self.subTest(side=side, xyz=xyz):
                    text = self.run_cpp("index", side, *xyz).stdout.strip()
                    row, morton = (int(value) for value in text.split())
                    self.assertEqual(row, row_index(*xyz, side))
                    self.assertEqual(morton, morton_index(*xyz, side))

    def test_cpp_rejects_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.env"
            bad = bytearray(pack(vector(8), Geometry(8)))
            bad[HEADER.size + 7] ^= 0x44
            path.write_bytes(bad)
            self.run_cpp("inspect", path, ok=False)

    def test_cpp_rejects_unrecognised_headers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.env"
            bad = bytearray(pack(vector(8), Geometry(8)))
            bad[29] = 2
            path.write_bytes(bad)
            self.run_cpp("inspect", path, ok=False)


if __name__ == "__main__":
    unittest.main()
