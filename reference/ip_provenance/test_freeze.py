"""Frozen IP baseline verification: crypto, repro and exact 3D witness."""
from __future__ import annotations

from fractions import Fraction
import hashlib
import unittest

from reference.ip_provenance.freeze import (
    FROZEN_COMMIT, FROZEN_GIT_TREE, _leaf, _sha256,
    capture, canonical_bytes, merkle,
)
from reference.ip_provenance.math_exact import (
    DOMAINS, STATES, VERTICES, audit_3d, encode_exact,
    exact_recurrence, status_coordinate_code, volume_exact,
)


class TestIPFreeze(unittest.TestCase):
    def test_leaf_commits_to_path_mode_length_and_payload(self):
        digest = _sha256(b"file-contents")
        original = _leaf(b"file.py", b"100644", digest, 13)
        self.assertEqual(len(original), 32)
        self.assertNotEqual(original, _leaf(b"file2.py", b"100644", digest, 13))
        self.assertNotEqual(original, _leaf(b"file.py", b"100755", digest, 13))
        self.assertNotEqual(original, _leaf(b"file.py", b"100644", digest, 14))
        self.assertNotEqual(original, _leaf(b"file.py", b"100644", _sha256(b"wrong"), 13))

    def test_merkle_fails_on_any_leaf_change_or_order_change(self):
        leaves = [_sha256(b"a"), _sha256(b"b"), _sha256(b"c")]
        root = merkle(leaves)
        self.assertEqual(len(root), 64)
        self.assertEqual(root, merkle(list(leaves)))
        self.assertNotEqual(root, merkle([leaves[1], leaves[0], leaves[2]]))
        self.assertNotEqual(root, merkle(leaves[:2]))
        self.assertNotEqual(root, merkle([leaves[0], leaves[1], _sha256(b"d")]))
        with self.assertRaises(ValueError):
            merkle([])
        with self.assertRaises(ValueError):
            merkle([b"short"])

    def test_exhaustive_3d_status_and_geometry(self):
        report = audit_3d()
        self.assertEqual(report["source_commit"], FROZEN_COMMIT)
        self.assertEqual(report["geometry"]["voxels"], 512)
        self.assertEqual(report["geometry"]["signed_volume_exact"],
                         {"numerator": "-125", "denominator": "3"})
        self.assertEqual(
            report["codec"]["status_configurations_exhaustively_verified"], 625
        )
        self.assertTrue(report["codec"]["bit_identical_reconstruction"])
        self.assertTrue(report["codec"]["atlas_anchor_verified"])
        self.assertEqual(len(report["codec"]["status_atlas_merkle_sha256"]), 64)
        self.assertEqual(report["production_deployment"], "BLOCKED")
        self.assertEqual(report["external_evidence"], "NOT_APPLICABLE")

    def test_1000_actual_exact_rational_steps(self):
        result = exact_recurrence()
        self.assertEqual(result["steps_executed_exactly"], 1000)
        self.assertEqual(result["rational_proof"], "PASS")
        e0 = Fraction(9, 4)
        self.assertEqual(Fraction(int(result["E0"]["numerator"]),
                                  int(result["E0"]["denominator"])), e0)
        final = Fraction(int(result["E1000"]["numerator"]),
                         int(result["E1000"]["denominator"]))
        self.assertEqual(final, e0 / 4**1000)
        self.assertEqual(result["commit_semantics"].split(":")[0], "NONE")

    def test_canonical_json_is_deterministic(self):
        obj = {"z": 2, "a": 1}
        self.assertEqual(canonical_bytes(obj), b'{"a":1,"z":2}\n')

    def test_base_git_blob_inventory_and_merkle_reproduce(self):
        report = capture()
        self.assertEqual(report["frozen_git_commit"], FROZEN_COMMIT)
        self.assertEqual(report["frozen_git_tree"], FROZEN_GIT_TREE)
        self.assertGreaterEqual(report["file_count"], 900)
        self.assertEqual(len(report["files"]), report["file_count"])
        self.assertEqual(len(report["sha256_merkle_root"]), 64)
        self.assertTrue(all(len(row["sha256"]) == 64 for row in report["files"]))
        self.assertTrue(all(len(row["git_blob_sha1"]) == 40 for row in report["files"]))
        self.assertEqual(report["production_gate"], "BLOCKED")


if __name__ == "__main__":
    unittest.main()
