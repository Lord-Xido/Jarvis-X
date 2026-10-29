"""Do not silently promote a research reference into production authority."""
from __future__ import annotations

import json
import subprocess
import sys
import unittest

from reference.canonical_shadow_tx.release_gate import (
    REQUIRED_PRODUCTION_EVIDENCE, SCHEMA, report,
)


class TestProductionGate(unittest.TestCase):
    def test_report_is_explicitly_blocked(self):
        output = report("0123456789abcdef")
        self.assertEqual(output["schema"], SCHEMA)
        self.assertEqual(output["production_readiness"], "BLOCKED")
        self.assertIs(output["production_deployment_allowed"], False)
        self.assertEqual(len(output["required_unverified"]), 8)
        self.assertEqual(
            {item["gate"] for item in output["required_unverified"]},
            set(REQUIRED_PRODUCTION_EVIDENCE),
        )
        self.assertTrue(all(x["status"] == "BLOCKED" for x in output["required_unverified"]))

    def test_plain_evidence_command_succeeds_for_ci(self):
        run = subprocess.run(
            [sys.executable, "-m", "reference.canonical_shadow_tx.release_gate",
             "--commit", "test-revision"],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(json.loads(run.stdout)["commit_sha"], "test-revision")
        self.assertFalse(json.loads(run.stdout)["production_deployment_allowed"])

    def test_production_flag_fails_closed_with_parseable_blockers(self):
        run = subprocess.run(
            [sys.executable, "-m", "reference.canonical_shadow_tx.release_gate",
             "--require-production"],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(run.returncode, 2, run.stderr)
        output = json.loads(run.stdout)
        self.assertEqual(output["production_readiness"], "BLOCKED")
        self.assertEqual(len(output["required_unverified"]), 8)


if __name__ == "__main__":
    unittest.main()
