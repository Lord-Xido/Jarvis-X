#!/usr/bin/env python3
"""Offline conformance tests for the proposal-only AI Commercial Operator."""
import json
from pathlib import Path
import tempfile
import unittest

from jarvis_x_auto_operator import make_demo, evaluate, plan, read_csv, hash_data

HERE = Path(__file__).resolve().parent


class OperatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((HERE / "config.json").read_text())
        cls.tmp = tempfile.TemporaryDirectory()
        cls.rows = make_demo(cls.config, Path(cls.tmp.name))
        cls.markets = {m["id"]: m for m in cls.config["markets"]}
        cls.model, cls.metrics = evaluate(cls.rows, cls.markets)
        cls.report = plan(cls.config, cls.rows, cls.model, cls.metrics, "SYNTHETIC_DEMO")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_synthetic_data_replay(self):
        with tempfile.TemporaryDirectory() as path:
            again = make_demo(self.config, Path(path))
        self.assertEqual(hash_data(again), hash_data(self.rows))

    def test_chronological_backtest(self):
        self.assertLess(self.metrics["train_last_week"], self.metrics["val_last_week"])
        self.assertGreater(self.metrics["test_rows"], 10)
        self.assertGreaterEqual(self.metrics["test_mae"], 0)

    def test_csv_contract(self):
        rows = read_csv(Path(self.tmp.name) / "synthetic_observations.csv", self.config)
        self.assertEqual(len(rows), len(self.rows))
        self.assertTrue(all(row["uncensored"] for row in rows))

    def test_censored_rows_are_excluded(self):
        censored = [dict(r, uncensored=False) if r["week"] == 30 else r for r in self.rows]
        _, metrics = evaluate(censored, self.markets)
        self.assertEqual(metrics["train_rows"], self.metrics["train_rows"] - len(self.markets))

    def test_budget_compute_cash_and_receivables(self):
        r = self.report
        self.assertEqual(r["weekly_promotion_zar"], sum(m["promo_zar"] for m in r["markets"]))
        self.assertLessEqual(r["weekly_promotion_zar"], self.config["weekly_marketing_budget_zar"])
        self.assertLessEqual(r["weekly_inference_requests"], self.config["compute_request_capacity_week"])
        self.assertGreaterEqual(r["forecast_end_cash_zar"], self.config["minimum_cash_reserve_zar"])
        self.assertAlmostEqual(r["weekly_operating_contribution_zar"],
            r["weekly_revenue_zar"] - r["weekly_variable_cash_zar"]
            - r["weekly_promotion_zar"] - r["weekly_platform_fixed_cash_zar"], places=2)
        self.assertAlmostEqual(r["forecast_end_cash_zar"],
            self.config["starting_cash_zar"] + r["forecast_customer_collections_zar"]
            - r["weekly_variable_cash_zar"] - r["weekly_promotion_zar"]
            - r["weekly_platform_fixed_cash_zar"], places=2)
        self.assertAlmostEqual(r["forecast_end_receivables_zar"],
            r["weekly_revenue_zar"] * (1 - self.config["cash_collection_fraction_week"]), places=2)

    def test_external_execution_never_authorised(self):
        r = self.report
        self.assertFalse(r["expansion_authorized"])
        self.assertFalse(r["approval_gate"]["external_execution_enabled"])
        self.assertTrue(r["approval_gate"]["human_approval_required"])
        if r["weekly_operating_contribution_zar"] <= 0:
            self.assertEqual(r["status"], "HOLD_UNIT_ECONOMICS")

    def test_infeasible_compute_budget(self):
        cfg = dict(self.config, compute_request_capacity_week=0)
        r = plan(cfg, self.rows, self.model, self.metrics, "SYNTHETIC_DEMO")
        self.assertEqual(r["status"], "INFEASIBLE")
        self.assertEqual(r["execution"], "PROPOSALS_ONLY")
        self.assertFalse(r.get("expansion_authorized", False))

    def test_no_real_mode_in_config(self):
        self.assertEqual(self.config["mode"], "simulation")


if __name__ == "__main__":
    unittest.main(verbosity=2)
