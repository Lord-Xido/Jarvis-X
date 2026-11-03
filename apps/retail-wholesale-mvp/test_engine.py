"""Conformance tests for the stateless retail/wholesale demo."""
import copy
import json
import math
import unittest

from demo import sample_payload
from engine import compute, InputError


class CommercialPlanTests(unittest.TestCase):
    def setUp(self):
        self.payload = sample_payload()

    def test_sample_has_distinct_valid_history(self):
        self.assertEqual(len(self.payload["products"]), 3)
        self.assertTrue(all(len(p["daily_sales"]) == 35 for p in self.payload["products"]))

    def test_deterministic_replay(self):
        result = compute(self.payload)
        self.assertEqual(result, compute(copy.deepcopy(self.payload)))
        self.assertFalse(result["execution_enabled"])
        self.assertTrue(result["requires_human_approval"])

    def test_financial_cash_conservation(self):
        r = compute(self.payload)
        s = r["summary"]
        self.assertAlmostEqual(s["forecast_ending_cash_zar"],
           self.payload["cash_available_zar"] - s["proposed_procurement_cash_zar"]
           - s["promotion_spend_zar"] + s["cash_collected_zar"],places=2)
        self.assertAlmostEqual(s["forecast_gross_margin_zar"],
           s["forecast_weekly_revenue_zar"] - s["forecast_weekly_cogs_zar"],places=2)
        self.assertAlmostEqual(s["forecast_weekly_revenue_zar"],
           s["cash_collected_zar"] + s["accounts_receivable_created_zar"],places=2)

    def test_sales_cannot_exceed_current_saleable_stock(self):
        r = compute(self.payload)
        for p in r["products"]:
            self.assertLessEqual(p["projected_sales_units"],p["on_hand"] - p["reserved"])
            self.assertGreaterEqual(p["projected_sales_units"],0)

    def test_procurement_within_cash_and_capacity(self):
        r = compute(self.payload)
        self.assertLessEqual(r["summary"]["proposed_procurement_cash_zar"],
                             self.payload["cash_available_zar"] + 0.01)
        self.assertLessEqual(r["summary"]["planned_replenishment_units"],
                             self.payload["production_capacity_units"])

    def test_zero_cash_prevents_procurement(self):
        p=copy.deepcopy(self.payload)
        p["cash_available_zar"]=0
        r=compute(p)
        self.assertEqual(r["summary"]["planned_replenishment_units"],0)

    def test_zero_capacity_prevents_procurement(self):
        p=copy.deepcopy(self.payload)
        p["production_capacity_units"]=0
        r=compute(p)
        self.assertEqual(r["summary"]["planned_replenishment_units"],0)

    def test_promotion_does_not_magically_increase_demand(self):
        before=compute(self.payload)
        altered=copy.deepcopy(self.payload)
        altered["products"][0]["promotion_zar"]=5000
        after=compute(altered)
        matching=lambda p,sku:next(x for x in p["products"] if x["sku"]==sku)
        self.assertEqual(matching(before,"WATER-FILTER")["weekly_forecast"],
                         matching(after,"WATER-FILTER")["weekly_forecast"])
        self.assertEqual(after["summary"]["promotion_spend_zar"],5000)

    def test_censored_sales_warn(self):
        p=copy.deepcopy(self.payload)
        p["products"][0]["stockout_days"]=[3,7]
        result=compute(p)
        self.assertTrue(any("stockout" in s.lower() for s in result["warnings"]))
        self.assertFalse(next(x for x in result["products"] if x["sku"]=="WATER-FILTER")["forecast_reliable"])

    def test_privacy_enforced_at_input_contract(self):
        p=copy.deepcopy(self.payload)
        p["products"][0]["customer_email"]="person@example.com"
        with self.assertRaises(InputError):
            compute(p)

    def test_sku_validation(self):
        p=copy.deepcopy(self.payload)
        p["products"][0]["sku"]="../../secrets"
        with self.assertRaises(InputError):
            compute(p)

    def test_non_finite_numeric_rejected(self):
        p=copy.deepcopy(self.payload)
        p["products"][0]["retail_price_zar"]=math.nan
        with self.assertRaises(InputError):
            compute(p)

    def test_reserved_stock_constraint(self):
        p=copy.deepcopy(self.payload)
        p["products"][0]["reserved"]=p["products"][0]["on_hand"]+1
        with self.assertRaises(InputError):
            compute(p)

    def test_zero_demand_is_not_invented(self):
        p=copy.deepcopy(self.payload)
        p["products"][0]["daily_sales"]=[0]*35
        r=compute(p)
        v=next(x for x in r["products"] if x["sku"]=="WATER-FILTER")
        self.assertEqual(v["weekly_forecast"],0)
        self.assertEqual(v["proposed_replenishment"],0)

    def test_serialize_payload_and_report(self):
        json.dumps(compute(self.payload),allow_nan=False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
