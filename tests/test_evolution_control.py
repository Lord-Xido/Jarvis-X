"""Standard-library tests for bounded JARVIS-X architecture evolution control."""
import copy
import importlib.util
import json
import pathlib
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
PATH = ROOT / 'src' / 'jarvisx' / 'evolution_control.py'
spec = importlib.util.spec_from_file_location('evolution_control', PATH)
mod = importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

class EvolutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inventory=mod.load_inventory()

    def test_inventory_has_distinct_domains(self):
        self.assertGreaterEqual(len(self.inventory['components']), 15)
        self.assertGreaterEqual(len({c['repository'] for c in self.inventory['components']}), 3)
        self.assertIn('no-auto-access',{c['authority'] for c in self.inventory['components']})

    def test_inventory_rejects_cycle(self):
        inv=copy.deepcopy(self.inventory)
        inv['tasks'][0]['requires']=['end_to_end_gate']
        with self.assertRaises(ValueError):mod.validate_inventory(inv)

    def test_inventory_rejects_reduced_gates(self):
        inv=copy.deepcopy(self.inventory)
        inv['tasks'][0]['required_gates'].remove('human_approval')
        with self.assertRaises(ValueError):mod.validate_inventory(inv)

    def test_deterministic_thousand_iteration_search(self):
        a=mod.evolve(self.inventory,1000,20261027)
        b=mod.evolve(self.inventory,1000,20261027)
        self.assertEqual(a,b)
        self.assertEqual(a['search']['iterations'],1000)
        self.assertEqual(a['search']['admissible_candidates'],1000)
        self.assertTrue(a['verification']['policy_admissible'])
        self.assertFalse(a['verification']['changes_applied'])
        self.assertFalse(a['verification']['production_authorized'])
        self.assertFalse(a['verification']['tests_executed'])

    def test_1000_iteration_receipt_replay(self):
        report=mod.evolve(self.inventory,1000,12)
        self.assertEqual(mod.verify_receipt(report,self.inventory),(True,()))

    def test_tamper_detected(self):
        report=mod.evolve(self.inventory,10,12)
        report['shadow_score']+=999
        self.assertFalse(mod.verify_receipt(report,self.inventory)[0])

    def test_rehashed_forgery_rejected(self):
        report=mod.evolve(self.inventory,10,12)
        report['candidate']['resident_brick_ceiling']=16 if report['candidate']['resident_brick_ceiling']!=16 else 8
        report['candidate_sha256']=mod.digest(report['candidate'])
        report['shadow_score']=mod.shadow_score(mod.Proposal(**report['candidate']),self.inventory)
        report['receipt_sha256']=mod.digest({k:v for k,v in report.items() if k!='receipt_sha256'})
        self.assertFalse(mod.verify_receipt(report,self.inventory)[0])

    def test_restricted_task_cannot_run(self):
        p=mod.make_proposal(self.inventory,profile='balanced',count=1,bricks=2,repeats=2,cycles=8)
        from dataclasses import replace
        p=replace(p,task_ids=p.task_ids+('financial_release',))
        ok,problems=mod.check_proposal(p,self.inventory)
        self.assertFalse(ok)
        self.assertIn('manual-only task selected: financial_release',problems)

    def test_missing_approval_rejected(self):
        from dataclasses import replace
        p=mod.make_proposal(self.inventory,profile='balanced',count=1,bricks=2,repeats=2,cycles=8)
        p=replace(p,gates=tuple(g for g in p.gates if g!='human_approval'))
        self.assertFalse(mod.check_proposal(p,self.inventory)[0])

    def test_invalid_resource_budget_rejected(self):
        from dataclasses import replace
        p=mod.make_proposal(self.inventory,profile='balanced',count=1,bricks=2,repeats=2,cycles=8)
        self.assertFalse(mod.check_proposal(replace(p,resident_brick_ceiling=10**30),self.inventory)[0])

    def test_absent_requirement_rejected(self):
        from dataclasses import replace
        p=mod.make_proposal(self.inventory,profile='minimal',count=3,bricks=1,repeats=1,cycles=4)
        p=replace(p,task_ids=(p.task_ids[-1],))
        self.assertFalse(mod.check_proposal(p,self.inventory)[0])

    def test_manual_only_not_in_shadow_plan(self):
        report=mod.evolve(self.inventory,250,42)
        selected=set(report['candidate']['task_ids'])
        self.assertTrue(selected.isdisjoint(report['blocked_manual_tasks']))
        self.assertIn('dm_pcs_private_review',report['blocked_manual_tasks'])
        self.assertIn('clinical_release',report['blocked_manual_tasks'])
        self.assertIn('vehicle_release',report['blocked_manual_tasks'])

    def test_receipt_inventory_mismatch(self):
        report=mod.evolve(self.inventory,10,1)
        altered=copy.deepcopy(self.inventory)
        altered['as_of']='2077-01-01'
        ok,issues=mod.verify_receipt(report,altered)
        self.assertFalse(ok)
        self.assertTrue(any('inventory identity' in s for s in issues))

    def test_plan_can_be_exported_and_verified(self):
        p=mod.evolve(self.inventory,12,7)
        with tempfile.TemporaryDirectory() as td:
            f=pathlib.Path(td)/'candidate.json'
            f.write_text(json.dumps(p))
            ok,errors=mod.verify_receipt(json.loads(f.read_text()),self.inventory)
        self.assertTrue(ok,errors)

if __name__=='__main__':
    unittest.main(verbosity=2)
