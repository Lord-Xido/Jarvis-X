import argparse
import unittest
from pathlib import Path
from swarm import *

class Tests(unittest.TestCase):
    def test_address(self):
        for i in range(1000):
            p=(L-1-i,i,(i*123456789)%L)
            self.assertEqual(coordinate(address(p)),p)
        self.assertIn((L-1,0,0),neighbors((0,0,0)))
        with self.assertRaises(ValueError): address((L,0,0))

    def test_kinetics(self):
        states={(x,y,z):"12" for x in range(2) for y in range(2) for z in range(2)}
        a=Swarm(states,fixture_backend,fixture_score)
        b=Swarm(states,fixture_backend,fixture_score,workers=1)
        prior=8*144
        commits=0
        for i in range(1000):
            r=a.step()
            self.assertEqual(r,b.step())
            self.assertEqual(r["neighbor_messages"],24)
            self.assertLessEqual(r["score"],prior)
            prior=r["score"]
            commits+=r["commits"]
        self.assertEqual(prior,0)
        self.assertEqual(commits,96)
        self.assertTrue(all(len(n.memory)<=4 for n in a.nodes.values()))
        print("1000 rounds; 8000 calls; 96 commits; 7904 rollbacks; score=0")

    def test_failure(self):
        p=(0,0,0)
        for bad in (lambda r:"999",lambda r:"",lambda r:"x"*257,lambda r:1):
            a=Swarm({p:"2"},bad,fixture_score)
            before=a.snapshot()
            self.assertEqual(a.step()["commits"],0)
            self.assertEqual(a.snapshot(),before)
        def fail(r): raise RuntimeError("fixture")
        a=Swarm({p:"2"},fail,fixture_score)
        self.assertEqual(a.step()["rollbacks"],1)
        a=Swarm({p:"2"},fixture_backend,fixture_score)
        before=a.snapshot()
        with self.assertRaises(FileNotFoundError):
            a.step(svm=Path("/nonexistent/omega36"))
        self.assertEqual(a.snapshot(),before)
        self.assertEqual(a.tick,0)

def integration(svm):
    a=Swarm({(0,0,0):"3",(L-1,0,0):"2"},fixture_backend,fixture_score)
    r=a.step(svm=svm)
    assert r["commits"]==2 and r["neighbor_messages"]==2
    assert r["archive"]["payload_sha256"]==r["state_sha256"]
    print(r)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--svm",type=Path,required=True)
    args=parser.parse_args()
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Tests)
    if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
        raise SystemExit(1)
    integration(args.svm)
