"""Independent cross-language JX8KT1 regression and failure tests."""
import hashlib,json,shutil,subprocess,tempfile,unittest
from pathlib import Path
from verify_packet import MAGIC,decode,morton3,verify
from ctr_proposal import build_proposal

ROOT=Path(__file__).resolve().parent
RAW_SHA="96cd4f53f3e83a7e20ad2f1f57b09e3625fb6021bdeffb8313c42d9c39eb1c56"
PACKET_SHA="cb3a4cc9254bd9f93971007a5437629e2b4d580e027716094788fabe395d327b"

class StackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler=shutil.which("g++") or shutil.which("clang++")
        if not compiler:raise RuntimeError("C++17 compiler required")
        cls.tmp=tempfile.TemporaryDirectory()
        cls.base=Path(cls.tmp.name)
        cls.exe=cls.base/"jx8k"
        subprocess.run([compiler,"-O2","-std=c++17","-Wall","-Wextra","-Wpedantic",
                        str(ROOT/"jarvis_x_8k_virtual.cpp"),"-o",str(cls.exe)],check=True)
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()
    def run_engine(self,name,rounds=2,budget=8,candidates=32):
        folder=self.base/name;folder.mkdir()
        report,raw,packet=(folder/n for n in ("report.json","raw.bin","tile.jx8k"))
        subprocess.run([str(self.exe),"--rounds",str(rounds),"--budget",str(budget),
                        "--candidates",str(candidates),"--seed","42",
                        "--report",str(report),"--tiles",str(raw),"--encoded",str(packet)],
                       check=True,capture_output=True,text=True)
        return json.loads(report.read_text()),packet.read_bytes(),raw.read_bytes()
    def test_exact_replay(self):
        a,pa,ra=self.run_engine("replay_a")
        b,pb,rb=self.run_engine("replay_b")
        self.assertEqual((pa,ra,a["observations"]),(pb,rb,b["observations"]))
        self.assertEqual(verify(a,pa,ra)["verified_tiles"],16)
        self.assertEqual(len(pa),8+16*8+a["encoded_bytes"])
    def test_corruption_rejected(self):
        r,p,raw=self.run_engine("corruption")
        altered=bytearray(p);altered[-1]^=1
        for bad in (b"badmagic"+p[8:],p[:-1],p+b"tail",bytes(altered)):
            with self.assertRaises(ValueError):verify(r,bad,raw)
        with self.assertRaises(ValueError):verify(r,p,raw[:-1])
    def test_morton_and_raw_mode(self):
        self.assertEqual(morton3(8191,8191,8191),(1<<39)-1)
        self.assertEqual(morton3(0,0,0),0)
        with self.assertRaises(ValueError):morton3(8192,0,0)
        with self.assertRaises(ValueError):decode(b"\xff")
        with self.assertRaises(ValueError):decode(b"\0")
        self.assertEqual(len(MAGIC),8)
    def test_budget_fail_closed(self):
        for flags in (["--rounds","0"],["--rounds","17"],["--budget","257"],
                      ["--rounds","16","--budget","256"],["--budget","32","--candidates","8"]):
            r=subprocess.run([str(self.exe),*flags],capture_output=True,text=True)
            self.assertNotEqual(r.returncode,0,msg=str(flags))
    def test_receipt_only_provisional(self):
        r,p,raw=self.run_engine("proposal")
        receipt=verify(r,p,raw)
        proposal=build_proposal(receipt,r)
        self.assertEqual(proposal["authority"],"provisional")
        self.assertFalse(proposal["commit_eligible"])
        self.assertFalse(proposal["production_state_modified"])
        self.assertEqual(proposal["gates"]["external_world_correspondence"],"not_evaluated")
        for bad in ({**receipt,"status":"FAIL"},{**receipt,"packet_sha256":"bad"},
                    {**receipt,"authority":"authoritative"}):
            with self.assertRaises(ValueError):build_proposal(bad,r)
        with self.assertRaises(ValueError):build_proposal(receipt,{**r,"world_side":4000})
    def test_192_tile_fixture(self):
        r,p,raw=self.run_engine("fixture",rounds=3,budget=64,candidates=256)
        self.assertEqual(r["tiles_verified"],192)
        self.assertEqual(len(raw),196608)
        self.assertEqual(len(p),92144)
        self.assertEqual(hashlib.sha256(raw).hexdigest(),RAW_SHA)
        self.assertEqual(hashlib.sha256(p).hexdigest(),PACKET_SHA)
        self.assertEqual(verify(r,p,raw)["verified_tiles"],192)

if __name__=="__main__":unittest.main()
