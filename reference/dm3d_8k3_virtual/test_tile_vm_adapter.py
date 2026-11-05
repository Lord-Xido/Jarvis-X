"""Finite typed tile-to-VM adapter tests independent of production DM3D."""
import json
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import unittest
from array import array

from ctr_proposal import build_proposal
from verify_packet import verify

from tile_vm_adapter import (
    BoundedTileVM, DEFAULT_PROGRAM, MAX_TILES, TileOp,
    default_bytecode, pack_tensor, pack_word, parse_program, unpack_tensor,
)

ROOT = Path(__file__).resolve().parent


class AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("g++") or shutil.which("clang++")
        if compiler is None:
            raise RuntimeError("C++17 compiler required")
        cls.tmp = tempfile.TemporaryDirectory()
        cls.base = Path(cls.tmp.name)
        cls.binary = cls.base / "reference"
        subprocess.run([compiler, "-O2", "-std=c++17", str(ROOT / "jarvis_x_8k_virtual.cpp"),
                        "-o", str(cls.binary)], check=True)
        cls.report = cls.base / "report.json"
        cls.packet = cls.base / "encoded.jx8k"
        cls.raw = cls.base / "raw.bin"
        subprocess.run([str(cls.binary), "--rounds", "1", "--budget", "8",
                        "--candidates", "24", "--seed", "42",
                        "--report", str(cls.report), "--encoded", str(cls.packet),
                        "--tiles", str(cls.raw)], check=True, capture_output=True)
        cls.report_data = json.loads(cls.report.read_text())
        cls.packet_data = cls.packet.read_bytes()
        cls.raw_data = cls.raw.read_bytes()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_float32_round_trip(self):
        source = bytes((0xE4 for _ in range(1024)))
        values = unpack_tensor(source)
        self.assertEqual(values.typecode, "f")
        self.assertEqual(len(values), 4096)
        self.assertEqual(list(values[:8]), [0., 1., 2., 3., 0., 1., 2., 3.])
        self.assertEqual(pack_tensor(values), source)
        values[3] = 0.5
        with self.assertRaises(ValueError):
            pack_tensor(values)
        with self.assertRaises(ValueError):
            pack_tensor(array("f", [0.] * 40))

    def test_bytecode_profile_safeguards(self):
        self.assertEqual(parse_program(default_bytecode()), DEFAULT_PROGRAM)
        self.assertEqual(len(default_bytecode()), 56)
        invalid = [b"", b"\x00", default_bytecode()[:-8],
                   default_bytecode() + struct.pack(">Q", 0xFF << 56),
                   struct.pack(">Q", 0x76 << 56) + default_bytecode()[8:],
                   struct.pack(">Q", pack_word(TileOp.LOAD_TILE, imm=1)) + default_bytecode()[8:]]
        for program in invalid:
            with self.subTest(program=program.hex()[:20]), self.assertRaises(ValueError):
                parse_program(program)
        with self.assertRaises(ValueError):
            pack_word(TileOp.LOAD_TILE, dst=256)

    def test_typed_vm_provisional_receipt(self):
        vm = BoundedTileVM(default_bytecode())
        result = vm.run(self.report_data, self.packet_data, self.raw_data)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["tiles_processed"], 8)
        self.assertEqual(result["physical_tensor_values"], 32768)
        self.assertEqual(result["instruction_steps"], 49)
        self.assertEqual(result["float32_working_set_bytes_per_tile"], 16384)
        self.assertEqual(result["authority"], "provisional")
        self.assertFalse(result["commit_eligible"])
        self.assertFalse(result["production_state_modified"])
        self.assertEqual(result["gates"]["external_world"], "not_evaluated")
        for observation in result["observations"]:
            self.assertEqual(sum(observation["counts"]), 4096)
            self.assertEqual(observation["tensor_shape"], [16, 16, 16])
            self.assertTrue(observation["bit_exact"])
            self.assertLess(observation["morton39"], 1 << 39)

    def test_fail_closed_corruption_or_untrusted_metadata(self):
        vm = BoundedTileVM(default_bytecode())
        packet = bytearray(self.packet_data)
        packet[-1] ^= 1
        for p, r, metadata in (
                (bytes(packet), self.raw_data, self.report_data),
                (self.packet_data, self.raw_data[:-1], self.report_data),
                (self.packet_data, self.raw_data, {**self.report_data, "tiles_verified": MAX_TILES + 1}),
                (self.packet_data + b"garbage", self.raw_data, self.report_data),
        ):
            with self.subTest(case=len(p)), self.assertRaises((ValueError, IndexError)):
                vm.run(metadata, p, r)

    def test_ctr_receipt_chain_and_corruption_rejection(self):
        vm = BoundedTileVM(default_bytecode())
        receipt = verify(self.report_data, self.packet_data, self.raw_data)
        result = vm.run(self.report_data, self.packet_data, self.raw_data)
        proposal = build_proposal(receipt, self.report_data, result)
        self.assertEqual(proposal["gates"]["tile_vm_staging"], "pass")
        self.assertFalse(proposal["commit_eligible"])
        self.assertFalse(proposal["production_state_modified"])
        self.assertEqual(proposal["typed_state"]["vm_staging"]["tiles_processed"], 8)
        with self.assertRaises(ValueError):
            build_proposal(receipt, self.report_data, {**result, "packet_sha256": "0" * 64})
        with self.assertRaises(ValueError):
            build_proposal(receipt, self.report_data, {**result, "instruction_steps": 999999})
        with self.assertRaises(ValueError):
            build_proposal(receipt, self.report_data, {**result, "observations": []})

    def test_cli_emits_only_provisional_receipt(self):
        outfile = self.base / "proposal.json"
        process = subprocess.run(["python", str(ROOT / "tile_vm_adapter.py"),
                                  "--report", str(self.report),
                                  "--encoded", str(self.packet),
                                  "--raw", str(self.raw),
                                  "--receipt", str(outfile)],
                                 check=True, capture_output=True, text=True)
        self.assertEqual(json.loads(outfile.read_text())["commit_eligible"], False)
        self.assertIn('"status": "PASS"', process.stdout)


if __name__ == "__main__":
    unittest.main()
