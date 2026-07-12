import json
import unittest

import numpy as np

from vann_rom_sdk import (
    Address3D,
    Assembler,
    Instruction,
    Opcode,
    TinyAutoencoder,
    VANNVirtualMachine,
    VoxelPage,
)
from vann_rom_sdk.demo import DEMO_SOURCE, run_demo
from vann_rom_sdk.geometry import morton3d_decode, morton3d_encode


class SDKTests(unittest.TestCase):
    def test_instruction_round_trip(self):
        instruction = Instruction(opcode=Opcode.ENCODE3D, immediate=-23, geo=123456)
        self.assertEqual(Instruction.decode(instruction.encode()), instruction)

    def test_instruction_crc_rejects_tampering(self):
        encoded = bytearray(Instruction(opcode=Opcode.ENCODE3D).encode())
        encoded[4] ^= 0x01
        with self.assertRaisesRegex(ValueError, "CRC"):
            Instruction.decode(bytes(encoded))

    def test_instruction_stream_round_trip(self):
        instructions = [
            Instruction(opcode=Opcode.LOAD_INPUT),
            Instruction(opcode=Opcode.HALT),
        ]
        image = Instruction.encode_stream(instructions)
        self.assertEqual(len(image), 32)
        self.assertEqual(Instruction.decode_stream(image), instructions)

    def test_morton_round_trip(self):
        point = (123, 456, 789)
        self.assertEqual(morton3d_decode(morton3d_encode(*point)), point)

    def test_rom_is_sealed_and_detects_tampering(self):
        program = Assembler().assemble("LOAD_INPUT\nHALT")
        vm = VANNVirtualMachine(TinyAutoencoder(4, 2))
        vm.load_program(program.instructions)
        self.assertTrue(vm.rom.sealed)
        with self.assertRaisesRegex(RuntimeError, "sealed"):
            vm.rom.map_page(VoxelPage(address=Address3D(99, 0, 0)))

        page = next(iter(vm.rom))
        page.metadata["tampered"] = True
        with self.assertRaisesRegex(RuntimeError, "integrity"):
            vm.rom.fetch_instruction(page.address)

    def test_demo_vm(self):
        program = Assembler().assemble(DEMO_SOURCE)
        vm = VANNVirtualMachine(TinyAutoencoder(12, 4))
        vm.load_program(program.instructions)
        vm.set_input(np.linspace(0.0, 1.0, 12, dtype=np.float32))
        result = vm.run()
        self.assertTrue(result.halted)
        self.assertEqual(result.cycles, len(program.instructions))
        self.assertIsNotNone(result.output)
        self.assertEqual(len(result.output[0]), 12)
        self.assertEqual(result.metrics["commits"], 2)
        self.assertEqual(result.metrics["rollbacks"], 0)
        self.assertGreaterEqual(
            sum(entry["event"] == "commit" for entry in result.journal),
            2,
        )

    def test_uncommitted_training_is_discarded(self):
        source = """
LOAD_INPUT
NORMALIZE
ENCODE3D
PREDICT
COMPARE
UPDATE_OMEGA
HALT
"""
        model = TinyAutoencoder(6, 2)
        before = model.state_dict()
        vm = VANNVirtualMachine(model)
        vm.load_program(Assembler().assemble(source).instructions)
        vm.set_input(np.linspace(0.0, 1.0, 6, dtype=np.float32))
        result = vm.run()

        after = model.state_dict()
        for name in ("w_enc", "b_enc", "w_dec", "b_dec"):
            np.testing.assert_array_equal(before[name], after[name])
        self.assertEqual(result.metrics["commits"], 0)
        self.assertEqual(result.metrics["rollbacks"], 1)

    def test_commit_cannot_bypass_projection_or_verification(self):
        source = """
LOAD_INPUT
NORMALIZE
ENCODE3D
PREDICT
COMPARE
UPDATE_OMEGA
COMMIT
DECODE3D
STAGE
COMMIT
HALT
"""
        vm = VANNVirtualMachine(TinyAutoencoder(6, 2))
        vm.load_program(Assembler().assemble(source).instructions)
        vm.set_input(np.linspace(0.0, 1.0, 6, dtype=np.float32))
        result = vm.run()

        self.assertEqual(result.metrics["commits"], 2)
        self.assertEqual(result.metrics["rollbacks"], 0)
        self.assertTrue(np.all(np.abs(vm.omega.working_bias) <= 0.25))
        self.assertIsNotNone(result.output)
        output = np.asarray(result.output)
        self.assertTrue(np.all((0.0 <= output) & (output <= 1.0)))

    def test_demo_report_is_json_serializable(self):
        report = run_demo()
        encoded = json.dumps(report)
        self.assertIn('"halted": true', encoded)
        self.assertTrue(report["rom"]["sealed"])


if __name__ == "__main__":
    unittest.main()
