import unittest

import numpy as np

from vann_rom_sdk import Assembler, Instruction, Opcode, TinyAutoencoder, VANNVirtualMachine
from vann_rom_sdk.demo import DEMO_SOURCE
from vann_rom_sdk.geometry import morton3d_decode, morton3d_encode


class SDKTests(unittest.TestCase):
    def test_instruction_round_trip(self):
        instruction = Instruction(opcode=Opcode.ENCODE3D, immediate=-23, geo=123456)
        self.assertEqual(Instruction.decode(instruction.encode()), instruction)

    def test_morton_round_trip(self):
        point = (123, 456, 789)
        self.assertEqual(morton3d_decode(morton3d_encode(*point)), point)

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


if __name__ == "__main__":
    unittest.main()
