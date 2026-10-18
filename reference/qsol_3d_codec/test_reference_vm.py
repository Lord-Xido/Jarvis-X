"""Conformance tests for the QSOL 3D codec reference VM.

Run directly from the repository root:

    python reference/qsol_3d_codec/test_reference_vm.py
"""

import unittest

from reference_vm import (
    CONST_POOL,
    INT32_MAX,
    PROGRAM_CANONICAL_V1,
    PROGRAM_SUBMITTED_DRAFT,
    Q16_ONE,
    Instruction,
    Opcode,
    QSOL3DVM,
    audit_submitted_register_bytes,
    iter_instructions,
    q16_mul,
)


class QSOL3DCodecTests(unittest.TestCase):
    def test_instruction_is_exactly_four_bytes(self) -> None:
        instruction = Instruction(0x04, 0x40, 0x30, 0x40)
        self.assertEqual(instruction.encode(), bytes.fromhex("04 40 30 40"))
        self.assertEqual(Instruction.decode(instruction.encode()), instruction)
        with self.assertRaises(ValueError):
            Instruction.decode(b"\x00\x00\x00")

    def test_constant_selectors_expand_to_q16_values(self) -> None:
        self.assertEqual(CONST_POOL[0x10], 0x00010000)
        self.assertEqual(CONST_POOL[0x33], 0x00003333)
        self.assertEqual(CONST_POOL[0x40], 0x00004000)
        self.assertEqual(CONST_POOL[0x9A], 0x0000D99A)

    def test_submitted_stream_is_preserved_but_not_canonical(self) -> None:
        self.assertNotEqual(PROGRAM_SUBMITTED_DRAFT, PROGRAM_CANONICAL_V1)
        self.assertEqual(
            audit_submitted_register_bytes(),
            [
                "word 4: source 0x48 -> 0x30",
                "word 5: source 0x64 -> 0x40",
                "word 6: source 0x64 -> 0x40",
                "word 7: source 0x96 -> 0x60",
            ],
        )

    def test_canonical_register_chain_uses_actual_byte_ids(self) -> None:
        instructions = list(iter_instructions(PROGRAM_CANONICAL_V1))
        self.assertEqual((instructions[3].dest, instructions[3].src), (0x30, 0x20))
        self.assertEqual((instructions[4].dest, instructions[4].src), (0x40, 0x30))
        self.assertEqual((instructions[5].dest, instructions[5].src), (0x50, 0x40))
        self.assertEqual((instructions[6].dest, instructions[6].src), (0x60, 0x40))
        self.assertEqual((instructions[7].dest, instructions[7].src), (0x70, 0x60))
        self.assertEqual((instructions[8].dest, instructions[8].src), (0x10, 0x70))

    def test_toroidal_fetch_wraps_all_six_axes(self) -> None:
        vm = QSOL3DVM((3, 3, 3))
        for x, y, z in vm.coords:
            vm.registers[(x, y, z)][0x10] = x + 10 * y + 100 * z

        vm.execute(Instruction(Opcode.FETCH_NB, 0x20, 0x10, 0x06), cycle=1, pc=0)
        self.assertEqual(vm.registers[(0, 0, 0)][0x20], (1, 2, 10, 20, 100, 200))

    def test_normalized_laplacian_is_neighbor_mean_minus_center(self) -> None:
        vm = QSOL3DVM((1, 1, 1))
        vm.registers[(0, 0, 0)][0x10] = Q16_ONE
        vm.registers[(0, 0, 0)][0x20] = (2 * Q16_ONE,) * 6
        vm.execute(Instruction(Opcode.LAPLACE_3D, 0x30, 0x20, 0x10), cycle=1, pc=0)
        self.assertEqual(vm.registers[(0, 0, 0)][0x30], Q16_ONE)

    def test_q16_multiply_and_saturation(self) -> None:
        self.assertEqual(q16_mul(Q16_ONE, CONST_POOL[0x40]), 0x00004000)
        self.assertEqual(q16_mul(INT32_MAX, 2 * Q16_ONE), INT32_MAX)

    def test_hamilton_proxy_is_half_x_squared(self) -> None:
        vm = QSOL3DVM((1, 1, 1))
        vm.registers[(0, 0, 0)][0x40] = Q16_ONE
        vm.execute(Instruction(Opcode.HAMILTON_CHK, 0x50, 0x40, 0x00), cycle=1, pc=0)
        self.assertEqual(vm.registers[(0, 0, 0)][0x50], Q16_ONE // 2)

    def test_sync_commit_residual_add_is_atomic_state_update(self) -> None:
        vm = QSOL3DVM((1, 1, 1))
        vm.registers[(0, 0, 0)][0x10] = Q16_ONE
        vm.registers[(0, 0, 0)][0x70] = Q16_ONE // 4
        vm.execute(Instruction(Opcode.SYNC_COMMIT, 0x10, 0x70, 0x01), cycle=1, pc=0)
        self.assertEqual(vm.registers[(0, 0, 0)][0x10], Q16_ONE + Q16_ONE // 4)
        self.assertEqual(vm.last_commit_drift, Q16_ONE // 4)
        self.assertEqual(vm.commit_generation, 1)

    def test_actuate_e_is_internal_only_register_copy(self) -> None:
        vm = QSOL3DVM((1, 1, 1))
        vm.registers[(0, 0, 0)][0x60] = 0x00001234
        vm.execute(Instruction(Opcode.ACTUATE_E, 0x70, 0x60, 0x00), cycle=1, pc=0)
        self.assertEqual(vm.registers[(0, 0, 0)][0x70], 0x00001234)
        self.assertEqual(vm.actuation_register[(0, 0, 0)], 0x00001234)

    def test_uniform_canonical_program_is_zero_drift_fixed_point(self) -> None:
        vm = QSOL3DVM((3, 3, 3))
        trace = vm.run_once()

        self.assertEqual(len(trace), 10)
        self.assertEqual([record.pc for record in trace], list(range(0x00, 0x28, 0x04)))
        self.assertEqual(trace[-1].opcode, Opcode.HALT_LOOP)
        self.assertTrue(vm.halted)
        self.assertEqual(vm.commit_generation, 1)
        self.assertEqual(vm.last_commit_drift, 0)

        for coord in vm.coords:
            self.assertEqual(vm.registers[coord][0x10], 0x00010000)
            self.assertEqual(vm.registers[coord][0x11], 0x00003333)
            self.assertEqual(vm.registers[coord][0x30], 0)
            self.assertEqual(vm.registers[coord][0x40], 0)
            self.assertEqual(vm.registers[coord][0x50], 0)
            self.assertEqual(vm.registers[coord][0x60], 0)
            self.assertEqual(vm.registers[coord][0x70], 0)
            self.assertEqual(vm.actuation_register[coord], 0)

    def test_invalid_fetch_neighbor_count_fails_closed(self) -> None:
        vm = QSOL3DVM((1, 1, 1))
        with self.assertRaises(ValueError):
            vm.execute(Instruction(Opcode.FETCH_NB, 0x20, 0x10, 0x05), cycle=1, pc=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
