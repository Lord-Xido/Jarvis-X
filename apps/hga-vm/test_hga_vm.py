import unittest

import numpy as np

from hga_vm import (
    BytecodeCodec,
    Clifford4,
    HGAConfig,
    HGAVM,
    Instruction,
    Opcode,
    ProofStatus,
)


class TestHGAVM(unittest.TestCase):
    def test_clifford_signature(self):
        cliff = Clifford4((1, -1, -1, -1))
        expected = [1.0, -1.0, -1.0, -1.0]
        for i, square in enumerate(expected):
            v = np.zeros(16)
            v[1 << i] = 1.0
            product = cliff.geometric_product(v, v)
            self.assertAlmostEqual(product[0], square)
            self.assertTrue(np.allclose(product[1:], 0.0))

    def test_default_program_is_well_formed(self):
        program = HGAVM.default_program()
        decoded = BytecodeCodec.decode_all(program)
        self.assertEqual(decoded[-1].opcode, Opcode.HALT)
        self.assertEqual(len(program) % 4, 0)

    def test_end_to_end_cycle_proves_and_reduces_loss(self):
        vm = HGAVM()
        history = vm.run(5)
        self.assertTrue(all(row["proof"] == "PROVED" for row in history))
        losses = [row["loss"] for row in history]
        self.assertLess(losses[-1], losses[0])
        self.assertTrue({"ZFC", "PA", "CT_META"}.issubset(vm.state.gamma))

    def test_unknown_preserves_state(self):
        cfg = HGAConfig(rec_epsilon=1e-12, unknown_rec_limit=1.0)
        vm = HGAVM(cfg)
        before = vm.state.psi.copy()
        vm.closed_cycle()
        self.assertEqual(vm.state.proof.status, ProofStatus.UNKNOWN)
        self.assertTrue(np.allclose(vm.state.psi, before))

    def test_disproved_rolls_back_non_finite_candidate(self):
        vm = HGAVM()
        vm.state.gamma.update({"ZFC", "PA", "CT_META"})
        bad = vm.state.clone()
        bad.psi[0] = np.nan
        proof = vm.verify(bad)
        self.assertEqual(proof.status, ProofStatus.DISPROVED)
        self.assertFalse(proof.checks["finite_state"])

    def test_nop_elimination_is_proof_carrying(self):
        program = BytecodeCodec.encode([
            Instruction(Opcode.AXIOM_LOAD, 0),
            Instruction(Opcode.NOP),
            Instruction(Opcode.GEOM_INIT),
            Instruction(Opcode.NOP),
            Instruction(Opcode.HALT),
        ])
        vm = HGAVM(bytecode=program)
        candidate = vm.propose_optimized_bytecode()
        proof = vm.verify_refinement(program, candidate)
        self.assertEqual(proof.status, ProofStatus.PROVED)
        self.assertLess(len(candidate), len(program))
        self.assertEqual(BytecodeCodec.semantic_signature(program), BytecodeCodec.semantic_signature(candidate))

    def test_time_block_residual_contracts(self):
        vm = HGAVM()
        _, residuals = vm.time_block_solve(vm.state, horizon=8, iterations=8)
        self.assertGreater(len(residuals), 1)
        self.assertLess(residuals[-1], residuals[0])

    def test_scheduler_is_acyclic_and_commits_last(self):
        levels = HGAVM.scheduler_levels()
        flat = [name for level in levels for name in level]
        self.assertEqual(len(flat), len(set(flat)))
        self.assertEqual(levels[-1], ["commit"])


if __name__ == "__main__":
    unittest.main()
