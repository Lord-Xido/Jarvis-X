from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from reference import (  # noqa: E402
    DEFAULT_PASSES,
    EPS_Q16,
    INT32_MAX,
    INT32_MIN,
    InwardRecursiveROM,
    mask_le_nonnegative,
    q16_from_float,
    q16_mul,
    q16_to_float,
    select_mask,
)


class Q16Tests(unittest.TestCase):
    def test_widened_multiply_matches_expected_q16(self) -> None:
        self.assertEqual(q16_from_float(0.25), q16_mul(q16_from_float(0.5), q16_from_float(0.5)))
        self.assertEqual(q16_from_float(-0.25), q16_mul(q16_from_float(-0.5), q16_from_float(0.5)))

    def test_multiply_saturates_after_widened_product(self) -> None:
        self.assertEqual(INT32_MAX, q16_mul(INT32_MAX, INT32_MAX))
        self.assertEqual(INT32_MIN, q16_mul(INT32_MIN, INT32_MAX))

    def test_inclusive_gap_comparator(self) -> None:
        self.assertEqual(-1, mask_le_nonnegative(EPS_Q16 - 1, EPS_Q16))
        self.assertEqual(-1, mask_le_nonnegative(EPS_Q16, EPS_Q16))
        self.assertEqual(0, mask_le_nonnegative(EPS_Q16 + 1, EPS_Q16))

    def test_mask_select_freezes_old_value(self) -> None:
        self.assertEqual(123, select_mask(-1, 123, 456))
        self.assertEqual(456, select_mask(0, 123, 456))


class RecursiveROMTests(unittest.TestCase):
    def test_fixed_work_budget_is_always_executed(self) -> None:
        engine = InwardRecursiveROM(passes=32)
        result = engine.run_float((0.4, 0.2, 0.1), (0.7, 0.35, 0.175))
        self.assertEqual(32, len(result.trace))

    def test_deterministic_fixture_converges_inside_default_budget(self) -> None:
        engine = InwardRecursiveROM()
        result = engine.run_float((0.4, 0.2, 0.1), (0.7, 0.35, 0.175))
        self.assertEqual(DEFAULT_PASSES, len(result.trace))
        self.assertTrue(result.converged)
        self.assertIsNotNone(result.first_lock_iteration)
        self.assertLess(result.first_lock_iteration, DEFAULT_PASSES)
        self.assertLessEqual(result.gap, EPS_Q16)
        self.assertGreater(result.trace[0].gap, result.gap)

    def test_lock_is_sticky_and_freezes_state_and_weights(self) -> None:
        result = InwardRecursiveROM().run_float((0.4, 0.2, 0.1), (0.7, 0.35, 0.175))
        first = result.first_lock_iteration
        self.assertIsNotNone(first)
        locked = result.trace[first]
        for row in result.trace[first:]:
            self.assertEqual(-1, row.lock_mask)
            self.assertEqual(locked.state, row.state)
            self.assertEqual(locked.weights, row.weights)
            self.assertEqual(locked.gap, row.gap)

    def test_anchor_is_immutable_and_state_finishes_near_it(self) -> None:
        result = InwardRecursiveROM().run_float((0.4, 0.2, 0.1), (0.7, 0.35, 0.175))
        for state_i, anchor_i in zip(result.state, result.anchor):
            # The component-wise distance is necessarily <= the L1 gap bound.
            self.assertLessEqual(abs(state_i - anchor_i), EPS_Q16)

    def test_nonconvergence_is_reported_not_relabelled(self) -> None:
        engine = InwardRecursiveROM(passes=1, eps_q16=0)
        result = engine.run_float((0.4, 0.2, 0.1), (0.1, 0.1, 0.1))
        self.assertFalse(result.converged)
        self.assertIsNone(result.first_lock_iteration)
        self.assertGreater(result.gap, 0)

    def test_result_is_reproducible(self) -> None:
        engine = InwardRecursiveROM()
        a = engine.run_float((0.4, 0.2, 0.1), (0.7, 0.35, 0.175))
        b = engine.run_float((0.4, 0.2, 0.1), (0.7, 0.35, 0.175))
        self.assertEqual(a, b)

    def test_reported_gap_has_q16_scale(self) -> None:
        result = InwardRecursiveROM().run_float((0.4, 0.2, 0.1), (0.7, 0.35, 0.175))
        self.assertLessEqual(q16_to_float(result.gap), q16_to_float(EPS_Q16))


class ROMSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = (HERE / "inward_recursive_rom.asm").read_text(encoding="utf-8")

    def test_runtime_control_has_no_jump_or_conditional_branch(self) -> None:
        op_lines = []
        for raw in self.source.splitlines():
            line = raw.split(";", 1)[0].strip()
            if line and not line.startswith(".") and not line.endswith(":"):
                op_lines.append(line)
        program = "\n".join(op_lines)
        self.assertIsNone(re.search(r"\b(?:JMP|JZ|JNZ|JE|JNE|BRA|BRANCH|LOOP)\b", program, flags=re.I))

    def test_recursion_is_compile_time_unrolled(self) -> None:
        self.assertIn(".REPT PASS_COUNT", self.source)
        self.assertIn(".equ PASS_COUNT,  192", self.source)
        self.assertEqual(1, len(re.findall(r"^\s*HALT\s*$", self.source, flags=re.M)))

    def test_accumulator_is_reset_inside_each_pass(self) -> None:
        repeat_body = self.source.split(".REPT PASS_COUNT", 1)[1].split(".ENDR", 1)[0]
        self.assertIn("MOV_IMM R12, 0x00000000", repeat_body)

    def test_anchor_is_written_before_repeat_and_not_rewritten_inside(self) -> None:
        before, body = self.source.split(".REPT PASS_COUNT", 1)
        self.assertIn("QSTORE    R0, R5, ANCHOR_X", before)
        self.assertIn("QSTORE    R1, R5, ANCHOR_Y", before)
        self.assertIn("QSTORE    R2, R5, ANCHOR_Z", before)
        self.assertNotRegex(body, r"QSTORE\s+R\d+,\s*R5,\s*ANCHOR_[XYZ]")


if __name__ == "__main__":
    unittest.main()
