import unittest

from dr_moagi_vnext import Config, demo_data
from evolve_architecture import evolve_architecture


class ArchitectureEvolutionTests(unittest.TestCase):
    def test_bounded_evolution_and_selection(self):
        cfg = Config(edge=8, tiles=4, levels=2, hidden=6, train_steps=3)
        selected, winner, evaluations = evolve_architecture(
            cfg, demo_data(cfg), 0.01, 4
        )
        self.assertIn("validation_packet_bytes", winner)
        self.assertEqual(len(evaluations), 4)
        self.assertIn(selected.levels, (1, 2))
        self.assertIn(selected.hidden, (4, 6))
        self.assertTrue(winner["meets_target"])

    def test_candidate_budget(self):
        cfg = Config(edge=8, tiles=4, levels=2, hidden=6, train_steps=0)
        with self.assertRaises(ValueError):
            evolve_architecture(cfg, demo_data(cfg), 0.01, 21)


if __name__ == "__main__":
    unittest.main()
