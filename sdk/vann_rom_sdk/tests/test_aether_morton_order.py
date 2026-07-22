import unittest

from vann_rom_sdk.aether import AetherConfig, AetherEngine, synthetic_aether_input


class MortonOrderingAuditTests(unittest.TestCase):
    def test_generated_morton_keys_are_monotone_without_unsigned_wraparound(self):
        for seed in range(20):
            with self.subTest(seed=seed):
                result = AetherEngine(
                    AetherConfig(
                        hidden_dim=12,
                        latent_dim=6,
                        max_tokens=96,
                        seed=seed,
                    )
                ).run(synthetic_aether_input(seed))
                keys = [int(value) for value in result.field.morton_keys]
                self.assertEqual(keys, sorted(keys))


if __name__ == "__main__":
    unittest.main()
