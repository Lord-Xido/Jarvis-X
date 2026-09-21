import unittest

import torch

from multimedia_inward_engine import (
    Inward3DConfig,
    InwardMultimedia3DANN,
    MultimodalBatch,
)


class MultimediaInwardEngineTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        self.cfg = Inward3DConfig(
            base_res=4,
            image_size=16,
            video_frames=2,
            video_size=16,
            audio_samples=128,
            vocab_size=64,
            text_length=4,
            text_model_dim=32,
            vector_hidden=32,
            recursive_steps=1,
            inner_steps=1,
        )
        self.engine = InwardMultimedia3DANN(self.cfg)

    def batch(self):
        cfg = self.cfg
        return MultimodalBatch(
            image=torch.randn(1, 3, 16, 16),
            video=torch.randn(1, 3, 2, 16, 16),
            audio_features=torch.randn(1, cfg.audio_features),
            text_features=torch.randn(1, cfg.text_features),
            sensor=torch.randn(1, cfg.sensor_features),
            text_tokens=torch.randint(0, cfg.vocab_size, (1, cfg.text_length)),
        )

    def test_end_to_end_shapes(self):
        output = self.engine(self.batch())
        self.assertEqual(tuple(output.field.shape), (1, self.cfg.field_channels, 4, 4, 4))
        self.assertEqual(tuple(output.image.shape), (1, 3, 16, 16))
        self.assertEqual(tuple(output.video.shape), (1, 3, 2, 16, 16))
        self.assertEqual(tuple(output.audio.shape), (1, 128))
        self.assertEqual(tuple(output.text_logits.shape), (1, 4, 64))
        self.assertEqual(tuple(output.sensor.shape), (1, self.cfg.sensor_features))
        self.assertTrue(torch.isfinite(output.convergence).all())

    def test_loss_backpropagates(self):
        batch = self.batch()
        output = self.engine(batch)
        loss, terms = self.engine.loss(batch, output)
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("cycle", terms)
        self.assertIn("fixed_point", terms)
        loss.backward()
        self.assertTrue(any(p.grad is not None for p in self.engine.parameters()))

    def test_generation_without_input_media(self):
        output = self.engine.generate(batch_size=2, recursive_steps=1)
        self.assertEqual(output.image.shape[0], 2)
        self.assertEqual(output.video.shape[0], 2)
        self.assertEqual(output.audio.shape[0], 2)


if __name__ == "__main__":
    unittest.main()
