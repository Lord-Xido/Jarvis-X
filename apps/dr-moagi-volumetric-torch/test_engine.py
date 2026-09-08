import unittest

import torch

from engine import (
    AdaptiveVoxelGrid3D,
    DynamicHyperNetwork3D,
    Multimodal3DProjectionAdapter,
    VolumetricMultimodalInterpreter,
)


class EngineTests(unittest.TestCase):
    def test_zero_flow_is_identity(self):
        shell = AdaptiveVoxelGrid3D(8, 16)
        source = torch.randn(1, 16, 8, 8, 8)
        flow = torch.zeros(1, 3, 8, 8, 8)
        warped = shell.warp(flow, grid=source)
        self.assertTrue(torch.allclose(source, warped, atol=1e-5, rtol=1e-5))

    def test_multimodal_projection_shapes(self):
        adapter = Multimodal3DProjectionAdapter(target_res=8, target_ch=16)
        image = adapter.project_image_2d(torch.randn(1, 3, 32, 32))
        audio = adapter.project_audio_1d(torch.randn(1, 128))
        video = adapter.project_video_3d(torch.randn(1, 3, 4, 16, 16))
        fused = adapter.fuse([image, audio, video])
        self.assertEqual(tuple(fused.shape), (1, 16, 8, 8, 8))

    def test_dynamic_hypernetwork_applies_per_sample_kernel(self):
        net = DynamicHyperNetwork3D(latent_dim=128 * 8)
        x = torch.randn(2, 32, 4, 4, 4)
        core = torch.randn(2, 128, 2, 2, 2)
        out = net(x, core)
        self.assertEqual(tuple(out.shape), tuple(x.shape))

    def test_forward_and_optimization_are_finite(self):
        torch.manual_seed(7)
        model = VolumetricMultimodalInterpreter(base_res=8)
        target = torch.randn(1, 16, 8, 8, 8)
        model.shells["outer"].set_grid(target * 0.9)
        pred, mid, inner, core = model(target, recursive_depth=1, commit_state=False)
        self.assertEqual(tuple(pred.shape), (1, 16, 8, 8, 8))
        self.assertEqual(tuple(mid.shape), (1, 32, 4, 4, 4))
        self.assertEqual(tuple(inner.shape), (1, 64, 2, 2, 2))
        self.assertEqual(tuple(core.shape), (1, 128, 2, 2, 2))
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        loss = model.optimize_step(target, optimizer, recursive_depth=1, use_ode_solver=False)
        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        self.assertIn("warp_reconstruction_loss", model.last_metrics)


if __name__ == "__main__":
    unittest.main()
