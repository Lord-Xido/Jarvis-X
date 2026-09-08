from __future__ import annotations

import math
import unittest

import torch

from engine import (
    MultimodalProjector,
    PermeatedVolumetricInterpreter,
    PermeationOperator,
    UnifiedVoxelField,
)


class UnifiedFieldTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)

    def test_zero_flow_warp_is_identity(self) -> None:
        field = UnifiedVoxelField(8, 16)
        source = torch.randn(1, 16, 8, 8, 8)
        flow = torch.zeros(1, 3, 8, 8, 8)
        warped = field.warp(flow, grid=source)
        self.assertTrue(torch.allclose(source, warped, atol=1e-5, rtol=1e-5))

    def test_multimodal_projectors_fuse_to_one_field(self) -> None:
        projector = MultimodalProjector(
            audio_dim=12,
            feature_dim=10,
            target_res=8,
            total_channels=16,
            seed_res=2,
        )
        image = projector.project_image(torch.randn(2, 3, 20, 20))
        video = projector.project_video(torch.randn(2, 3, 4, 12, 12))
        audio = projector.project_audio(torch.randn(2, 12))
        features = projector.project_features(torch.randn(2, 10))
        fused = projector.fuse([image, video, audio, features])
        self.assertEqual(tuple(fused.shape), (2, 16, 8, 8, 8))
        self.assertTrue(torch.isfinite(fused).all())

    def test_spectral_mixer_matches_rfft_geometry(self) -> None:
        permeator = PermeationOperator(16, 8, spectral_rank=4)
        x = torch.randn(2, 16, 8, 8, 8, requires_grad=True)
        y = permeator(x)
        self.assertEqual(y.shape, x.shape)
        y.square().mean().backward()
        self.assertIsNotNone(permeator.spectral_delta_real.grad)
        self.assertTrue(torch.isfinite(permeator.spectral_delta_real.grad).all())

    def test_forward_without_commit_is_transactional(self) -> None:
        model = PermeatedVolumetricInterpreter(
            res=8,
            outer_ch=4,
            mid_ch=4,
            inner_ch=8,
            spectral_rank=4,
        )
        initial = model.field.grid.clone()
        injection = torch.randn_like(initial)
        outer, mid, inner, core, unified = model(
            injection,
            iterations=2,
            commit_state=False,
        )
        self.assertEqual(tuple(outer.shape), (1, 4, 8, 8, 8))
        self.assertEqual(tuple(mid.shape), (1, 4, 8, 8, 8))
        self.assertEqual(tuple(inner.shape), (1, 8, 8, 8, 8))
        self.assertEqual(tuple(core.shape), (1, 16, 1, 1, 1))
        self.assertEqual(unified.shape, initial.shape)
        self.assertTrue(torch.equal(model.field.grid, initial))

    def test_optimize_step_closes_warp_kinetic_and_equilibrium_loops(self) -> None:
        model = PermeatedVolumetricInterpreter(
            res=8,
            outer_ch=4,
            mid_ch=4,
            inner_ch=8,
            spectral_rank=4,
            max_update=0.5,
        )
        target_field = torch.randn(1, 16, 8, 8, 8) * 0.05
        model.field.set_grid(target_field)
        target_outer = target_field[:, :4].contiguous()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        before = [parameter.detach().clone() for parameter in model.warp_net.parameters()]
        loss = model.optimize_step(
            target_outer,
            optimizer,
            target_field=target_field,
            iterations=2,
        )
        self.assertTrue(math.isfinite(loss))
        self.assertTrue(math.isfinite(float(model.last_metrics["equilibrium_loss"])))
        self.assertTrue(math.isfinite(float(model.last_metrics["warp_reconstruction_loss"])))
        after = list(model.warp_net.parameters())
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, after)))
        self.assertTrue(torch.isfinite(model.field.grid).all())


if __name__ == "__main__":
    unittest.main()
