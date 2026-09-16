"""CUDA/Triton accelerator for the monotone inward swarm operator.

This is an optional accelerator for ``jarvisx.inward_swarm_fixed_point``.  It
preserves the same two contracts as the dependency-free reference engine:

* coordinates use a clipped step and therefore cannot overshoot the core;
* packed words correct one differing bit toward an explicit target per pass.

The logical 1,000,000^3 domain is not densely allocated.  Only ``N`` sparse
agents and their packed state words are materialized.

Install CUDA PyTorch and Triton separately before running this example.
"""

from __future__ import annotations

import argparse
import math

import torch
import triton
import triton.language as tl


@triton.jit
def inward_fixed_point_kernel(
    positions_ptr,
    states_ptr,
    targets_ptr,
    center_x,
    center_y,
    center_z,
    num_agents,
    SPATIAL_STEP: tl.constexpr,
    WORDS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """One monotone spatial + latent contraction pass.

    ``states_ptr`` and ``targets_ptr`` use structure-of-arrays layout ``[WORDS, N]``
    so adjacent Triton lanes read adjacent int32 words for a fixed word index.
    """

    pid = tl.program_id(axis=0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < num_agents

    x = tl.load(positions_ptr + idx * 3 + 0, mask=mask)
    y = tl.load(positions_ptr + idx * 3 + 1, mask=mask)
    z = tl.load(positions_ptr + idx * 3 + 2, mask=mask)

    dx = center_x - x
    dy = center_y - y
    dz = center_z - z

    adx = tl.where(dx < 0, -dx, dx)
    ady = tl.where(dy < 0, -dy, dy)
    adz = tl.where(dz < 0, -dz, dz)

    sx = tl.where(adx <= SPATIAL_STEP, dx, tl.where(dx > 0, SPATIAL_STEP, -SPATIAL_STEP))
    sy = tl.where(ady <= SPATIAL_STEP, dy, tl.where(dy > 0, SPATIAL_STEP, -SPATIAL_STEP))
    sz = tl.where(adz <= SPATIAL_STEP, dz, tl.where(dz > 0, SPATIAL_STEP, -SPATIAL_STEP))

    tl.store(positions_ptr + idx * 3 + 0, x + sx, mask=mask)
    tl.store(positions_ptr + idx * 3 + 1, y + sy, mask=mask)
    tl.store(positions_ptr + idx * 3 + 2, z + sz, mask=mask)

    for word in range(WORDS):
        offset = word * num_agents + idx
        current = tl.load(states_ptr + offset, mask=mask)
        target = tl.load(targets_ptr + offset, mask=mask)
        diff = current ^ target
        correction = diff & (-diff)
        refined = current ^ correction
        tl.store(states_ptr + offset, refined, mask=mask)


class TritonInwardSwarm:
    """Finite sparse accelerator over a large logical coordinate domain."""

    def __init__(
        self,
        *,
        num_agents: int = 65_536,
        scale: int = 1_000_000,
        spatial_step: int = 1_000,
        words: int = 64,
        word_bits: int = 31,
        block_size: int = 256,
        seed: int = 41,
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("this Triton example requires a CUDA device")
        if num_agents <= 0 or scale < 2 or spatial_step <= 0:
            raise ValueError("invalid swarm geometry")
        if words <= 0 or not 1 <= word_bits <= 31:
            raise ValueError("invalid packed-state configuration")
        if block_size <= 0 or block_size & (block_size - 1):
            raise ValueError("block_size must be a positive power of two")

        self.device = torch.device("cuda")
        self.num_agents = int(num_agents)
        self.scale = int(scale)
        self.center = self.scale // 2
        self.spatial_step = int(spatial_step)
        self.words = int(words)
        self.word_bits = int(word_bits)
        self.block_size = int(block_size)

        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)
        self.positions = torch.randint(
            0,
            self.scale,
            (self.num_agents, 3),
            dtype=torch.int64,
            device=self.device,
            generator=generator,
        )
        # SoA [WORDS, N] layout: contiguous accesses across agents for each word.
        self.states = torch.randint(
            0,
            1 << self.word_bits,
            (self.words, self.num_agents),
            dtype=torch.int32,
            device=self.device,
            generator=generator,
        )
        self.targets = torch.zeros_like(self.states)
        self.grid = (triton.cdiv(self.num_agents, self.block_size),)

    @property
    def domain_worst_case_passes(self) -> int:
        farthest = max(self.center, (self.scale - 1) - self.center)
        spatial = math.ceil(farthest / self.spatial_step)
        return max(spatial, self.word_bits)

    def launch(self) -> None:
        inward_fixed_point_kernel[self.grid](
            self.positions,
            self.states,
            self.targets,
            self.center,
            self.center,
            self.center,
            self.num_agents,
            SPATIAL_STEP=self.spatial_step,
            WORDS=self.words,
            BLOCK_SIZE=self.block_size,
        )

    def run_verified(self, passes: int | None = None) -> dict[str, float | int | bool]:
        """Run a finite bound and verify the actual terminal state on device."""

        count = self.domain_worst_case_passes if passes is None else int(passes)
        if count <= 0:
            raise ValueError("passes must be positive")

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(count):
            self.launch()
        end.record()
        torch.cuda.synchronize()
        elapsed_ms = float(start.elapsed_time(end))

        spatial_ok = bool(torch.all(self.positions == self.center).item())
        latent_ok = bool(torch.equal(self.states, self.targets))
        return {
            "passes": count,
            "agents": self.num_agents,
            "logical_scale": self.scale,
            "elapsed_ms": elapsed_ms,
            "mean_kernel_pass_us": elapsed_ms * 1_000.0 / count,
            "spatial_fixed_point": spatial_ok,
            "latent_fixed_point": latent_ok,
            "verified_fixed_point": spatial_ok and latent_ok,
        }


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Verified Triton inward fixed-point swarm")
    p.add_argument("--agents", type=int, default=65_536)
    p.add_argument("--scale", type=int, default=1_000_000)
    p.add_argument("--step", type=int, default=1_000)
    p.add_argument("--words", type=int, default=64)
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--passes", type=int)
    return p


def main() -> None:
    args = parser().parse_args()
    engine = TritonInwardSwarm(
        num_agents=args.agents,
        scale=args.scale,
        spatial_step=args.step,
        words=args.words,
        block_size=args.block_size,
    )
    metrics = engine.run_verified(args.passes)
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
