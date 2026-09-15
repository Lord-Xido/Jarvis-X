#!/usr/bin/env python3
import math
import runpy
from pathlib import Path
import numpy as np

MODULE = Path(__file__).with_name("terabyte_3d_engine.py")
ns = runpy.run_path(str(MODULE), run_name="terabyte_test_module")

TBConfig = ns["TBConfig"]
SyntheticBlockSource = ns["SyntheticBlockSource"]
BlockAddress = ns["BlockAddress"]
BlockCodec3D = ns["BlockCodec3D"]
Terabyte3DEngine = ns["Terabyte3DEngine"]


def tiny_cfg(**kwargs):
    base = dict(
        logical_gb=0.00001,
        world_side=8,
        block_side=4,
        factor=2,
        channels=4,
        residual_quantum=0.02,
        root_quantum=0.01,
        correction_cycles=2,
        seed=11,
    )
    base.update(kwargs)
    return TBConfig(**base)


def test_geometry():
    cfg = tiny_cfg()
    assert cfg.blocks_per_axis == 2
    assert cfg.blocks_per_volume == 8
    a = BlockAddress.from_id(7, cfg)
    assert (a.volume, a.bx, a.by, a.bz) == (0, 1, 1, 1)
    b = BlockAddress.from_id(8, cfg)
    assert (b.volume, b.bx, b.by, b.bz) == (1, 0, 0, 0)


def test_exact_residual_roundtrip():
    cfg = tiny_cfg(residual_quantum=0.0, root_quantum=0.0)
    src = SyntheticBlockSource(cfg)
    codec = BlockCodec3D(cfg)
    block = src.get(0)
    enc = codec.encode(BlockAddress.from_id(0, cfg), block)
    dec = codec.decode(enc)
    assert np.max(np.abs(block - dec)) < 2e-6


def test_lossy_codec_is_bounded():
    cfg = tiny_cfg()
    src = SyntheticBlockSource(cfg)
    codec = BlockCodec3D(cfg)
    block = src.get(0)
    enc = codec.encode(BlockAddress.from_id(0, cfg), block)
    dec = codec.decode(enc)
    mse = float(np.mean((block - dec) ** 2))
    assert mse < 5e-4
    assert enc.encoded_bytes < enc.original_bytes


def test_relaxed_contraction_bound():
    cfg = tiny_cfg()
    codec = BlockCodec3D(cfg)
    q = codec.core.relaxed_bound
    assert q < 1.0
    assert math.isclose(q, 1-cfg.rho+cfg.rho*cfg.spectral_target, rel_tol=1e-7)


def test_global_reducer_and_runtime():
    cfg = tiny_cfg()
    src = SyntheticBlockSource(cfg)
    engine = Terabyte3DEngine(cfg, src)
    run = engine.run([0, 1])
    assert run.processed_blocks == 2
    assert run.processed_bytes == 2 * cfg.block_bytes
    assert math.isfinite(run.global_norm)
    assert run.mean_compression_ratio > 1.0


if __name__ == "__main__":
    tests = [
        test_geometry,
        test_exact_residual_roundtrip,
        test_lossy_codec_is_bounded,
        test_relaxed_contraction_bound,
        test_global_reducer_and_runtime,
    ]
    for t in tests:
        t()
        print("PASS", t.__name__)
    print("All hierarchical terabyte 3D engine tests passed.")
