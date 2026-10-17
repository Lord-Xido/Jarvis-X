import numpy as np
import pytest

from jarvisx.cmx3_codec import CMX3Codec, Mandelbulb3D, MandelbulbConfig, reconstruction_metrics


def test_procedural_round_trip():
    gen = Mandelbulb3D(MandelbulbConfig(max_iter=6))
    codec = CMX3Codec()
    seed, meta = codec.encode(gen, res=8)
    out = codec.decode(seed)
    assert seed[:4] == b"CMX3"
    assert len(seed) <= 1024
    assert meta["mode"] == "procedural-only"
    assert out["grid"].shape == (8,8,8)
    assert reconstruction_metrics(gen.generate(8), out["grid"])["mse"] < 1e-10


def test_residual_search_improves_nonprocedural_source():
    res = 8
    gen = Mandelbulb3D(MandelbulbConfig(max_iter=6))
    prior = gen.generate(res)
    axis = np.linspace(-1,1,res,dtype=np.float32)
    z,y,x = np.meshgrid(axis,axis,axis,indexing="ij")
    source = prior + (0.01*np.exp(-6*(x*x+y*y+z*z))).astype(np.float32)
    seed, meta = CMX3Codec().encode(gen, res, source, anchors=(0,4,6,8))
    recon = CMX3Codec().decode(seed)["grid"]
    assert len(seed) <= 1024
    assert meta["candidate_count"] >= 1
    assert reconstruction_metrics(source,recon)["mse"] <= reconstruction_metrics(source,prior)["mse"] + 1e-12


def test_crc_rejects_corrupt_residual():
    res = 8
    gen = Mandelbulb3D(MandelbulbConfig(max_iter=6))
    source = gen.generate(res); source[3:5,3:5,3:5] += np.float32(0.02)
    seed, meta = CMX3Codec().encode(gen,res,source,anchors=(8,),levels=(9,))
    if meta["payload_size"] == 0:
        pytest.skip("no residual selected")
    bad = bytearray(seed); bad[-1] ^= 1
    with pytest.raises(ValueError, match="CRC32"):
        CMX3Codec().decode(bytes(bad))
