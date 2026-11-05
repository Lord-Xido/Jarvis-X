# CMX3 procedural + residual 3D codec

CMX3 represents a volumetric scalar field as a compact procedural prior plus an optional quantized residual lattice.

```text
X -> G(theta) -> R = X - G(theta)
  -> residual lattice -> int8 quantization -> zlib
  -> S = [versioned header | residual payload], |S| <= 1024 bytes
  -> decode -> G(theta_hat) + D_R(z_R) -> X_hat
  -> MSE / RMSE / PSNR / maximum error
```

The encoder searches valid lattice sizes and compression levels and selects the minimum-MSE candidate that fits the hard seed budget. Payloads are never truncated. The header carries generator parameters, source resolution, residual metadata, payload length and CRC32.

The optional inward warp is spherical inversion,

```text
I_lambda(z) = lambda z / ||z||^2,
```

whose radius map is `r' = lambda/r`. The sphere `r=sqrt(lambda)` is invariant and the map is an involution, so CMX3 treats it as generator geometry rather than claiming it is itself a contraction.

Run the focused tests with:

```bash
python -m pip install -e ".[test]"
pytest tests/test_cmx3_codec.py -q --no-cov
```

Compression claims should use measured byte counts. Procedural regeneration from identical parameters is not evidence that arbitrary voxel fields are losslessly represented at the same seed size; source-specific information is carried by the residual path.
