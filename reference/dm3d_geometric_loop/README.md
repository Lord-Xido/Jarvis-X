# DM3D-GEOM1: bounded, verified 3D geometric encode/feedback reference

**Status: isolated C++17 numerical research reference.** No production runtime, network operations, financial/vehicle interfaces, model training, AGI inference, or autonomous code execution. This module does **not** implement or replace DM3D-ENV1, ADR-016 authoritative state, or the existing DM3D operational least-squares engine. Integration requires an explicit versioned adapter and independent review.

## Execute

```bash
g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror \
  reference/dm3d_geometric_loop/dm3d_geometric_loop.cpp -o /tmp/dm3d-geom
/tmp/dm3d-geom --self-test
/tmp/dm3d-geom --side 8 --brick 4 --passes 6 --out-prefix /tmp/dm3d-run > /tmp/dm3d-receipt.json
python3 reference/dm3d_geometric_loop/test_cli.py --exe /tmp/dm3d-geom
```

Supply arbitrary **raw 8-bit** source volume of exactly `side³` bytes with `--input input.u8`. Use `--lossy` to omit the exact XOR residual; `--out-prefix` creates `.dm3dtri` and `.recovered.u8`. The JSON receipt includes sampled `(x,y,z,input,approximation,recovered)` voxels, actual per-pass MSE, exact fixed-point and period-two checks, elapsed wall time, and wire byte accounting. You may open `viewer.html` locally, then select the produced JSON receipt to inspect measured 3D data; it does not access a server.

### Explicit mathematical geometry

The X-fastest address is `i = x + side*(y + side*z)`. Each `brick³` tile reads the eight actual corner voxels `c_abc`. The reconstruction at local coordinate `(x,y,z)` uses exact integer trilinear interpolation:

```
D = (brick - 1)^3
w_abc = (a ? x : brick-1-x) * (b ? y : brick-1-y) * (c ? z : brick-1-z)
X_hat(x,y,z) = floor((sum_abc c_abc*w_abc + D/2) / D)
```

This is a geometric computation, not a torus rendering. The formula reproduces all eight brick corners exactly. Coordinates and values are bounded; the integer numerator cannot overflow uint64_t for allowed brick sizes.

The default **lossless** mode emits every reconstructed voxel's XOR difference `r = X XOR X_hat`. Thus `X = X_hat XOR r` byte-for-byte. This guarantees recovery, **not compression**: a typical 8³ volume with brick=4 occupies 602 wire bytes for 512 original U8 bytes (26-byte header, 64 corner bytes and 512 residual bytes). The lossy mode uses only 90 bytes for this configuration, but does not preserve arbitrary input.

### Real inward loop and acceptance criterion

For the *lossy* transform `F(X) = DecodeLossy(EncodeCorners(X))`, each feedback pass computes `X[t+1]=F(X[t])` and reports the MSE relative to both the original and the preceding state. A fixed point is detected only by exact equality of all voxel bytes. A nontrivial two-cycle requires `X[t+1] == X[t-1]` **and** `X[t+1] != X[t]`. Because all corner samples survive the interpolation exactly, `F(F(X)) = F(X)` and a fixed point is reached from pass 2. This describes idempotence, **not learning**: approximation quality does not improve through repetition.

### Strict independent packet `DM3DTRI1`

All multi-byte integers are **big-endian**. Fixed 26-byte header:

| Byte offset | Length | Meaning |
|---:|---:|---|
| 0 | 8 | `DM3DTRI1` magic |
| 8 | 2 | version (1) |
| 10 | 2 | cube side 2..64 |
| 12 | 1 | brick 2..8, must divide side |
| 13 | 1 | flags: 0 = lossy, 1 = XOR residual |
| 14 | 4 | corner count, exactly `8*(side/brick)^3` |
| 18 | 4 | residual count, exactly 0 or `side³` |
| 22 | 4 | CRC32 of bytes `[0,22)` concatenated with payload `[26,end)` |
| 26 | variable | corner bytes, then optional XOR residual |

CRC32 is **accidental-corruption detection, not authentication or encryption**. The parser rejects inconsistent lengths, wrong magic/version, invalid geometry, extra bytes and checksum mismatch. No untrusted packet is executed as code. This wire format is **not** DM3D-ENV1 and must never be silently parsed as that protocol.

## CI and capability boundary

The dedicated workflow compiles with GCC and Clang at `-Werror`, runs built-in invariants plus independent Python subprocess tests over arbitrary U8 content, invalid inputs, both lossless/lossy modes, and a JSON verification receipt. The viewer only reads the actual receipt supplied by the operator. No acceleration, neural generality, compression win, hardware or operational deployment is claimed.