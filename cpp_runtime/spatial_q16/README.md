# Jarvis-X deterministic Q16.16 sparse 3D field and reversible page codec

**Status: integration candidate.** This is a corrected and extended reference implementation of the proposed 3D fixed-point spatial field. It is not a trained ANN, compression codec, cryptographic cipher, quantum computer, or CPU bytecode interpreter. The 0x90 values are stored data only, never executed as machine instructions.

## Capabilities

- Sparse on-demand pages with (x,y,z) each in [0,999], forming one billion logical coordinate locations. Each active page stores 4096 instruction-data bytes and 1024 Q16.16 field variables. The default 2x2x2 example creates eight pages, not a billion.
- Checked fixed-point addition, subtraction and multiplication with explicit floor rounding for negative products; rational initialization avoids platform-dependent float conversion.
- Deterministic per-page state recurrence driven by five named numerical registers.
- An operational convergence criterion: the maximum raw update must be <= 1 Q16.16 LSB. Exhausting the iteration budget is reported separately.
- Exact reversible XOR/delta byte transform, independent decoding by regenerating the same field state, and CRC32 checks of input and output. CRC32 is not a cryptographic integrity mechanism.

## Build and test

Requires C++20 and CMake 3.16+.

    cmake -S cpp_runtime/spatial_q16 -B /tmp/jx-q16 -DSPATIAL_Q16_SANITIZERS=ON
    cmake --build /tmp/jx-q16 -j2
    ctest --test-dir /tmp/jx-q16 --output-on-failure
    /tmp/jx-q16/spatial_q16 demo 30
    /tmp/jx-q16/spatial_q16 demo 256

Standalone: c++ -std=c++20 -O2 cpp_runtime/spatial_q16/spatial_q16.cpp -o spatial_q16 && ./spatial_q16 selftest

Encode and decode a file with exactly 4096 input bytes:

    ./spatial_q16 encode page.bin page.jxq16 500 500 500 30
    ./spatial_q16 decode page.jxq16 restored.bin
    cmp page.bin restored.bin

## Fixed-point math

For each 4-byte group, start w_0=0.5. The coordinate-specific per-group bias b is a deterministic fraction in [0,255/256]. Let phi=985/1000, lambda=15/1000, theta=420/1000 and alpha=1/8, quantized to Q16.16.

    g_t = phi * w_t - b + lambda * (w_t - theta)
    w_(t+1) = w_t - alpha * g_t

Each multiplication floors toward negative infinity in Q16.16. Ignoring intermediate quantization, the corresponding real-number affine map has derivative 1-alpha*(phi+lambda), approximately 0.875. In finite precision, the algorithm uses an operational stopping test, not a proof of exact fixed-point convergence.

For each four-byte group p0,p1,p2,p3, derive a deterministic key byte from its current field value, coordinate and group index.

    encode: e0=p0 XOR key; e1=p1 XOR p0; e2=p2 XOR p1; e3=p3 XOR p2
    decode: p0=e0 XOR key; p1=e1 XOR p0; p2=e2 XOR p1; p3=e3 XOR p2

These identities prove reversibility when the same coordinates and rounds are used. The data are not encrypted.

## Binary frame (little-endian)

| Offset | Length | Description |
| ---: | ---: | --- |
| 0 | 8 | Magic: ASCII JXQ16V1 then NUL |
| 8 | 4 | x |
| 12 | 4 | y |
| 16 | 4 | z |
| 20 | 4 | rounds (0..512) |
| 24 | 4 | CRC32 of original page |
| 28 | 4 | CRC32 of encoded payload |
| 32 | 4096 | transformed page payload |

A frame occupies 4128 bytes for 4096 bytes of input: 32 bytes of expansion. This program performs *reversible mapping*, not entropy compression.

## Integration boundary

- The field variables are not trained weights; no learning or backpropagation occurs.
- No AVX intrinsic is used. There are no measured dense 1-billion-voxel inference/throughput results.
- The phi, lambda, omega, theta, and psi names designate arithmetic state fields. No quantum, moral, holographic, or cognitive physical properties are established.
- The standalone target does not change existing Jarvis-X codecs. Integration to an ANN is future work requiring an API contract and regression baselines.

## Source

The C++ implementation is in this directory. Selftest covers overflow, coordinates, replay, CRC failure, and 36 round trips.
