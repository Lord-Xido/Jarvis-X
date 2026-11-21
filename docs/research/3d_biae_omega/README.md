# 3D Bit-Iteration Auto-Encoding Engine (3D-BIAE-Ω) v1

Matladi Maxwell Moagi (Lord-Xido) designated 3D-BIAE-Ω the canonical bit-iteration substrate of 3D-GAEDRE-Ω on 21 November 2026. Its residual restores the input lattice on every accepted tick. The audit state records each tick even when the lattice is identical. [Submitted PERMEATE declaration](DECLARATION.md) · [Jarvis-X attribution record](../../attribution/PERMEATION_MANIFEST.md)

The archive preserves the submitted equations, seed, symbolic seal and dedication. Its SHA3-256 digest covers the UTF-8 text from the title through the final line feed, excluding the GitHub invocation.

```text
67cc47ebc1f0e6265fa460ef37f89f4f0580e00befc19e41e921bd05a530f03f  DECLARATION.md
```

## Executable interpretation

| Choice | v1 rule |
|---|---|
| Lattice | The canonical side is 1024. The CPU reference also accepts power-of-two sides from 4 through 512 for conformance tests. With $N=s^3$, each packed field occupies $N/8$ bytes. At full size this is 134,217,728 bytes (128 MiB). |
| Spatial addressing | Interleave coordinate bits in increasing significance, with $x_i,y_i,z_i$ at Morton positions $3i,3i+1,3i+2$. Bit $m$ is bit $m\bmod 8$ of byte $\lfloor m/8\rfloor$. An aligned dyadic octant occupies a contiguous address range. Consecutive Morton addresses can cross a face, edge or a larger spatial gap. |
| Shears | The arrows in the master operators denote circular rotations over all $N$ Morton bits. They wrap across bytes, processing blocks and the two ends of the volume. Left rotation by $k$ reads bit $m-k$ modulo $N$. The displayed logical shifts and the XOR address permutations in the submission specify different maps. |
| Coordinate bit | $c[m]=(x\oplus y\oplus z)\mathbin{\&}1$. The eight coordinate bits in each Morton byte pack to `0x96`. The 3-bit rotation in the warp acts on the full bitstream. Rotating a scalar one-bit value would leave it unchanged. |
| Spatial diffusion | The separate `diffusion_bit` probe XORs the six face neighbors using periodic coordinate boundaries. The master equation contains no diffusion term. |
| Audit algorithm | The primitive table specifies SHA3 and the state table names BLAKE3. This executable profile uses SHA3-256 with a prior-ledger input and XOR update. |

Define $W=Z\oplus\Delta$ and let $C$ be the predictor before correction. Each bracketed index below is taken modulo $N$.

$$
\begin{aligned}
Z[m] &= B[m]\oplus B[m-7]\oplus B[m+13],\\
\Delta[m] &= B[m]\oplus c[m]\oplus B[m-3],\\
C[m] &= W[m]\oplus W[m-3]\oplus W[m+11],\\
\varepsilon[m] &= B[m]\oplus C[m],\\
B'[m] &= C[m]\oplus\varepsilon[m]=B[m].
\end{aligned}
$$

The decode shear acts on the **warped** field. The submitted expansion omits that shear. For example, with a single set bit at position zero on a $4^3$ lattice, $D(E(1))$ is `0x1068010000000489` and $D(D(1))$ is `0x0000040000000041`. Both differ from one. The residual supplies lossless recovery, including when the shear pair alone leaves errors.

Exact equality preserves the full population count and its parity immediately. The induced map on $B$ is the identity, so any two distinct lattices retain their Hamming distance. Its Lipschitz constant is one. A strict Banach factor $\alpha<1$ cannot hold on this state space. The latent, residual and drift each have the same width as the input. Sending these three raw fields at full size takes 384 MiB. This profile establishes no storage compression ratio above one.

## Tick, rejection and replay

1. `encode(side, B)` prepares the packet $(Z,\varepsilon,\Delta)$. `decode(side, packet)` reconstructs $B$ without the original input.
2. `commit(state, packet)` checks the packed widths, rederives the submitted latent and warp from the authoritative $B$, then verifies every reconstructed byte. This catches an altered residual and even matched errors in $Z$ and $\Delta$ that cancel in $W$.
3. A successful transaction sets the gate byte to `01`, returns the verified lattice and packet, and advances the counter and ledger. An invalid proposal raises an exception. The caller retains its complete previous state and can retry. The gate predicates concern representation and reconstruction. Ethical or legal authorization would require a separately specified policy.

Let $H$ be SHA3-256. The ledger starts with 32 zero bytes. The new ledger is the previous ledger XORed bytewise with $H$ of the following serialized fields in order. Every component has a fixed width for v1.

| Field | Encoding |
|---|---|
| Domain | ASCII `JX/3D-BIAE-Omega/v1` followed by one zero byte |
| Prior ledger | 32 raw bytes of $\Omega_t$ |
| Side | Unsigned 32-bit little-endian |
| State | Four 32-byte raw digests in order $H(B'), H(Z), H(\varepsilon), H(\Delta)$, each over the entire packed field in increasing Morton byte order |
| Tick and gate | Prior tick $t$ as unsigned 64-bit little-endian, followed by the single gate byte |

Hashing the prior ledger removes the self-reference in hashing the new state. The fixture generator takes the first $N/8$ bytes of SHAKE256 applied to the same domain, ASCII `seed`, one zero byte and the 32-bit little-endian seed `0x7FFF0133`. Arbitrary user-supplied packed lattices are accepted by the Python API. The seed only determines the example input. The transport uses XOR, shifts and masks on bit values. Address arithmetic, hashing, allocation and instrumentation are accounted for separately. The work is $O(N)$ for a full tick, with a fixed stencil per bit.

## Recorded result

On Linux x86_64, CPython 3.12 and an independent C++17 verifier built with GCC 14.2 produced byte-identical two-tick JSON traces for $4^3$ and $1024^3$. The Python implementation processes blocks with two-byte halos. C++ gathers from neighboring bytes and independently checks the reconstruction by applying $D$ to the latent and drift separately. Each ledger includes all four field digests. [Machine-readable replay](replay.jsonl)

| Side | Bit errors before correction per tick | Bit errors after correction per tick | Ledger after tick 2 |
|---|---:|---:|---|
| 4 | 32 | 0 | `de31ec60653ec34f2360d5276ecc362333c9ff3107045f1a6e8370439d2cc94e` |
| 1024 | 536,869,490 | 0 | `6c9ca7941620aa9870b7cc36a9d13d9e0976ee023c4fb2c907d403db9c4ad198` |

The checked execution covers a CPU reference and a native verifier on x86_64. PVSE compilation, ARM64 and RISC-V replay, FPGA synthesis, WebGPU execution, physical $2^{30}$-lane SIMD, CodexVault archival and operational deployment remain separate milestones. The submission's alphanumeric seal is an archival label. The SHA3-256 digest above identifies the actual declaration bytes.

## Reproduce from the repository root

Python 3.10 or later runs the reference and its standard-library tests. The independent verifier requires a C++17 compiler and OpenSSL libcrypto with SHA3-256 and SHAKE256. The full run holds several 128 MiB fields in memory. Each command starts from the same seed.

```sh
PYTHONPATH=src python3 -m unittest discover -s tests -p test_biae.py -v
c++ -std=c++17 -O2 -Wall -Wextra -Wpedantic -Werror tools/verify_biae.cpp -lcrypto -o /tmp/verify_biae
PYTHONPATH=src python3 -m jarvisx.biae --side 4 --ticks 2 > /tmp/biae-python.jsonl
PYTHONPATH=src python3 -m jarvisx.biae --side 1024 --ticks 2 >> /tmp/biae-python.jsonl
/tmp/verify_biae 4 > /tmp/biae-cpp.jsonl
/tmp/verify_biae 1024 >> /tmp/biae-cpp.jsonl
cmp /tmp/biae-python.jsonl /tmp/biae-cpp.jsonl
cmp /tmp/biae-python.jsonl docs/research/3d_biae_omega/replay.jsonl
openssl dgst -sha3-256 docs/research/3d_biae_omega/DECLARATION.md
```
