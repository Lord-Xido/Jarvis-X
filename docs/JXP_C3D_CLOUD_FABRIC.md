# JXP-C3D: Sparse 3D Cloud Media Fabric

JXP-C3D is a reference implementation of the Jarvis-X geometric exchange model as a
bounded, testable cloud-runtime primitive.

It does **not** allocate a dense exabyte-scale memory object and it does **not** claim
that the reference codec is a trained production ANN. The implementation separates the
logical geometry from physical residency and preserves a clean replacement boundary for a
learned encoder/decoder.

## 1. Logical manifold

The protocol exposes

\[
\mathcal V = \{0,\ldots,999\}^3,
\qquad |\mathcal V| = 1000^3 = 10^9
\]

logical spatial cells.

If one chose to associate 1 GB with every logical cell, the namespace would describe
\(10^9\) GB = 1 EB of logical capacity. That is an address-space statement only.
The reference store materializes only the active set

\[
\mathcal A_t \subset \mathcal V.
\]

Physical memory therefore scales with \(O(|\mathcal A_t|)\), not \(O(10^9)\).

## 2. 30-bit spatial address

Because every coordinate is smaller than \(2^{10}\), each axis requires ten bits.
The runtime interleaves those bits into one Morton/Z-order key:

\[
M(x,y,z) =
\operatorname{interleave}_{10}(x,y,z) \in [0,2^{30}).
\]

This gives a compact key while retaining useful spatial locality.

A stream that does not provide an explicit coordinate is placed deterministically:

\[
i =
\operatorname{SHA256}(\text{stream-id}\Vert\text{sequence}) \bmod 10^9
\]

followed by base-1000 decomposition into \((x,y,z)\).

## 3. Transaction path

The reference runtime executes

\[
X
\xrightarrow{E}
Z_0
\xrightarrow{\Phi_{\rm in}^{\,n}}
Z_n
\xrightarrow{D}
\hat X
\xrightarrow{\text{residual}}
R
\xrightarrow{\text{JXP frame}}
\text{candidate}
\xrightarrow{\text{verify}}
\text{commit}.
\]

A candidate is never admitted to the sparse store before byte-exact reconstruction and
SHA-256 verification succeed.

### Reference encoder

For byte \(x_i\),

\[
z_i^{(0)} = \left\lfloor x_i / 16 \right\rfloor.
\]

The initial latent values therefore occupy a four-bit domain.

### Inward contraction

At each bounded refinement step, let

\[
c_t = \frac{1}{N}\sum_i z_i^{(t)}
\]

and

\[
z_i^{(t+1)}
=
\operatorname{clip}_{[0,15]}
\left(
\operatorname{round}
\left[
c_t+\rho(z_i^{(t)}-c_t)
\right]
\right),
\qquad 0 \le \rho < 1.
\]

This is a finite reference implementation of the inward geometric contraction. It is
not an assertion that all future learned Jarvis-X codecs should use this scalar rule.

### Exact residual

The provisional decoder is

\[
\hat x_i = 16 z_i.
\]

The exact signed residual is

\[
r_i = x_i-\hat x_i.
\]

The receiver reconstructs

\[
x_i = 16 z_i+r_i.
\]

The reference residual is intentionally stored as signed 16-bit values. This makes
correctness obvious but can make the wire representation larger than the source. No
compression ratio is claimed.

A production learned codec can replace \(E\), \(\Phi\), and \(D\) while retaining the
same candidate/verification/commit contract.

## 4. JXP-C3D frame

The wire representation is:

~~~text
+----------------------+ 8 bytes
| JXPC3D1 magic        |
+----------------------+ 4 bytes
| JSON header length   |
+----------------------+
| version              |
| stream / sequence    |
| x / y / z            |
| 30-bit Morton key    |
| modality / opcode    |
| inward iterations    |
| latent length        |
| residual length      |
| source SHA-256       |
| payload SHA-256      |
+----------------------+
| latent payload       |
+----------------------+
| exact residual       |
+----------------------+
~~~

The payload hash detects frame corruption. The source hash independently verifies the
fully reconstructed media.

## 5. Multimodal contract

The transport tags payloads as binary, text, image, audio, video, tensor, or bytecode.

The reference codec treats each as bytes. Modality-specific tokenizers, neural encoders,
video transforms, spectrogram encoders, or implicit neural representations can be added
above the same JXP-C3D transaction layer.

## 6. Cloud mapping

A distributed deployment can route a logical voxel to a physical shard using

\[
s =
H(
\text{tenant}
\Vert
\text{session}
\Vert
M(x,y,z)
)
\bmod N_{\rm shards}.
\]

The current Python implementation is an in-process sparse reference store. Replacing
that store with an object store, KV service, actor system, GPU worker pool, or QUIC
transport must not change spatial identity or verification semantics.

~~~text
logical 1000^3 manifold
          |
          v
30-bit Morton identity
          |
          v
cloud shard / worker scheduler
          |
          +--> ANN encoder / inward refinement
          |
          +--> JXP-C3D frame
          |
          +--> decode + residual
          |
          +--> SHA-256 verification
          |
          '--> sparse commit
~~~

## 7. Physical accounting

For \(A=|\mathcal A_t|\) active voxels and average resident frame size \(B\),

\[
C_{\rm resident} \approx A B.
\]

Runtime throughput must be reported from measured execution:

\[
R_{\rm actual}
=
\min(
R_{\rm compute},
R_{\rm memory},
R_{\rm network},
R_{\rm storage},
R_{\rm codec}
).
\]

Logical voxel count, requested iterations, and symbolic scale are not hardware-throughput
measurements.

## 8. Running the reference engine

~~~bash
python -m jarvisx.jxp_c3d
~~~

The demo transfers 1024 bytes through encode -> inward contraction -> exact residual ->
frame serialization -> frame validation -> decode -> SHA-256 verification -> sparse
commit and prints a JSON receipt.

Focused tests:

~~~bash
pytest tests/test_jxp_c3d.py
~~~

The repository's normal pull-request CI also compiles and tests the package across its
supported Python matrix.

## 9. Invariants

1. \(0 \le x,y,z < 1000\).
2. Every valid address has a 30-bit Morton key.
3. Dense \(1000^3\) allocation is never required.
4. Inward iteration count is finite and explicit.
5. The reference contraction never intentionally increases latent spread.
6. A committed voxel reconstructs byte-exactly.
7. Frame payload corruption is rejected before commit.
8. A failed reconstruction is never admitted into the active set.
9. Logical scale is not reported as measured physical throughput.
10. Learned codecs remain subordinate to the same verification boundary.

## 10. Master operational form

For active voxel \(v\),

\[
S_{v,t}
=
[X,Z,\hat X,R,\Omega,\Pi,H,C]_{v,t}
\]

and the cloud transaction is modeled as

\[
S_{v,t+1}
=
\mathcal C_{\rm verify}
\circ
\mathcal R_{\rm cloud}
\circ
D_\phi
\circ
\Phi_{\rm in}^{(n)}
\circ
E_\theta
(S_{v,t},X_t).
\]

In this reference implementation, \(E_\theta\), \(\Phi_{\rm in}\), and \(D_\phi\) are
deterministic bounded transforms. The interface is deliberately shaped so a measured,
trained ANN implementation can replace them without weakening the verification and sparse
allocation guarantees.
