# DM-IMTE-Q16: Inward Meta-Evolution Tensor Engine

## Status

This is a bounded, deterministic C++17 reference implementation of the submitted
\`64^3 -> 16^3 -> 4^3 -> 16^3 -> 64^3\` fixed-point architecture.

It uses Q16.16 arithmetic for the latent and core paths. It is a software reference
runtime, not evidence of physical hardware acceleration.

The original "1000x" label is intentionally not encoded as a performance claim.
Any acceleration claim must be backed by an explicit benchmark against a named
baseline on the same hardware.

## Geometry

World:

\[
X \in Q16.16^{64 \times 64 \times 64}
\]

Latent:

\[
Z \in Q16.16^{16 \times 16 \times 16}
\]

Core:

\[
C \in Q16.16^{4 \times 4 \times 4}
\]

Every latent coefficient represents a \`4^3 = 64\`-voxel world block and every
core coefficient represents a \`4^3 = 64\`-node latent block.

The block transform is scaled by \`1/sqrt(64)=1/8\`, so in exact arithmetic:

\[
W W^T = I
\]

and

\[
P = W^T W
\]

is the orthogonal projector onto the piecewise-constant \`4^3\` block subspace.

For the built-in shell/core/lattice target, the reference implementation reports
an initial RMSE near \`0.41704593\` and a representation floor near
\`0.19644432\`.

## Corrected inward update

The submitted prototype multiplied the latent state itself:

\[
z \leftarrow z(1 + \alpha m_c)
\]

which creates a zero-state deadlock because \`z=0\` remains zero until noise is
added. The operational version instead uses multiplicative permeation as a
bounded preconditioner on the residual correction:

\[
d_i = a_i - z_i
\]

\[
c_j = \frac{1}{64}\sum_{i \in B_j} d_i
\]

\[
m_j^{t+1} = \rho m_j^t + (1-\rho)c_j
\]

\[
g_j = \operatorname{clip}(1+\eta m_j, g_{\min}, g_{\max})
\]

\[
z_i^{t+1}
=
z_i^t
+
\lambda g_{c(i)} d_i
+
\gamma m_{c(i)}.
\]

The default bounded parameters are:

- local step \`lambda = 0.50\`
- gate sensitivity \`eta = 0.25\`
- core gain \`gamma = 0.25\`
- core EMA \`rho = 0.75\`
- gate range \`[0.50, 1.50]\`

This retains the inward/core/outward topology while guaranteeing progress from
the zero state.

## Hot-path reduction

The target latent is precomputed once:

\[
a = W x.
\]

Because

\[
W(x-W^T z) = Wx - WW^Tz = a-z,
\]

the iterative path does not need to decode a \`64^3\` field and re-encode it on
every latent update.

The recurrent hot path therefore operates on 4,096 latent nodes plus 64 core
nodes. The full \`64^3\` decode is performed only for reconstruction telemetry.

This is a structural reduction in world-sized passes. It is not reported as a
wall-clock speedup until benchmarked.

## Fixed-point safety corrections

The implementation also corrects four low-level issues in the submitted source:

1. Core residual accumulation is reset every step. The prototype retained
   \`core_dz\` between calls and therefore accumulated old gradients.
2. Core restriction uses 64-bit accumulators to avoid signed 32-bit overflow.
3. Q16 multiplication and addition saturate explicitly.
4. Annealing avoids left-shifting a negative signed integer, which is undefined
   behavior in C++.

Diffusion is applied only to the next warm start:

\[
z_{\mathrm{warm},t+1} = z_t^* + \xi_t.
\]

The reported solved state and reconstruction are never contaminated by the
post-solve noise.

## Build

From the repository root:

\`\`\`bash
cmake -S cpp_runtime -B build/cpp-runtime -DCMAKE_BUILD_TYPE=Release
cmake --build build/cpp-runtime --target jarvisx-inward-meta-q16 --parallel
./build/cpp-runtime/jarvisx-inward-meta-q16 --steps 15 --no-noise
\`\`\`

Run the regression target:

\`\`\`bash
cmake --build build/cpp-runtime --target jarvisx-inward-meta-q16-tests --parallel
ctest --test-dir build/cpp-runtime -R inward-meta-q16 --output-on-failure
\`\`\`

## Verification contract

The regression executable checks:

- exact \`64^3\`, \`16^3\`, and \`4^3\` dimensions;
- saturated Q16 arithmetic;
- exact \`WW^T\` behavior for representable coefficients;
- deterministic noise-free execution;
- monotone reconstruction improvement;
- convergence to within \`1e-6\` RMSE of the block projection floor after
  15 bounded steps.

The architecture remains a block-constant codec. Reaching the projection floor
does not imply exact reconstruction of arbitrary \`64^3\` geometry. Lower error
requires a richer basis or adaptive spatial refinement.
