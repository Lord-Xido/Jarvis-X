# Jarvis-X multipurpose 3D ANN — audited reference app

An offline, dependency-free 3D voxel demo combining an exact GF(2) XOR7 jump kernel and a trained, task-conditioned 10–18–1 neural network with a convergent inward latent feedback operator.

## Run

Open `index.html` (in this directory) in a browser; it loads local `dm3d_core.js` and `ui.js`. Rotate the voxels, select restoration / edges / XOR distillation, adjust feedback parameters, and train the ANN. Run `node tests/verification.js` to execute deterministic numerical tests. No server, GPU, network, or external dependencies.

## What is exact and what is learned

For a 3D periodic power-of-two lattice, `K = I XOR Tx+ XOR Tx- XOR Ty+ XOR Ty- XOR Tz+ XOR Tz-`. Over GF(2), the Frobenius identity gives `K^(2^j) = I XOR Tx+^(2^j) XOR ...`; specifically `K^(L/2)=I`. The implementation composes `popcount(n modulo (L/2))` passes. The shortcut is *not* valid for nonlinear ANN layers.

**Critical methodology:** the XOR *ANN* receives the exact five-step XOR kernel result as feature #3. Therefore XOR model accuracy measures **oracle-assisted distillation**, not independent prediction or generalization. The UI additionally shows accuracy after *masking that feature* (a stress test, not a model trained without it). Restoration and boundary tasks do not receive their exact target in that feature.

Training data, update-gating validation data, and a separate reporting-audit set are generated in sequence with a deterministic pseudorandom generator. Only the validation set is consulted for accept/rollback. The audit set is not used for weight updates; repeated human tuning using its displayed metrics could still create selection bias. `balancedAccuracy`, `iou`, and class prevalence are returned alongside raw voxel accuracy for class-imbalance analysis.

## Fixed-point contract

With `E=Q=mean6`, `D(z)=(1-gamma)h_theta+gamma*z`, input anchor `b=0.45x+0.55E(x)` and `0 <= lambda <1`, `0 <= gamma <=1`:

`F(z)=(1-lambda)b+lambda/2 [Q(z)+E(D(z))]`.

`||F(u)-F(v)||_infinity <= q ||u-v||_infinity`, where `q=lambda(1+gamma)/2<1` for allowed controls. The a posteriori fixed-point error bound is `||z_(k+1)-z_k||_infinity/(1-q)`. **This does not certify correct task predictions nor guarantee convergence of outer SGD training.**

## Boundaries and honest limitations

- HTML is a 16³ JavaScript demonstration, not a 1e6-engine simulation or HPC throughput benchmark.
- The 3D latent field occupies a full-resolution volume, so this prototype does **not** physically compress model memory.
- Synthetic spheres, tori and wave fields do **not** validate real clinical imagery, audio, natural-language, or complex media processing.
- Read-only UI refreshes no longer modify the persistent `omega` diagnostic state; loading a new volume updates it once.
- Time and FLOP metrics need a separately instrumented hardware benchmark; this package does not claim a measured astronomical speedup.

## CI

The isolated `multipurpose-ann-verify.yml` workflow tests the arithmetic and numerical assertions on Node 20 and Node 22 for pull requests and changes to this app. The default branch remains protected by review practices outside this patch.
