# Jarvis-X Inward 3D C++ Self-Editing Engine

This subsystem gives the C++ runtime a bounded source-editing and self-refinement kernel. It maps active C/C++ source bytes into a cubic voxel field, recursively pools `2×2×2` neighborhoods into a `1³` latent core, ranks a conservative mutation set, applies one mutation transactionally, validates it, and either accepts the next source state or restores the rollback anchor.

It does **not** autonomously invent arbitrary native code, bypass repository controls, or persist failed mutations. The built-in autonomous mutation policy is intentionally semantics-conservative: trailing whitespace removal, excessive blank-line collapse, and canonical final-newline insertion. Exact replacement and insertion primitives are exposed for higher-level tooling, but ambiguous anchors and no-op edits are rejected.

## Inward recurrence

```text
source tree
  -> byte voxels V(x,y,z)
  -> recursive 2x2x2 pooling
  -> 1^3 core
  -> ranked admissible mutation
  -> transactional source edit
  -> optional compile/test validator
  -> objective gate
  -> accept next state OR rollback
  -> repeat to fixed point
```

The optimizer reports source-volume metrics and minimizes a bounded objective over trailing whitespace, excessive vertical whitespace, missing final newlines, and the measured structural hotspot field. A candidate is accepted only when validation succeeds and the objective does not increase.

## Build and test

```bash
cmake -S cpp_runtime/self_editor3d -B build/self-editor3d -DCMAKE_BUILD_TYPE=Release
cmake --build build/self-editor3d --config Release --parallel
ctest --test-dir build/self-editor3d -C Release --output-on-failure
```

## Inspect the 3D source state

```bash
./build/self-editor3d/jarvisx-self-editor3d analyze-3d cpp_runtime
./build/self-editor3d/jarvisx-self-editor3d field-3d cpp_runtime
./build/self-editor3d/jarvisx-self-editor3d propose-3d cpp_runtime
```

`field-3d` prints the inward side sequence for every source file, for example `24^3 -> 12^3 -> 6^3 -> 3^3 -> 2^3 -> 1^3`, together with the final core energies.

## Bounded self-optimization

Run without a validator only for source-canonicalization experiments:

```bash
./build/self-editor3d/jarvisx-self-editor3d optimize-3d cpp_runtime 16
```

For repository self-refinement, pass a compile/test command so every proposed mutation must survive validation before it can remain authoritative:

```bash
./build/self-editor3d/jarvisx-self-editor3d optimize-3d cpp_runtime 16 \
  "cmake --build ../build/cpp-runtime --config Release"
```

The validation command is explicitly supplied by the operator and is executed from the selected workspace root. It is not synthesized by the autonomous mutation policy.

## Transactional editing primitives

```bash
./build/self-editor3d/jarvisx-self-editor3d replace \
  cpp_runtime src/example.cpp "old_exact_text" "new_exact_text" \
  "cmake --build ../build/cpp-runtime --config Release"

./build/self-editor3d/jarvisx-self-editor3d insert-after \
  cpp_runtime src/example.cpp "unique_anchor" "inserted_text" \
  "cmake --build ../build/cpp-runtime --config Release"
```

Safety invariants:

- edits are confined to the selected workspace after canonical path resolution;
- absolute paths and path escapes are rejected;
- exact replacement/insertion anchors must occur exactly once;
- no-op transactions do not commit;
- writes use a temporary file followed by rename;
- failed compile/test validation restores the pre-edit bytes;
- objective-regressing autonomous mutations are rolled back;
- the refinement loop terminates when no admissible mutations remain or the pass budget is exhausted.

## Closed 3D symmetry autoencoding loop

The same subsystem now includes an executable mathematical closure of the three-plane pixel autoencoder. For an `n×n` frame `X`, the fixed encoder is

```text
E(X) = [ X, H(X), V(X) ]
```

with exact binary reconstruction under the aligned majority decoder. The operational inward permutation follows the supplied example convention

```text
[L0, L1, L2] -> [L2, L0, L1]
```

and satisfies `P^3 = I` in latent space. A learnable `3×3` row-stochastic transport `P_theta` is parameterized by row-wise softmax logits. The continuous decoder aligns the three transported planes, averages them, and exposes a hard threshold only after optimization.

The bounded objective is

```text
J(theta) = L_reconstruction
         + 0.25 L_latent-cycle
         + 0.25 L_fixed-point
         + 0.002 H(P_theta)
```

where the original binary frame remains immutable during parameter search. Deterministic coordinate descent accepts only objective-reducing logit moves. Once `P_theta` converges, the optimized operator is fed back recurrently:

```text
X_(t+1) = D_soft(P_theta E(X_t))
```

until the fixed-point residual falls below tolerance.

Run the reference fixture:

```bash
./build/self-editor3d/jarvisx-symmetry-loop3d
```

The regression suite proves the exact baseline invariant `D(E(X)) = X`, `P^3 = I`, the supplied fixture's decoded period-two orbit under exact cyclic transport, row-stochastic learnable transport, material objective reduction from the cyclic initialization, and convergence of the optimized feedback loop back to the invariant frame.

## Regression coverage

The CTest suite verifies workspace containment, ambiguous-anchor rejection, no-op rejection, recursive folding to a `1³` core, objective reduction, source canonicalization, fixed-point convergence, exact symmetry reconstruction, latent cyclic order, stochastic transport normalization, and closed-loop parameter/state convergence. GitHub Actions builds and tests the subsystem on GCC, Clang with sanitizers, and MSVC.
