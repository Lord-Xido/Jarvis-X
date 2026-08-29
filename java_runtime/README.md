# Geometric 3D Java Runtime

`OperationalInvertedLLMEngine` is a dependency-free Java 17 recurrent-field substrate with explicit 3D geometry.

## Operational state

Each voxel `i` has activation `a_i`, recurrent weight `w_i`, six toroidally wrapped neighbours `N6(i)`, a precomputed world-space toroidal embedding, curvature, and geometric drive.

The synchronous update is:

```text
u_i      = w_i a_i + beta mean(a_j, j in N6(i)) + lambda G_i
a'_i     = tanh(u_i)
r_i      = a'_i - a_i
L_fixed  = 1/2 r_i^2
g_i      = r_i (1-a'^2_i) a_i + decay w_i
w'_i     = w_i - eta g_i
```

All reads come from the previous field and all writes go to scratch buffers. The barrier copy after each ForkJoin step makes neighbour coupling deterministic and removes in-place update races.

## Geometry

For lattice angles `theta=2πx/d`, `phi=2πy/d`, `psi=2πz/d`, the runtime embeds each voxel into a nested toroidal shell using major radius `R1=1000` and minor radius `R2=250`. The z-axis phase modulates shell radius and depth. A normalized Gaussian-curvature term modulates inward geometric coupling.

The index topology wraps independently on x, y and z, producing a periodic 3-torus neighbourhood graph.

## Convergence

A cycle reports fixed-point loss, fixed-point RMS, neighbour-coherence loss, mean parameter update, mean gradient magnitude, stability and duration. Convergence is declared only when fixed-point RMS and stability satisfy their thresholds for three consecutive cycles; otherwise the runtime reports `MAX_ITERATIONS_REACHED_WITHOUT_FIXED_POINT`.

## Run

```bash
mkdir -p java_runtime/out
javac --release 17 -d java_runtime/out $(find java_runtime/src/main/java java_runtime/src/test/java -name '*.java')
java -cp java_runtime/out com.engine.virtual.llm.operational.OperationalInvertedLLMEngineTest
java -cp java_runtime/out com.engine.virtual.llm.operational.OperationalInvertedLLMEngine 128 25
```

This is the geometric recurrent compute substrate. Tokenization, embeddings, vocabulary projection and sampling remain separate language-model surfaces to be layered above it.
