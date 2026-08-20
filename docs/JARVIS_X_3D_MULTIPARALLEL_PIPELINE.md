# JARVIS X 3D Multiparallel Pipeline

## Status

Bounded Python reference proposed by issue #113. The implementation is isolated from
the canonical bytecode VM and task-level system runtime.

This subsystem operationalizes four parts of the supplied architecture:

1. deterministic package splitting and multiprocess execution;
2. typed 3D asset transforms and read-only source-code geometry;
3. immutable branch snapshots and ordered reconciliation;
4. seeded, candidate-first topology search over a finite safe family.

It does **not** establish unrestricted self-modification, source-code rotation,
continuous online learning, distributed execution or measured acceleration.

## 1. Layer map

| Layer | Reference component | Implemented responsibility |
|---|---|---|
| Data | `Vector3`, `VertexBatch`, `Mesh` | Immutable finite 3D values and index-safe triangle meshes |
| Package | `WorkPackage`, `PackageReceipt` | Deterministic identities, source order, stage trace and failure receipt |
| Pipeline | `PipelineNode`, `PipelineTopology`, `ParallelPipeline` | Closed stage vocabulary, validated linear DAG, sequential/process execution and ordered merge |
| Encoding | `EncodedChunk`, `FramedArtifact` | UTF-8/binary/geometry encoding, bounded zlib decode, SHA-256 integrity and v1 framing |
| Spatial | `SpatialProcessor`, `CodeGeometry` | AST-validated code observation and typed asset coordinate transforms |
| Branching | `BranchSnapshot`, `JarvisX3DEngine` | Immutable snapshots, independent processing and typed merge |
| Evolution | `TopologyEvolution` | Seeded finite search with deterministic fitness and candidate-first promotion |
| Interface | `jarvisx-multiparallel` | `run`, `map-code` and `evolve` commands with strict JSON summaries |

The reference uses immutable Python tuples rather than introducing NumPy as a required
package dependency. A future NumPy adapter may accelerate `VertexBatch` operations if
it preserves the same dimensional, serialization and determinism contracts.

## 2. Operational state

One engine state is

\[
\mathcal E_t = (T_t, R_t, B_t, S_t),
\]

where:

- \(T_t\) is the active validated topology;
- \(R_t\) is the last successfully reconciled main run;
- \(B_t\) is the immutable branch map;
- \(S_t\) is aggregate telemetry and bounded-search state.

Research output is not authoritative VM state. There is no dependency from
`jarvisx.core` or `jarvisx.system_runtime` to this module.

## 3. Kinetic package flow

Given input \(X\), topology \(T\) and worker ceiling \(P\), the splitter produces
source-ordered packages:

\[
\operatorname{Split}(X;P,b)
=
(W_0,W_1,\ldots,W_{n-1}),
\qquad
n\leq \min(P,N_{\text{packages}}).
\]

Every package identity binds:

\[
\operatorname{id}(W_i)
=
H(\text{domain}\|H(T)\|H(X)\|i\|H(X_i)).
\]

Each worker evaluates the same ordered stage tuple:

\[
Y_i
=
f_m\circ f_{m-1}\circ\cdots\circ f_1(X_i).
\]

The v1 stage vocabulary is closed:

| Stage | Input | Output | Behavior |
|---|---|---|---|
| `LOAD` | raw value | same type | Identity/load boundary |
| `TRANSFORM` | text, bytes or geometry | same raw type | Text/bytes unchanged; typed geometry axis permutation and scale |
| `ENCODE` | raw value | `EncodedChunk` | Canonical type-tagged bytes plus raw SHA-256 |
| `COMPRESS` | `EncodedChunk` | `EncodedChunk` | Bounded zlib level `0..9` |
| `DECOMPRESS` | `EncodedChunk` | `EncodedChunk` | Strict size-checked decompression |
| `DECODE` | `EncodedChunk` | raw value | Exact typed reconstruction |
| `VERIFY` | any supported value | same type | Integrity/type verification without mutation |

Arbitrary callables are not accepted as stages. This keeps process messages
serializable and prevents a topology document from becoming arbitrary execution
authority.

### Deterministic reconciliation

Process completion may be out of order. Reconciliation is always:

\[
Y
=
\operatorname{Merge}
\left(
\operatorname{sort}_{i}
\{(i,Y_i)\}
\right).
\]

Merge rules are exact and typed:

| Output type | Merge rule |
|---|---|
| text | direct source-order concatenation, with no inserted newline |
| bytes | direct source-order concatenation |
| `VertexBatch` | ordered vertex concatenation |
| `EncodedChunk` | versioned `JXMP` framed artifact |
| `Mesh` | one complete package only |
| mixed types | reject |

If any package or the merge gate fails, the run returns failure receipts and no new
main run is committed:

\[
R_{t+1}
=
\begin{cases}
R^{*}, & \forall i:\operatorname{valid}(Y_i)\land\operatorname{valid}(Y),\\
R_t, & \text{otherwise}.
\end{cases}
\]

## 4. Framed encoding

The persisted multi-package envelope begins with:

```text
magic             4 bytes   "JXMP"
version           1 byte    0x01
data kind         1 byte
chunk count       4 bytes   unsigned big-endian
```

Every chunk then contains:

```text
compression       1 byte    none or zlib
raw size          8 bytes   unsigned big-endian
raw SHA-256      32 bytes
payload size      8 bytes   unsigned big-endian
payload           N bytes
```

The decoder checks frame bytes, chunk count, cumulative decoded size, compression
termination, trailing data, raw length and raw SHA-256 before returning a value. The
declared raw size is validated before decompression to bound compressed expansion.

Run telemetry records `zlib.ZLIB_RUNTIME_VERSION`. Exact compressed bytes and search
results are reproducible under the same declared codec runtime; cross-version zlib
byte identity is not assumed.

SHA-256 provides integrity and provenance; it does not make a lossy operation
reversible. The bundled encoders are exact for supported types.

## 5. 3D spatial observation

For validated Python source with line \(\ell\), the reference maps:

\[
g(\ell)
=
\left(
\ell,
\operatorname{indentColumns}(\ell),
\operatorname{nameTokens}(\ell)
\right).
\]

- `x` is the one-based line number;
- `y` is expanded indentation width, using tab width four;
- `z` is the number of non-keyword Python name tokens.

Each line and the complete source have SHA-256 identities. A spatial permutation

\[
(x,y,z)\mapsto s(a_1,a_2,a_3)
\]

creates a transformed `CodeGeometry` only. It does not mutate, reconstruct, reorder or
execute source. General line rotation is not semantics preserving because control
flow, indentation, definitions and data dependencies are ordered.

For `VertexBatch` and `Mesh`, the same axis permutation and uniform scale act on typed
coordinates and may therefore be used as an actual asset transform.

## 6. Branching and merging

Creating a branch captures:

\[
B_j=(j,n,H(X_j),H(T_j),X_j,\varnothing).
\]

The payload is restricted to immutable supported types. Processing produces a run
receipt attached to a replacement snapshot; it never mutates the original payload.

`merge_branches([j_0,...,j_k])` requires every named branch to have a successful run
and applies the same ordered type gate as package reconciliation. The call returns a
value; it does not silently replace main or VM state.

## 7. Bounded topology search

The search space is deliberately finite. Candidates may change:

- one of five type-safe stage templates;
- worker hint within `RuntimeLimits.max_workers`;
- positive batch size within `max_batch_size`;
- zlib level `0..9`.

The initial population includes the active topology. Selection retains the highest
scoring fraction, crossover chooses bounded fields from two survivors, and mutation
changes one bounded field. A seeded `random.Random` instance makes candidate generation
replayable.

### Fitness

Wall-clock time is reported but cannot select the active topology. The deterministic
fitness is:

\[
J(T)
=
0.50(1-r_T)
+0.30\frac{1}{1+\widehat W_T/U}
+0.20\frac{p_{\text{active}}}{n_{\text{packages}}}
-0.02\max(0,|T|-4),
\]

where:

- \(r_T\) is measured compressed payload size divided by raw size;
- \(\widehat W_T\) is a documented stage-weight cost model;
- \(U\) is bounded input size;
- \(p_{\text{active}}\) is effective package parallelism;
- \(|T|\) is node count.

This is a constrained parameter/topology search objective, not autonomous objective
discovery. It can still be gamed by an unrepresentative fixture, so promotion performs
one final complete verification on the declared test data. The active topology changes
only after that gate succeeds.

Observed `elapsed_ns` and packages per second remain telemetry because process startup,
scheduler load and host state are environmental signals.

## 8. Resource bounds

Default ceilings are:

| Resource | Default bound |
|---|---:|
| workers | 8 |
| packages per run | 256 |
| nodes per topology | 16 |
| input bytes | 16 MiB |
| merged/frame bytes | 64 MiB |
| vertices | 1,000,000 |
| branches | 128 |
| population | 64 |
| generations | 64 |

The process backend is local. Worker count is not a claim of CPU speedup; serialization,
startup and workload size may make it slower than sequential execution.

## 9. Python API

```python
from jarvisx.multiparallel import FramedArtifact, JarvisX3DEngine

source = "def f(x):\n    return x + 1\n"
engine = JarvisX3DEngine(code=source)

run = engine.process_parallel(num_workers=4, backend="process")
assert run.success
assert isinstance(run.output, FramedArtifact)
assert run.output.decode() == source

geometry = engine.spatial_process(axis_order="yzx", scale=2.0)
assert geometry.source_sha256 == engine.spatial_process().source_sha256
```

Branch exploration:

```python
left = engine.create_branch("left", "alpha")
right = engine.create_branch("right", "beta")
engine.process_branch(left)
engine.process_branch(right)

merged = engine.merge_branches((left, right))
assert merged.decode() == "alphabeta"
```

Seeded search:

```python
from jarvisx.multiparallel import EvolutionConfig

result = engine.auto_evolve(
    source,
    EvolutionConfig(generations=5, population_size=10, seed=7),
)
assert result.promoted
```

## 10. CLI

Process and optionally persist a framed artifact:

```bash
jarvisx-multiparallel run program.py \
  --workers 4 \
  --backend process \
  --output program.jxmp
```

Map code without rewriting it:

```bash
jarvisx-multiparallel map-code program.py --axis-order yzx --scale 2
```

Run deterministic bounded search:

```bash
jarvisx-multiparallel evolve program.py \
  --generations 5 \
  --population 10 \
  --seed 7
```

The commands emit strict JSON. Fields suffixed `_telemetry` may vary across runs and do
not participate in topology selection or provenance identities.

## 11. Validation matrix

`tests/test_multiparallel.py` and `tests/test_multiparallel_cli.py` cover:

- finite vector and mesh-index validation;
- topology cycle, fork and disconnection rejection;
- sequential/process artifact equivalence;
- completion-order-independent reconciliation;
- text, bytes, vertex and mesh round trips;
- frame truncation, tamper and compressed-expansion rejection;
- observational source mapping and syntax rejection;
- stage-type failure receipts and main-state preservation;
- branch isolation, ordered merge and mixed-type rejection;
- seeded evolution reproducibility;
- candidate-first promotion and failed-configuration rollback;
- worker, byte and branch ceilings;
- CLI artifact, JSON and no-source-rewrite contracts.

## 12. Promotion boundary

Promotion to a canonical reference laboratory requires:

1. Python 3.10–3.13 tests passing;
2. package compilation and type checks passing;
3. sequential/process equivalence on CI;
4. dependency audit and package build passing;
5. review of framing limits and process failure behavior;
6. explicit acceptance of ADR-010.

Even after promotion, the subsystem remains outside authoritative VM state unless a
future ADR defines a capability-gated adapter through `jarvisx.system_runtime`.
