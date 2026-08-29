# Dr Moagi Runtime Fabric

The runtime fabric federates the repository's specialist engines beneath the Java 17 multimodal kernel without collapsing their trust boundaries or pretending that every runtime is the same kind of model.

## Operational topology

```text
                         +---------------------------+
user / application ----> | Dr Moagi Java 17 kernel  |
                         | recurrent 4x4x4 field     |
                         +-------------+-------------+
                                       |
                         explicit capability routing
                                       |
       +---------------+---------------+----------------+---------------+
       |               |               |                |               |
       v               v               v                v               v
 DMVANN Chat      Cognitive UI    Geometry core   QSOL 3D codec   Field simulator
 chat / proxy     control plane    3D optimizer    scene state      policy semantics
       |                                                               
       +-------------------------+-------------------------------------+
                                 |
                                 v
                        C++ self-editor3d
                    bounded code refinement
```

The machine-readable contract is `apps/dr-moagi-platform-java/runtime-fabric.json`. `tools/verify-dr-moagi-runtime-fabric.mjs` fails CI if a registered runtime disappears, declares an unsafe repository path, duplicates an identifier/capability, omits an authority boundary, or attempts to grant implicit host execution.

## Authority model

The fabric is an **orchestration contract**, not a superuser process.

- Specialist runtimes remain authoritative for their own measured invariants.
- Network-backed routing is opt-in through environment configuration; endpoint values and credentials are never stored in the manifest.
- The fabric itself grants no shell/process execution authority.
- Browser-only runtimes remain browser-only unless a separate, reviewed service adapter is implemented.
- Provisional state never becomes authoritative merely because another runtime requested it.
- A codec is not relabelled as a renderer, a simulation is not relabelled as production execution, and internal optimization objectives are not relabelled as external benchmark wins.

## Registered specialist domains

### `dmvann-chat`

Provides the repository's chat/control-field interface and optional server-side model proxy. Its Node runtime already exposes health/runtime/chat HTTP endpoints, so it is the first natural service bridge for the Java chat surface when an operator explicitly configures a trusted endpoint.

### `dr-moagi-cognitive`

Provides the production-oriented browser control plane: deterministic local state, visualization, session persistence, bounded command compilation, and verified firmware client operations. The browser cannot bypass firmware verification.

### `dr-moagi-geometry`

Owns bounded topology-preserving 3D geometry optimization. Candidate parameters remain provisional until measured geometry gates accept them.

### `qsol-graphics-codec`

Owns executable 3D scene-state encoding/decoding, quantization search, reconstruction-error gating, topology preservation, and animation-state persistence. It is intentionally renderer-agnostic.

### `moagi-field-sim`

Owns synthetic 3-bit physical/processing/microstructure state semantics and the generation/epoch commit barrier. It has no production market transmission authority.

### `cpp-self-editor3d`

Owns bounded native source refinement, recursive byte-to-voxel folding, closed-loop 3D symmetry autoencoding, and the associated deterministic benchmark harness. Source mutations remain workspace-confined, transactional, validator-gated, and rollback-capable.

## Platform surfaces versus specialist domains

The Java kernel exposes modality/task surfaces:

```text
chat | image | audio | video | code | data | compute
```

The runtime fabric exposes specialist domains:

```text
chat-proxy | control | geometry | graphics-state | field-simulation | code-refinement
```

These are deliberately orthogonal. For example, `qsol-graphics-codec` may provide geometry/animation state to an image or video renderer, but it is not itself claimed to synthesize photorealistic frames. Likewise, the C++ self-editor can validate bounded source refinements but does not automatically receive arbitrary code-generation authority from the Java `/code` surface.

## Federated CI invariant

`Dr Moagi Runtime Fabric` CI makes the federation executable as a repository contract. It:

1. validates the manifest structurally and against the checked-out tree;
2. recompiles/tests the Java platform kernel;
3. runs the DMVANN, cognitive, and geometry deterministic Node tests;
4. runs the QSOL graphics codec self-test;
5. builds and executes the C++ self-editor3d CTest suite.

This gives the platform a measurable integration invariant:

```text
platform-contract-valid
AND specialist-invariants-pass
=> fabric-compatible commit
```

It does **not** imply that all runtimes share memory, latency, deployment, or security context. Cross-runtime transport remains explicit and independently reviewable.

## Next operational layer

The next safe evolution is a versioned request/result envelope for service-capable runtimes:

```text
request_id
surface / capability
input content reference
latent signature + bounded telemetry
runtime target
provisional result
specialist validation evidence
promotion decision
```

That envelope can support distributed workers, artifact stores, GPU renderers, model gateways, and persistent memory while retaining the central invariant:

```text
PROVISIONAL != AUTHORITATIVE
```

until the owning specialist validates and promotes the result.
