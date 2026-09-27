# 1M³ Cognitive Swarm Auto-Encoder — React/WebGL Visualization

## Status

Canonical browser visualization:

`examples/1M3_Cognitive_Swarm_AutoEncoder.html`

This artifact is a pre-bundled React 18 / Three.js WebGL visualization of the large-scale Jarvis-X
cognitive-swarm architecture. It is a visualization and interaction layer, not a claim that the
logical 1M³ / 1e18-voxel state is physically allocated.

## Visual architecture

The artifact presents a logarithmically scaled virtual system with:

- a logical `1,000,000 × 1,000,000` input bus;
- a displayed `1,000,000³ = 1e18` virtual voxel domain;
- instanced input/output transceiver fields;
- cyan encoder and purple decoder swarms;
- a nested icosahedral latent/core visualization;
- point-cloud cognitive activation rather than dense rendered cubes;
- sparse threshold control `τ`;
- swarm-count and velocity controls;
- GPU Bézier transport in shader space;
- FPS and point-count telemetry;
- input-compress, core-think and output-reconstruct stage states.

The visual scale markers deliberately distinguish the logical scale from the physically rendered
working set. The renderer uses a bounded point/instance population to emulate the larger address
space.

## Performance strategy

The artifact is designed around GPU-friendly primitives:

- `InstancedMesh` for repeated transceiver geometry;
- `BufferGeometry` / `Float32Array` point state;
- vertex-shader Bézier motion rather than JavaScript all-pairs boid dynamics;
- point-size attenuation as level-of-detail;
- sparse activation thresholding to modulate the cognitive point field.

This is the correct architectural direction for a browser renderer because cost scales with the
resident working set, not with the full logical 1e18 address domain.

## Telemetry boundary

The current artifact contains presentation/simulation telemetry rather than measured Jarvis-X
runtime telemetry.

In the uploaded bundle:

- MSE is synthesized from a bounded random expression;
- latency is synthesized from a random term plus velocity-dependent presentation logic;
- coherence includes a sinusoidal presentation term;
- active-node counts are derived from sparsity and animation time;
- the `LIVE` label is UI state, not evidence of a remote service connection;
- no `fetch`, XMLHttpRequest, WebSocket or EventSource transport is invoked by the uploaded
  artifact.

Accordingly, these values should be interpreted as interactive visualization telemetry until the
page is explicitly bound to runtime receipts.

## Runtime correspondence

The visualization maps conceptually onto the executable Jarvis-X 3D stack:

| React/WebGL visualization | Executable analogue |
| --- | --- |
| logical 1M×1M input bus | bounded input/ingest plane |
| cyan encoder swarm | ENCODE3D / spatial routing |
| cognitive point nebula | active sparse 3D working set |
| nested latent geometry | PHI3D_ITER / fixed-point latent refinement |
| purple decoder swarm | DECODE3D / reconstruction routing |
| displayed MSE | runtime residual/MSE receipt when instrumented |
| sparsity τ | active-working-set threshold |
| stage state | spatial VM execution region |
| FPS / point telemetry | renderer performance telemetry |

The authoritative numerical runtime remains:

`src/jarvisx/dr_moagi_inward_3d_vm.py`

with explicit fixed-point telemetry, residual localization, contractivity checks, shadow candidate
evaluation, promotion and rollback.

## Scaling contract

The central invariant is:

[
N_{resident} \ll N_{virtual}
]

and browser cost should scale primarily as:

[
O(N_{resident} + N_{swarm})
]

rather than:

[
O(10^{18}).
]

The `1M³` label therefore describes the logical/virtual state space. It must not be interpreted as
dense browser memory allocation.

## Running

Open `examples/1M3_Cognitive_Swarm_AutoEncoder.html` in a modern WebGL-capable browser.

The bundle includes its application code directly in the HTML. The embedded font declarations
reference `/fonts/OptimisticAI_VF_Optimized.woff2` and
`/fonts/OptimisticMono_W_TextRegular.woff2`; if those files are absent, browser font fallback
applies.

## Claim boundary

The visualization is suitable for architectural demonstration, interaction design, sparse working-set
experiments, and renderer profiling. External hardware throughput, physical memory capacity,
energy efficiency, nanosecond latency, and state-of-the-art performance require separately
instrumented benchmarks.
