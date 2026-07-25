# Jarvis-X

Jarvis-X is a deterministic, auditable virtual machine with a reflex control
layer, policy gate, and sparse predictive runtime.

## Runtime v1

The Runtime v1 branch introduces a bounded 30-dimensional sparse 3D swarm over
a virtual `1000^3` address space. Only active voxels are materialised.

Each transaction follows:

```text
observe -> predict -> evolve -> residual -> update Ω
        -> project Λ -> allocate/prune -> journal -> commit
```

Implemented controls include:

- explicit 3D diffusion stability validation;
- local-global-memory prediction;
- persistent residual memory;
- prediction, motion, and constraint residuals;
- bounded deterministic sparse expansion;
- optional state projection;
- reproducible execution manifests and chained state hashes;
- evidence-based architecture scoring with weakest-link gating.

See:

- [`docs/JARVIS_X_RUNTIME_V1_SPEC.md`](docs/JARVIS_X_RUNTIME_V1_SPEC.md)
- [`docs/30D_SPARSE_SWARM.md`](docs/30D_SPARSE_SWARM.md)

## EarthTwin browser demo

[`examples/earthtwin/`](examples/earthtwin/) is a static Three.js demonstration of
the runtime's encode/decode, introspection, caching, adaptive LOD, and
usage-driven refinement concepts.

Run it from the repository root:

```bash
python -m http.server 8000
```

Then open `http://localhost:8000/examples/earthtwin/`.

The chat interface is a deterministic city-ROM query parser, not a general
language model. See [`examples/earthtwin/README.md`](examples/earthtwin/README.md)
for its operational and security boundaries.

## JARVIS X neural echo demo

[`examples/jarvisx-echo/`](examples/jarvisx-echo/) replaces simulated cognitive
telemetry with trace-driven execution and a small trainable neural core.

It includes:

- a deterministic `12 -> 16 -> 4` classifier trained by online backpropagation;
- hidden-activation and loss visualization driven by actual inference;
- versioned, hashed, evidence-gated instruction-order mutation;
- checksum-verified binary ROM recompilation;
- bounded LRU/TTL query memory;
- declarative alias learning through `teach <alias> = <city>`;
- explicit runtime state transitions and execution traces;
- optional browser speech synthesis and a 3D neural mirror graph;
- browser-core tests for neural learning, cache bounds, ROM integrity, query
  behavior, state transitions, and mutation invariants.

Open `http://localhost:8000/examples/jarvisx-echo/` after starting the same local
server. The implementation is sentient-feeling by design but does not claim
consciousness or sentience. See
[`examples/jarvisx-echo/README.md`](examples/jarvisx-echo/README.md).

## Install

```bash
git clone https://github.com/Lord-Xido/Jarvis-X.git
cd Jarvis-X
pip install -r requirements.txt
pip install .
```

## Test

```bash
pip install -e ".[test]"
pytest
node --test examples/jarvisx-echo/runtime-core.test.mjs
```
