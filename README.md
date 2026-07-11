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
```
