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
