# Jarvis-X

Jarvis-X is a deterministic, auditable virtual machine with a reflex
control layer and policy gate.

## Install

```bash
git clone https://github.com/Lord-Xido/Jarvis-X.git
cd Jarvis-X
pip install -r requirements.txt
pip install .
```

## 3D visual-memory permeation

Jarvis-X now includes a dependency-free reference implementation of a
transactional 3D visual-memory ANN. It encodes a scalar voxel volume into a
compact geometric latent lattice, recalls associative memory, recursively
projects reconstruction residuals back into the latent field, decodes the
result, and evaluates a bounded set of mechanics candidates.

```text
Volume3D
  -> GeometricCodec.encode
  -> LatentField
  -> SpatialMemory.read/write
  -> recursive residual refinement
  -> GeometricCodec.decode
  -> objective and mechanics gate
```

The optimiser does not rewrite arbitrary source code. Every candidate is a
validated `GeometricConfig`, every candidate runs from the same memory snapshot,
and a change is committed only when its measured reconstruction-plus-compute
objective improves over the baseline.

Run the deterministic demonstration:

```bash
jarvisx visual-memory 12
```

The command emits JSON containing the selected mechanics, reconstruction loss,
estimated operation count, candidate count, and auditable latent-refinement
telemetry.
