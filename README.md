# Jarvis-X

Jarvis-X is a deterministic, auditable virtual machine with a reflex control
layer, policy gate, transactional ledger, and an experimental 3D geometric
visual-memory subsystem.

## Install

```bash
git clone https://github.com/Lord-Xido/Jarvis-X.git
cd Jarvis-X
pip install -r requirements.txt
pip install .
```

## Unified execution model

The VM is the authoritative control plane. Bytecode and geometric operations
both pass through policy, journaling, and trace boundaries.

```text
Assembly -> Parser -> Assembler -> CodexVM -> Registers / Ledger / Trace
                                      |
                                      +-> V3D.PERMEATE
                                           |
Volume3D -> GeometricCodec -> LatentField -> SpatialMemory
                                           |
                                  spatial residual refinement
                                           |
                                     reconstruction
                                           |
                                bounded candidate commit
```

## 3D visual-memory permeation

The current 3D module is a deterministic reference state-transition kernel. It
is ANN-compatible, but it is not yet a trained neural network: encoding,
decoding, associative recall, and residual projection are explicit mathematical
operators rather than learned weights.

The runtime:

- encodes scalar voxel data into a compact geometric latent lattice;
- recalls finite content-addressed memory;
- decodes a reconstruction;
- projects each spatial residual region back into its corresponding latent cell;
- evaluates a bounded set of validated mechanics candidates from identical
  memory snapshots;
- commits only a strictly improved reconstruction-plus-compute objective;
- journals every candidate measurement and the selected VM-authoritative event.

Run the deterministic demonstration:

```bash
jarvisx visual-memory 12
```

The command emits JSON containing selected mechanics, reconstruction loss,
estimated operation count, all candidate objectives, and numerical refinement
telemetry.

## Local API and dashboard

Jarvis-X uses its declared FastAPI/Uvicorn stack:

```bash
jarvisx api   # http://127.0.0.1:8080
jarvisx web   # http://127.0.0.1:5000
```

Available API routes include:

```text
GET  /health
POST /run
POST /visual-memory
```

These development servers bind to loopback by default. Production exposure
requires an external authentication, authorisation, TLS, rate-limiting, and
process-isolation boundary.

## Invariants

1. Equal initial state and input must produce equal geometric results.
2. Candidate mechanics must start from identical memory snapshots.
3. A candidate cannot commit unless its declared objective improves.
4. Geometric execution can be blocked by the Lambda policy gate.
5. Acceleration may change mechanics; it may not silently change semantics.
