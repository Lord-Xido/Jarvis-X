# Jarvis-X

Jarvis-X is a deterministic, auditable virtual machine with a reflex control
layer, policy gate, transactional ledger, and experimental 3D geometric and
multimodal reference runtimes.

## Install

```bash
git clone https://github.com/Lord-Xido/Jarvis-X.git
cd Jarvis-X
pip install -r requirements.txt
pip install .
```

## Unified execution model

The VM is the authoritative control plane. Bytecode, geometric reconstruction,
and MM3D multimodal cycles pass through policy, journaling, and trace boundaries.

```text
Assembly -> Parser -> Assembler -> CodexVM -> Registers / Ledger / Trace
                                      |
                                      +-> V3D.PERMEATE
                                      |      |
                                      |   spatial visual memory
                                      |
                                      +-> MM3D.CYCLE
                                             |
                                  Z8 QCA -> factorized metric
                                             |
                                  geometric VQ encode/decode
                                             |
                                  classical bounded exploration
                                             |
                                      Lambda -> Omega -> Theta
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

## MM3D-AED-BCE-Ω⁴ operational kernel

`src/jarvisx/mm3d_omega4.py` preserves the intended continuous cycle:

```text
Psi input
  -> Z8 quantized cellular substrate
  -> factorized geometric encoder
  -> vector-quantized 3D latent code
  -> bounded classical Phi-QAS exploration
  -> geometric decoder
  -> Lambda projection
  -> deterministic Omega hash chain
  -> bounded text/image/audio/video projections
```

The operational profile deliberately separates declared scaling targets from
allocated runtime state:

- `50T` remains a conceptual total-parameter target;
- `0.5%` active means `250B`, not `500B`;
- the default reference engine allocates a small, reported NumPy model;
- configuration validation rejects profiles above the declared memory guard;
- the 13.7 ms cycle goal is reported as a measured target, never assumed true.

The voxel layout is exactly 384 bits: 16 token bytes, 12 visual bytes, 8 audio
bytes, 6 motion bytes, one float32 attention weight, and two flag bytes. The
Xi-cube is a contiguous NumPy structured array rather than millions of Python
objects.

Run one VM-authoritative cycle:

```bash
jarvisx mm3d-cycle
jarvisx mm3d-cycle 65536
```

The command reports actual parameter count, conceptual parameter targets,
allocated bytes, latent and projection shapes, Omega head, state hash, measured
cycle time, and whether the 13.7 ms target was met.

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
POST /mm3d-cycle
```

These development servers bind to loopback by default. Production exposure
requires an external authentication, authorisation, TLS, rate-limiting, and
process-isolation boundary.

## Invariants

1. Equal initial state and input must produce equal reference results.
2. Candidate mechanics must start from identical memory snapshots.
3. A candidate cannot commit unless its declared objective improves.
4. Geometric and MM3D execution can be blocked by the Lambda policy gate.
5. Conceptual capacity must not be represented as physically allocated capacity.
6. Classical Phi-QAS exploration must not be represented as quantum execution.
7. Acceleration may change mechanics; it may not silently change semantics.
