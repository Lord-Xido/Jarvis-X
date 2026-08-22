# Dr Moagi 3D OS

## Status

**Bounded full-stack operating control plane for Jarvis-X.**

This track turns the Dr Moagi sparse 3D auto-encoding/decoding runtime into an end-to-end service with an authoritative kernel state, a Uint64 bit-plane substrate, inward spatial folding, transactional auto-execution, fixed-point stabilization, checkpoint persistence, an auto-run scheduler, FastAPI control plane, browser 3D UI, Prometheus-compatible metrics, CLI, container image and CI smoke tests.

It is called an "OS" because it owns lifecycle, state, scheduling, persistence and control-plane responsibilities for the Dr Moagi runtime. It is **not** a replacement for Linux/Windows/macOS, does not execute arbitrary host commands and does not perform unrestricted self-modification.

## End-to-end execution path

```text
external sparse 3D field
        |
        v
+-----------------------+
| OS boot + input gate  |
+-----------+-----------+
            |
            v
+-----------------------+
| voxel activation map  |  magnitude threshold -> Z in {0,1}
+-----------+-----------+
            |
            v
+-----------------------+
| sparse Uint64 packer  |  B[x,y,q], q = floor(z/64)
+-----------+-----------+
            |
            v
+-----------------------+
| inward fold/attenuate |  centroid contraction + radial attenuation
+-----------+-----------+
            |
            v
+-----------------------+
| 3D autoexec engine    |  parse -> encode -> decode -> field step
+-----------+-----------+
            |
            v
+-----------------------+
| Pi_Lambda verification|  MSE + finite state + active-cell budget
+-----------+-----------+
      reject|             |accept
         rollback         v
                   +-----------------------+
                   | DM-vOmegaXi+ pass     |
                   | Phi/Lambda/Omega/Theta|
                   +-----------+-----------+
                               |
                               v
                   +-----------------------+
                   | authoritative commit  |
                   +-----------+-----------+
                               |
                  +------------+------------+
                  |                         |
                  v                         v
          checkpoint + journal        next closed-loop cycle
```

## 1. OS kernel

`DrMoagiOSKernel` owns the authoritative state:

```text
M_t = (Psi_t, B_t, pi_t, cycle_t, journal_t, lifecycle_t)
```

Lifecycle states are:

```text
offline -> ready -> running
                  |      |
                  +--> halted on rejected transactional transition
```

A rejected candidate never replaces the authoritative sparse state.

## 2. Uint64 bit-plane substrate

For a logical side length `N`, Z is packed in 64-bit words:

```text
q = floor(z / 64)
r = z mod 64
B[x,y,q] |= 1 << r
```

The general logical shape is therefore:

```text
B in U64^(N x N x ceil(N/64))
```

Only non-zero words are resident. `SparseBitPlane3D` reports:

- active bits;
- logical bits;
- resident packed words;
- logical packed words;
- binary density;
- occupancy entropy;
- normalized Hamming phase velocity;
- normalized kinetic energy.

The implementation uses Python integers constrained to the `uint64` range as the portable reference representation.

## 3. Inward spatial fold

For centroid

```text
c = ((N-1)/2, (N-1)/2, (N-1)/2)
```

the coordinate contraction is

```text
r' = c + (1-kappa)(r-c),  0 <= kappa < 1
```

and radial attenuation is

```text
x'(r) = x(r) * exp(-beta * ||r-c||^2 / R_max^2)
```

Coordinate collisions retain the value with greatest absolute magnitude. This prevents collision-driven amplitude amplification and makes the reference transform deterministic.

## 4. Auto-encoding/decoding execution

The folded sparse field enters the existing `DrMoagiAutoExecutionEngine`:

```text
parse -> sparse block encode -> support-only decode -> residual field
      -> transactional field update -> validation -> bounded policy search
```

The policy remains bounded to:

```text
pi = (block_size, quantization, prune_epsilon)
```

No arbitrary native code is mutated.

## 5. Fixed-point inward pass

Accepted autoexec candidates pass through the locked `DM-vOmegaXi+` operator stack:

```text
Psi -> Phi -> Lambda^-1 -> Omega -> Theta
```

The fixed-point residual is measured, not declared. A converged state is one whose bounded update residual meets the configured tolerance.

## 6. Closed-loop kinetics

For packed bit states `B_t` and `B_(t-1)`, phase velocity is normalized Hamming distance:

```text
v_t = popcount(B_t XOR B_(t-1)) / N^3
```

Bit density is:

```text
rho_t = active_bits / N^3
```

and normalized kinetic energy is:

```text
E_k(t) = 0.5 * rho_t * v_t^2
```

Binary occupancy entropy is:

```text
H_t = -rho_t log2(rho_t) - (1-rho_t) log2(1-rho_t)
```

These are telemetry. Convergence is not inferred solely from entropy decay; the fixed-point engine supplies an independent residual criterion.

## 7. Persistence and audit

Every attempted outer OS cycle is appended to the SHA-256 hash-chain journal inherited from the autoexec layer. Successful state is also checkpointed atomically to:

```text
state/dr-moagi-os/checkpoint.json
state/dr-moagi-os/os-journal.jsonl
```

On restore, checkpoint hash, state hash and journal head must agree before the state is accepted.

## 8. Auto-execution scheduler

The scheduler runs only internal Dr Moagi cycles at a bounded interval:

```text
start_autorun(interval)
    -> wait interval
    -> step()
    -> stop on rejection or fixed-point convergence
```

It does not spawn shells, arbitrary programs or uncontrolled background agents.

## 9. API

Run:

```bash
jarvisx-dr-moagi-os serve --host 0.0.0.0 --port 10000
```

Endpoints:

| Endpoint | Purpose |
|---|---|
| `GET /` | Browser 3D control plane |
| `GET /healthz` | Health probe |
| `POST /v1/os/boot` | Boot and optionally restore checkpoint |
| `POST /v1/os/demo` | Load deterministic demo field |
| `POST /v1/os/load` | Load sparse 3D field |
| `POST /v1/os/step` | One transactional cycle |
| `POST /v1/os/run` | Bounded multi-cycle run |
| `POST /v1/os/autorun/start` | Start internal scheduler |
| `POST /v1/os/autorun/stop` | Stop scheduler |
| `POST /v1/os/halt/reset` | Clear halted lifecycle after inspection |
| `GET /v1/os/status` | Kernel state and telemetry |
| `GET /v1/os/snapshot` | Bounded sparse state sample |
| `GET /v1/os/bitplane` | Packed-word sample and metrics |
| `GET /metrics` | Prometheus-compatible gauges |

Example input:

```json
{
  "field": [
    {"x": 30, "y": 32, "z": 32, "value": 0.75},
    {"x": 32, "y": 32, "z": 32, "value": 1.0},
    {"x": 34, "y": 32, "z": 32, "value": 0.75}
  ]
}
```

## 10. Browser UI

The root route serves a dependency-free browser application that:

- renders the current sparse state as a rotatable 3D point field;
- boots and loads the deterministic demo;
- executes one cycle or bounded runs;
- starts/stops auto-run;
- displays density, phase velocity, entropy, kinetic energy, packed-word occupancy, reconstruction MSE and fixed-point residual;
- polls the authoritative API rather than simulating hidden state in the browser.

## 11. Container deployment

Build:

```bash
docker build -f deploy/dr_moagi_os/Dockerfile -t jarvisx-dr-moagi-os .
```

Run:

```bash
docker run --rm \
  -p 10000:10000 \
  -v jarvisx-state:/var/lib/jarvisx \
  -e JARVISX_STATE_DIR=/var/lib/jarvisx \
  jarvisx-dr-moagi-os
```

Or:

```bash
docker compose -f compose.dr-moagi-os.yml up --build
```

Then open `http://localhost:10000/`.

## 12. Local CLI

```bash
jarvisx-dr-moagi-os demo --cycles 8 --pretty
```

or:

```bash
jarvisx-dr-moagi-os run-file field.json --side 64 --cycles 8 --pretty
```

## Operational boundary

This reference system demonstrates a full end-to-end runtime and service boundary, but it does not claim that a sparse control plane is a general-purpose host OS, that bit packing creates superlinear information capacity, or that fixed-point self-consistency proves external-world correctness. Performance, reconstruction quality and convergence behavior remain empirical properties that must be benchmarked.
