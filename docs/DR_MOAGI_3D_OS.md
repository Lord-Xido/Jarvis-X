# Dr Moagi 3D OS

## Status

**End-to-end bounded sparse adaptive operating control plane for Jarvis-X.**

The Dr Moagi 3D OS owns lifecycle, authoritative sparse state, bit-plane execution,
inward folding, codec execution, Deep Distiller adaptation, fixed-point refinement,
transaction verification, exact Morton transport, checkpoint recovery, scheduling,
API/CLI control, telemetry, browser visualization, container deployment and CI.

It is an application/runtime OS for the Dr Moagi computational substrate. It is
**not** a replacement for Linux, Windows or macOS. It does not execute arbitrary
host commands, spawn unrestricted agents or rewrite its own source code.

## End-to-end execution path

```text
external sparse 3D state
        |
        v
+--------------------------+
| Boot / parser input gate |
+------------+-------------+
             |
             v
+--------------------------+
| Sparse Uint64 bit-plane  |
+------------+-------------+
             |
             v
+--------------------------+
| Inward fold + attenuation|
+------------+-------------+
             |
             v
+--------------------------+
| AutoExec codec/runtime   |
| encode -> decode -> field|
+------------+-------------+
             |
             v
+--------------------------+
| Deep Distiller (DM-DD)   |
| Z, residual, Omega, Theta|
+------------+-------------+
             |
             v
+--------------------------+
| DM-vOmegaXi+ fixed point |
+------------+-------------+
             |
             v
+--------------------------+
| Resource/Pi_Lambda gate  |
+------------+-------------+
             |
             v
+--------------------------+
| DMOS2 Morton exact packet|
| encode -> decode verify  |
+------------+-------------+
      reject |             | accept
   rollback  |             v
             |   +-----------------------+
             +---| Atomic OS commit      |
                 +-----------+-----------+
                             |
                +------------+-------------+
                |                          |
                v                          v
       checkpoint + journal      API/UI/export/next cycle
```

The system-wide invariant is:

```text
PROVISIONAL != AUTHORITATIVE
```

until all enabled stages have passed.

## 1. Authoritative state

The runtime state is conceptually:

```text
M_t = (
  X_t,           sparse scalar 3D state
  B_t,           sparse Uint64 occupancy plane
  Omega_t,       persistent residual memory
  Theta_t,       Deep Distiller parameters
  pi_t,          bounded codec/runtime policy
  cycle_t,
  journal_t,
  lifecycle_t
)
```

A rejected outer cycle leaves `X`, `Omega` and `Theta` unchanged.

## 2. Sparse Uint64 substrate

For logical side length `N`, binary occupancy is packed along Z:

```text
q = floor(z / 64)
r = z mod 64
B[x,y,q] |= 1 << r
```

Only non-zero words are resident. The logical volume may therefore be much larger
than the materialized active support.

Measured logical telemetry includes:

- active/logical bits;
- resident/logical packed words;
- occupancy density;
- binary entropy;
- normalized Hamming phase velocity;
- normalized logical kinetic metric.

These are computational metrics, not physical joules or electromagnetic telemetry.

## 3. Inward spatial fold

For centroid `c`, sparse coordinates contract as:

```text
r' = c + (1-kappa)(r-c),  0 <= kappa < 1
```

with radial attenuation:

```text
x'(r) = x(r) * exp(-beta * ||r-c||^2 / R_max^2)
```

Collisions deterministically retain the greatest absolute magnitude, preventing
collision-driven amplification.

## 4. AutoExec stage

The folded sparse field enters the bounded `DrMoagiAutoExecutionEngine`:

```text
parse -> block encode -> support-only decode -> residual field
      -> field step -> validator -> bounded policy search
```

The adaptive policy is constrained to:

```text
pi = (block_size, quantization, prune_epsilon)
```

No arbitrary native code mutation occurs.

## 5. Deep Distiller integration

DM-DD is now inside the OS transaction rather than operating as a disconnected
product surface.

```text
Z_t        = E_Theta(X_t)
X_hat_t    = D_Theta(Z_t)
E_t        = X_t - X_hat_t
Omega_t+1  = rho Omega_t + (1-rho) E_t
Theta'_t+1 = Theta_t - eta grad_Theta ||E_t||^2
X'_t+1     = X_hat_t + omega Omega_t+1
```

The OS executes this in an isolated staging instance. A later fixed-point,
resource or transport failure cannot partially commit the staged adaptive state.
Only after the whole outer OS transaction passes are `X`, `Omega` and `Theta`
promoted together.

The reference DM-DD model remains deliberately small and measurable. A larger
learned encoder/decoder may replace it without changing the transaction boundary.

## 6. Fixed-point stage

Accepted DM-DD candidates may pass through the existing `DM-vOmegaXi+` bounded
fixed-point runtime. The residual is measured and the candidate remains provisional
until the full OS transaction completes.

## 7. Exact Morton transport: DMOS2

`SparseStateCodec3D` provides a lossless byte packet for committed sparse state:

```text
sparse coordinates
 -> Morton sort
 -> Morton delta varints
 -> IEEE-754 float64 values
 -> DEFLATE
 -> SHA-256 packet checksum
```

Format identifier:

```text
DMOS2
```

Before commit, the OS performs:

```text
candidate -> DMOS2 encode -> DMOS2 decode -> state-hash equality
```

so persistence/network serialization is part of the transaction verification.
The packet is exact for the encoded float64 state; it is distinct from the
quantized research entropy codec in the frontier runtime.

## 8. Checkpoint recovery

Checkpoint version 2 persists:

- authoritative sparse state;
- OS cycle;
- runtime policy;
- DM-DD iteration;
- DM-DD `Theta`;
- DM-DD `Omega`;
- exact transport metadata;
- state hash;
- journal head.

Version 1 checkpoints remain readable. Version 2 restores adaptive state rather
than restarting learning from default parameters.

Files:

```text
state/dr-moagi-os/checkpoint.json
state/dr-moagi-os/os-journal.jsonl
```

Checkpoint state hash, transport hash and journal head are verified before the
restored state is admitted.

## 9. Scheduler

The scheduler advances only internal bounded OS cycles:

```text
start_autorun(interval)
 -> step()
 -> stop on rejection or configured convergence
```

It does not provide a shell-command execution primitive.

## 10. API

Run:

```bash
jarvisx-dr-moagi-os serve --host 0.0.0.0 --port 10000
```

| Endpoint | Purpose |
|---|---|
| `GET /` | Live Three.js control plane |
| `GET /healthz` | Health probe |
| `GET /v1/os/capabilities` | Runtime capability contract |
| `POST /v1/os/boot` | Boot / restore checkpoint |
| `POST /v1/os/demo` | Load deterministic demo state |
| `POST /v1/os/load` | Load sparse 3D state |
| `POST /v1/os/step` | One complete transactional cycle |
| `POST /v1/os/run` | Bounded multi-cycle execution |
| `POST /v1/os/autorun/start` | Start internal scheduler |
| `POST /v1/os/autorun/stop` | Stop internal scheduler |
| `POST /v1/os/halt/reset` | Clear inspected halt state |
| `GET /v1/os/status` | State, DM-DD and transport telemetry |
| `GET /v1/os/snapshot` | Bounded sparse state sample |
| `GET /v1/os/bitplane` | Packed occupancy sample |
| `GET /v1/os/export` | Export exact DMOS2 packet as base64 |
| `POST /v1/os/import` | Verify/import exact DMOS2 packet |
| `GET /metrics` | Prometheus-compatible gauges |

## 11. CLI

Local execution:

```bash
jarvisx-dr-moagi-os demo --cycles 8 --pretty
jarvisx-dr-moagi-os run-file field.json --side 64 --cycles 8 --pretty
```

Exact transport:

```bash
jarvisx-dr-moagi-os pack field.json state.dmos --side 64 --pretty
jarvisx-dr-moagi-os inspect-packet state.dmos --pretty
jarvisx-dr-moagi-os run-packet state.dmos --cycles 8 --pretty
```

## 12. Browser control plane

The Three.js UI renders measured sparse state while keeping shells, rings, glow and
particle motion explicitly visual. It polls the authoritative service rather than
maintaining a hidden browser-side simulation of OS state.

## 13. Container deployment

```bash
docker build -f deploy/dr_moagi_os/Dockerfile -t jarvisx-dr-moagi-os .

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

## Operational boundary

This is a complete end-to-end **domain-specific user-space operating runtime**:
it owns its internal computational state, scheduling, adaptation, validation,
persistence, transport and control surfaces. It does not claim host-kernel status,
unbounded storage, arbitrary-code execution, physical electromagnetic computation,
or general intelligence. Those remain separate engineering/research questions.
