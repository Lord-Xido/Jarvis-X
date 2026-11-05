# QSOL Kinetic 3D Runtime

## Status

Integration candidate. This document describes the deterministic software reference in `jarvisx.qsol_kinetic_runtime`. It does not claim hardware DMA bandwidth, zero-copy transport, trained neural reconstruction quality, or production device control.

## Objective

Operationalize the QSOL 3D bytecode architecture as a closed kinetic state machine:

```text
3D uint8 source
  -> bounded DMA read
  -> deterministic 3D cosine encoding
  -> source-anchored second-order latent dynamics
  -> 3D reconstruction
  -> reconstruction residual
  -> admission test
  -> COMMIT / ROLLBACK
  -> feedback
  -> repeat until convergence or cycle budget
```

The central authority invariant is:

```text
candidate != authoritative state
```

and the kinetic invariant is:

```text
adaptation cannot bypass verification or rollback
```

## 1. 64-bit instruction word

The reference exposes a packed QSOL word:

```text
opcode[63:56] | flags[55:48] | regdst[47:40] | payload[39:0]
```

A short spatial payload is:

```text
X[39:30] | Y[29:20] | Z[19:10] | IMM[9:0]
```

The deterministic test vector is:

```text
opcode = 0x10       DMA_BURST_ST
flags  = 0x0D
regdst = 1
x      = 32
y      = 64
z      = 16
imm    = 1023

word = 0x100D0108040043FF
```

This software reference models packing semantics; it is not yet a hardware fetch/decode pipeline.

## 2. DMA contract

The voxel substrate is explicitly `uint8`, so byte accounting matches the one-byte-per-voxel QSOL contract.

A descriptor is:

```text
D = (channel, origin_xyz, shape_xyz)
```

The reference supports up to eight channels and rejects out-of-bounds regions before transfer.

The implementation deliberately performs a NumPy copy. It models deterministic DMA transfer semantics and byte counters only. It does not claim physical zero-copy transport or a measured 1 TB/s fabric.

## 3. Deterministic 3D encoder

For source block

```text
X in R^(Nx x Ny x Nz)
```

uint8 values are normalized to `[-1, 1]` and flattened only after the spatial basis has been defined.

The encoder uses the first `d_z` low-frequency separable orthonormal cosine modes:

```text
z0 = B X
```

where each row of `B` is a product of one-dimensional cosine basis vectors over x, y, and z.

The latent source anchor `z0` is immutable for one run.

The reference decoder is:

```text
Xhat = B^T z
```

followed by reshape and uint8 quantization.

Because `d_z` may be smaller than the voxel count, this is a lossy projection reference rather than a universal invertible codec.

## 4. Kinetic latent law

The previous repeated-`tanh` loop had the trivial fixed point `z = 0`, so convergence could occur by erasing the latent signal.

The kinetic runtime instead keeps the source encoding inside the dynamics.

For latent state `z_k`, velocity `v_k`, residual memory `Omega_k`, and immutable source anchor `z0`:

```text
e_k       = z_k - z0
z_desired = z0 + beta tanh(e_k)

a_k = k_s (z_desired - z_k)
      - gamma v_k
      - mu Omega_k

v_(k+1) = clip(v_k + dt a_k, -v_max, v_max)
z_(k+1) = z_k + dt v_(k+1)

Omega_(k+1)
  = rho Omega_k
  + (1-rho)(z_(k+1)-z0)
```

with

```text
0 <= beta < 1.
```

At the source anchor:

```text
z_k = z0
=> e_k = 0
=> z_desired = z0
```

so `z0` is a fixed point of the source-conditioned dynamics.

This is a discrete reference specialization of the second-order form:

```text
M(z) z_ddot + Gamma z_dot + grad U(z) = F_control.
```

## 5. Verification and transaction semantics

For every kinetic cycle, the decoded candidate is scored against the immutable normalized source:

```text
MSE_k = mean((X - Xhat_k)^2)
```

and latent distance is:

```text
r_z = sqrt(mean((z_k - z0)^2)).
```

A candidate commits only when it is finite and does not regress against the current authoritative reconstruction:

```text
COMMIT_k = finite(candidate)
           and MSE_candidate <= MSE_authoritative + epsilon_regression
```

Otherwise:

```text
ROLLBACK
```

restores the last committed latent state and damps kinetic momentum.

The committed reconstruction error is therefore monotone non-increasing under the configured tolerance.

Convergence is:

```text
COMMIT_k = true
and r_z <= epsilon_z.
```

## 6. Runtime receipt

Every cycle records:

```text
cycle
committed
converged
candidate_mse
authoritative_mse
latent_rms
velocity_rms
bytes_transferred
major_phase
micro_phase
state_hash
```

The SHA-256 state hash binds the authoritative reconstruction bytes to the cycle number and principal error metrics. It is an integrity receipt, not a reversible encoding.

## 7. Toroidal/kinetic phases

The runtime maintains two bounded phases:

```text
phi_(k+1)   = (phi_k + omega_major) mod 2pi
theta_(k+1) = (theta_k + omega_micro) mod 2pi
```

They are currently deterministic orchestration coordinates and do not claim physical electromagnetic dynamics.

## 8. Operational state

The complete kinetic state can be represented as:

```text
S_k = (
  X,
  z0,
  z_k,
  v_k,
  Omega_k,
  Xhat_authoritative,
  MSE_authoritative,
  phi_k,
  theta_k,
  DMA_counters,
  cycle_k
)
```

One complete transition is:

```text
S_k
  -> candidate kinetic update
  -> decode
  -> reconstruction error
  -> verify
  -> COMMIT / ROLLBACK
  -> receipt
  -> S_(k+1)
```

## 9. Demo

Run:

```bash
python examples/qsol_kinetic_demo.py
```

The reference fixture is the sparse `8 x 8 x 8` uint8 volume used in the original QSOL emulator discussion.

With the documented default-style fixture (`latent_dim=64`, `latent_tolerance=1e-3`), the kinetic loop is expected to converge within the configured 160-cycle budget while retaining a non-zero source-conditioned latent state.

Exact performance depends on CPU/NumPy and is not a hardware throughput claim.

## 10. Hardware lowering path

The software reference defines semantics for a future accelerator lowering:

```text
QSOL control word
  -> DMA descriptor fabric
  -> voxel SRAM/VRAM
  -> encoder matrix/tensor core
  -> latent kinetic core
  -> decoder/geometry core
  -> residual verifier
  -> commit/rollback state bank
  -> execution receipt
```

A hardware implementation must separately measure:

```text
peak bandwidth
effective bandwidth
DMA efficiency
voxel/s
latent updates/s
reconstructions/s
commit rate
rollback rate
power
energy per verified execution
```

The architectural roofline remains:

```text
throughput_system
  = min(
      throughput_DMA,
      throughput_encode,
      throughput_kinetic,
      throughput_decode,
      throughput_verify,
      throughput_commit
    ).
```

## 11. Jarvis-X integration boundary

The QSOL kinetic runtime is a candidate-producing research accelerator. It does not supersede Jarvis-X authority semantics.

The intended composition is:

```text
ANN / planner / swarm
  -> Jarvis-X capability projection
  -> bounded QSOL kinetic execution
  -> kinetic receipt
  -> SystemRuntime verification
  -> COMMIT / REJECT
  -> optional Verified Execution settlement
```

Thus increasing kinetic sophistication does not increase authority automatically.
