# Dr Moagi Bit-Wise World500 Kernel

## Status

This document defines the bit-exact companion model for the existing VMAD128 World Engine. It does **not** replace the VMAD128 sparse 3D address fabric. Instead, it specifies how a bounded `500 x 500 x 500` logical world maps bits, bytes, latent channels, residuals, memory, attention, and transactional state updates onto that runtime.

## 1. World footprint

The logical world contains

```text
500^3 = 125,000,000 agents
```

For 16 FP32 latent channels per agent:

```text
16 channels x 32 bits = 512 bits/agent
125,000,000 x 512 = 64,000,000,000 bits
                        8,000,000,000 bytes
                        ~7.451 GiB
```

The 8 GB figure is the **single-buffer latent payload only**. Candidate buffers, reconstruction buffers, residual/memory fields, indices, and allocator/page metadata increase resident or virtual storage beyond this base figure.

The runtime exposes these quantities as compile-time constants in:

```text
cpp_runtime/include/jarvisx/bitwise_world500.hpp
```

## 2. Deterministic address geometry

For coordinates `0 <= x,y,z < 500`, the canonical row-major address is

```text
A = x + 500*y + 500^2*z
```

with range

```text
0 <= A < 125,000,000.
```

The kernel implements both `linear_address()` and `coordinate_from_address()` and regression-tests their bijection at the domain boundary.

This bounded logical address is distinct from VMAD128. VMAD128 remains the outer sparse address fabric and can map a World500 cell into a larger virtual region, modality, and attribute namespace.

## 3. Bit ingestion and byte packing

Primitive logical input is binary:

```text
b_i in {0,1}.
```

Eight LSB-first logical bits are packed as

```text
Byte = sum(b_k * 2^k), k=0..7.
```

The companion kernel validates that every logical bit is exactly `0` or `1`; malformed bit streams are rejected rather than silently coerced.

Normalized byte-domain tensor values use

```text
x = Byte / 255.
```

## 4. Numeric latent domain versus physical bit representation

A critical distinction is enforced:

1. **Numeric reconstruction residual** measures value-space error, for example `abs(x - x_hat)` or a higher-level MSE/L1 objective.
2. **Representation XOR residual** compares encoded bit patterns, for example `bits(x) XOR bits(x_hat)`.

These are not interchangeable for FP32. Two nearby floating-point values may have several representation bits different, while a bit flip in a high-order field can have a large numeric effect. Therefore the commit quality objective should be defined in numeric space unless exact representation identity is explicitly required.

The kernel exposes both forms:

```text
numeric_abs_error(source, reconstructed)
fp32_xor_residual(source, reconstructed)
xor_residual(byte_source, byte_reconstructed)
byte_xor_error_rate(...)
```

## 5. Encoder and latent formation

At the model layer, an encoder may be represented as

```text
z_j = sigma(sum_i W_ji*x_i + b_j).
```

The existing World Engine already provides a bounded latent reduction micro-op (`EncLatVol`). The World500 layer therefore does not duplicate the encoder. It supplies bit/byte semantics around the existing reduction path.

## 6. Spatial permeation

The expression

```text
z' = 0.82*z + 0.18*mean(neighbours)
```

is a **real-valued latent-channel operation**, not a binary-bit operation: its output generally lies in a continuum rather than `{0,1}`.

The kernel names this explicitly as `permeate_channel()` and validates finite values and a mixing coefficient in `[0,1]`.

If a strictly binary cellular rule is required later, it should be implemented as a separate quantized transition operator after thresholding or probabilistic sampling, not by calling the real-valued mixture a bit.

## 7. Attention routing

For a 16-dimensional latent channel vector, the canonical scaled dot-product logit is

```text
logit(q,k) = dot(q,k) / sqrt(16).
```

Attention across multiple keys is then

```text
a = softmax(logits).
```

The kernel provides a numerically stable softmax and a 16-dimensional scaled-dot logit. The existing World Engine's `FuseAttn` micro-op remains the bounded byte-domain execution primitive.

## 8. Decode and reconstruction

The existing World Engine expands latent bytes through `DecPixVol` and stores the reconstructed result through its sparse VMAD-backed volume. World500 treats that output as the candidate reconstruction used by the two residual domains above.

Logical pipeline:

```text
bits -> bytes -> numeric tensor -> encode -> latent
     -> permeate -> attention -> decode -> reconstruction
```

## 9. Residual memory

A residual signal may feed an exponentially weighted memory field:

```text
Omega_(t+1) = rho*Omega_t + (1-rho)*R_t
```

where `R_t` must have a declared domain. Typical choices are numeric reconstruction error, signed numeric correction, or normalized XOR error density. Raw XOR masks should not be added directly to floating-point state without an explicit encoding/normalization rule.

The kernel provides `memory_update()` with `rho` constrained to `[0,1]`.

## 10. Latent velocity

A scalar channel form of the kinetic update is

```text
dZ/dt = -alpha*grad(E)
        + beta*attention_pressure
        + gamma*memory_pressure
        - delta*Z.
```

The kernel exposes this as `latent_velocity()` with finite-input and non-negative-coefficient guards.

## 11. Transactional commit/reject

The canonical quality gate is strict improvement:

```text
commit <=> E_candidate < E_current.
```

Equality is not improvement. Invalid, negative, NaN, or infinite error values are rejected by the companion helper rather than entering the transactional state machine.

This complements the existing World Engine's `Validate` + `CommitIf` candidate/authoritative bias transaction path.

## 12. Fixed-point semantics

Representation-level fixed point:

```text
B_out == B_in
B_in XOR B_out == 0.
```

Numeric fixed point additionally requires the selected state update to stop changing the latent state within its declared tolerance.

For a memory-coupled system, a complete fixed point must also satisfy the memory recurrence. Omitting `Omega` from the fixed-point equation is valid only when the memory contribution is zero, constant, or absorbed into the state/operator definition.

A more explicit coupled form is

```text
B_(t+1) = Lambda[D(A(P(E(B_t)))) + Omega_t]
Omega_(t+1) = rho*Omega_t + (1-rho)*R(B_t, B_(t+1)).
```

At a coupled fixed point `(B*, Omega*)`:

```text
B* = Lambda[D(A(P(E(B*)))) + Omega*]
Omega* = rho*Omega* + (1-rho)*R(B*,B*).
```

When `R(B*,B*) = 0` and `rho != 1`, the residual-driven component of `Omega*` converges to zero unless another memory source is defined.

## 13. End-to-end operational mapping

```text
Logical bits
  -> pack bytes
  -> normalize/tensorize
  -> EncLatVol
  -> spatial permeation
  -> FuseAttn
  -> DecPixVol
  -> numeric + XOR residual telemetry
  -> Omega update
  -> candidate evaluation
  -> Validate
  -> CommitIf / rollback
  -> next state
```

This is the operational interpretation of the bit-wise Dr Moagi World500 equation while preserving the repository's established VMAD128 sparse addressing and transactional execution model.
