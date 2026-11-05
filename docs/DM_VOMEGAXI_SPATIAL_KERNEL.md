# DM-vΩΞ⁺ Spatial Kernel (SK-3D)

## Status

SK-3D is an executable, deterministic research reference for the submitted three-axis Dr Moagi equation. It maps the symbolic geometry into bounded software semantics and a render-neutral tetrahedral manifold. It is **not** a claim that software directly alters physical reality, that recursion is literally infinite, or that a 6.4 GB ROM image must be densely allocated.

## Three-axis geometry

For committed state \(S_t=(\Psi_t,\Theta_t,\Xi_t,\Omega_t)\), the decoded coordinate is

\[
\mathbf r_t=
\begin{bmatrix}
X_t\\Y_t\\Z_t
\end{bmatrix}
=
\begin{bmatrix}
\Phi\Xi_t\\
\Psi_t\Theta_t\\
\Lambda^{\Omega_t}
\end{bmatrix}.
\]

The contraction/singularity scalar is

\[
s_t=\frac{\Psi_t\Phi}{\Lambda^{\Omega_t}}.
\]

The reference uses finite positive `recursion_base` for \(\Lambda\) and a bounded projection manifold instead of literal infinity.

## Bounded recurrence

One cycle evaluates a concrete form of the canonical recurrence

\[
\Xi_{t+1}=\Pi_{\rho}
\left[
\Xi_t + P_t - E_t + \Omega_{t+1}
+\kappa R_t
-\eta\nabla_\Theta L_t
-\zeta\nabla_H C_t
\right],
\]

where

\[
E_t=X_t^{obs}-s_t,
\]

and the bounded residual memory is

\[
\Omega_{t+1}=\Pi_{\rho}
\left[d_\Omega\Omega_t+g_\Omega E_t\right].
\]

`Πρ` is an explicit clamp to `[-projection_limit, projection_limit]`. This makes every persistent symbolic coordinate finite and machine representable.

## Bytecode

Each micro-instruction is exactly 64 bits:

```text
[ opcode:8 ][ flags:8 ][ operand:16 ][ immediate:32 ]
```

The canonical program is:

```text
ENCODE
EVOLVE
PROJECT
DECODE
ROTATE
PULSE
SEAL
HALT
```

Eight instructions therefore occupy 64 physical bytes. The configured `6_400_000_000` byte ROM value is a logical capacity/namespace, not a dense allocation requirement.

## Transaction boundary

Every program executes on a shadow working state. Authoritative state changes only after `HALT` and only if all values remain finite and satisfy the projection manifold. Invalid inputs, malformed bytecode, instruction-budget violations and cycle-budget violations fail before partial state mutation.

## Manifold decoder

`DECODE` emits four semantic vertices:

- Ψ vertex: `(0, Ψ, 0)`;
- Φ vertex: `(Φ, 0, 0)`;
- Λ vertex: `(0, 0, Λ^Ω)`;
- Ω/Θ/Ξ closure vertex: `(-Ω, -Θ, -Ξ)`.

`ROTATE` applies a deterministic Rodrigues rotation around `(1,1,1)/√3`. Angular velocity is

\[
\omega_t^{rot}=v(\Omega_t\Xi_t),
\]

and `PULSE` exposes reconstruction mismatch as

\[
A_t=A_0+\alpha|E_t|.
\]

The resulting `SpatialFrame` is renderer-neutral and can be consumed by a browser/WebGL, native graphics or telemetry frontend without making rendering part of authoritative execution.

## Integrity

`SEAL` computes a SHA-256 digest over the program image, configuration-defining geometry constants and complete persistent state encoded with deterministic big-endian IEEE-754 values. Equal initial states and equal inputs therefore produce equal state/frame digests.

## Run

```bash
python -m pip install -e ".[dev]"
pytest tests/test_dm_spatial_kernel.py
python examples/dm_spatial_kernel.py
```

## Implemented versus proposed

Implemented here:

- 64-bit spatial microcode;
- bounded Ψ/Θ/Ξ/Ω state evolution;
- exact X/Y/Z projection;
- residual-memory update;
- Λ-style projection gate;
- tetrahedral manifold decoding;
- rotation and pulse telemetry;
- deterministic state sealing;
- cycle/instruction budgets;
- transaction-before-commit semantics.

Not implemented or claimed here:

- literal infinite recursion;
- physical 6.4 GB ROM hardware;
- autonomous modification of arbitrary native code;
- consciousness or self-awareness;
- physical-world effects from symbolic execution;
- learned multimodal encoders/decoders;
- GPU acceleration or production performance guarantees.
