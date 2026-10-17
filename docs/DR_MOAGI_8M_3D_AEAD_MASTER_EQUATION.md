# Dr Moagi 8M-Pixel 3D AE/AD Master Equation

This document defines the operational auto-encoding/decoding abstraction for
the exact `4000 x 2000 = 8,000,000`-pixel geometric framebuffer path.

The implementation is intentionally split into an authoritative path and a
shadow candidate path. Recursive evolution is never assumed to improve the
state: CTR measures it and either commits it or rolls it back.

## Unified operator

Let the total system state be

```text
S_t = [X_t, Z_t, Z*_t, Xhat_t, e_t, Omega_t, Theta_t, F_t, W_t].
```

The master operator is

```text
G_DM,8M^3D =
  Pi_Lambda
  o U_(Theta,Omega)
  o C_CTR
  o S_sensor
  o D_panel
  o P_present
  o F_FBO
  o H_frag
  o Z_depth
  o R_raster
  o Pi_camera
  o D_phi^3D
  o Phi_(T3,Omega)^K
  o E_theta^3D
  o B_3D
```

and the recurrent state equation is

```text
S_(t+1) = G_DM,8M^3D(S_t, X_t).
```

A total fixed point satisfies

```text
S* = G_DM,8M^3D(S*, X).
```

## Exact pixel domain

The default runtime evaluates

```text
W = 4000
H = 2000
N_pix = W * H = 8,000,000.
```

Pixels are streamed rather than retained as a dense floating-point frame, so
the reference implementation verifies the full logical pixel domain while
keeping resident state bounded.

## 2D -> periodic 3D fold

Flatten

```text
p = y*W + x.
```

For the reference latent edge `L=32`:

```text
i = p mod L
j = floor(p/L) mod L
k = floor(p/L^2) mod L
```

which maps the frame into

```text
T^3 ~= Z_L x Z_L x Z_L
```

with periodic wraparound.

The default compression ratio is

```text
8,000,000 / 32^3 = 244.140625 : 1.
```

## Encoder

The reference encoder is a deterministic aggregation operator:

```text
Z_ijk = mean { X_xy | B_3D(x,y) = (i,j,k) }.
```

This gives a concrete executable autoencoder baseline without requiring an
external ML framework.

## Inward fold

A shadow candidate is evolved on the 3-torus:

```text
Z' = c + lambda * (Z + kappa * Delta_T3(Z) - c)
```

with

```text
lambda = 0.985
kappa  = 0.020
K      = 6.
```

The six-neighbour toroidal Laplacian is

```text
Delta_T3 Z(i,j,k) =
    Z(i+1,j,k) + Z(i-1,j,k)
  + Z(i,j+1,k) + Z(i,j-1,k)
  + Z(i,j,k+1) + Z(i,j,k-1)
  - 6 Z(i,j,k).
```

Every coordinate wraps modulo `L`.

## Decoder

The deterministic decoder uses the inverse address relation:

```text
Xhat_xy = Z_(B_3D(x,y)).
```

The runtime computes full-frame MSE, MAE and maximum absolute reconstruction
error over all logical pixels.

## CTR authority boundary

Let

```text
L_base = MSE(X, D(E(X)))
L_cand = MSE(X, D(Phi^K(E(X)))).
```

CTR applies

```text
Z_authoritative =
  Z_candidate   if L_cand <= L_base + delta
  Z_encoded     otherwise.
```

The current numerical slack is `delta = 1e-15`.

Therefore recursive evolution is transactional: a non-improving fold is
observed and rejected rather than silently committed.

## Persistent memory

For accepted/authoritative latent state `Z_t`:

```text
Omega_(t+1) = beta*Omega_t + (1-beta)*Z_t
beta = 0.88.
```

This provides bounded state continuity between cycles.

## Executable verification contract

A run is successful only if:

1. the full configured pixel domain is encoded and decoded;
2. all authoritative latent values remain in `[0,1]`;
3. Omega memory remains in `[0,1]`;
4. reconstruction metrics are finite;
5. the inward fixed-point residual is finite;
6. CTR never authorizes a reconstruction regression beyond the configured slack;
7. failure propagates through the process exit code.

## Build and run

```bash
cmake -S cpp_runtime -B build/cpp-runtime \
  -DCMAKE_BUILD_TYPE=Release \
  -DJARVISX_BUILD_GL_VISUALIZER=OFF

cmake --build build/cpp-runtime \
  --target jarvisx-dr-moagi-8m-aead \
  --parallel

./build/cpp-runtime/DrMoagi-8M-3D-AEAD 4000 2000 2
```

The default execution is the exact eight-million-pixel configuration.

## Relationship to the rendering and kinetics layers

This operator is the codec/control layer. It composes naturally with:

```text
latent 3D field
 -> geometric decoder
 -> projection/raster/depth/shading
 -> framebuffer/present
 -> panel/sensor
 -> CTR residual
 -> Omega/Theta update
 -> latent field.
```

The present C++ reference verifies the `B3D -> encode -> inward fold -> decode
-> CTR -> memory -> recur` core. GPU rasterization and physical panel transfer
remain separate implementation layers and can be attached without changing the
master state equation.
