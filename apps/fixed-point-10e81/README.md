# Jarvis-X 10^81 Sparse Fixed-Point 3D Emulator

## Logical scale

The notation is operationalized as three axes of length `(10^9)^3 = 10^27`:

```text
X: 0 ... 10^27 - 1
Y: 0 ... 10^27 - 1
Z: 0 ... 10^27 - 1
```

The Cartesian product has exactly `10^81` logical sites.

Each axis needs 90 bits because `2^89 < 10^27 < 2^90`. A packed row-major logical coordinate therefore occupies at most 270 bits:

`A(x,y,z) = x + 10^27 * (y + 10^27 * z)`.

JavaScript `BigInt` is used for exact address arithmetic.

## Physical execution boundary

The emulator does not allocate `10^81` voxels. It materializes a bounded active support (1,024 points by default). Coordinates are generated deterministically from the virtual address domain. This is sparse virtualization, not dense residency.

## Fixed-point recurrence

For immutable sampled target field `X`, latent state `Z`, residual `R`, and temporal memory `Omega`:

```text
Xhat_k = tanh(Z_k)
R_k = X - Xhat_k
Zcand = Z_k + eta * (0.80 * R_k + 0.20 * Omega_k)
Lcand = mean((X - tanh(Zcand))^2)
```

The candidate does not become authoritative immediately:

```text
COMMIT   iff Lcand <= Lcurrent
ROLLBACK otherwise
```

After verification:

`Omega <- rho * Omega + (1-rho) * R`.

This is a bounded executable instance of the Jarvis-X recurrent grammar:

`INGEST -> ENCODE -> INWARD -> DECODE -> CONTRAST -> RECKON -> VERIFY -> MEMORY -> RECUR`.

## Run

```bash
python -m http.server 8000 --directory apps/fixed-point-10e81
```

Open `http://localhost:8000`.

## Test

```bash
node --test apps/fixed-point-10e81/test_model.mjs
```

The tests verify exact `10^81` geometry, 270-bit address round trips, deterministic sparse support, monotone accepted MSE, rollback of destabilizing candidates, and remapping semantics.

## Capability boundary

`10^81` is logical cardinality only. The app does not claim `10^81` physical operations, resident voxels, hardware workers, or measured throughput. The visual cube is a projection of the bounded active support.
