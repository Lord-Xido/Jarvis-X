# Dr Moagi 6400x1000 3D Geometric Coordinating AE/AD Engine

This C++17 runtime is the executable reference for the 6400-panel / 1000-programmable-iteration Dr Moagi architecture.

## Geometry

    20 x 20 x 16 = 6400 logical panels
     5 x  5 x  4 = 100 clusters
     4 x  4 x  4 = 64 panels per cluster

Every panel has a spatial coordinate and persistent input, latent, Omega memory, neighbour, reconstruction and residual state.

Default logical program domain:

    6400 panels x 1000 programmable slots
    = 6,400,000 logical slots per macrocycle

The executable code image is shared. Panel state is replicated. A bounded worker pool maps the 6400 logical cells onto the hardware actually available.

## End-to-end invariant

    ingest
     -> six-neighbour 3D exchange
     -> encode
     -> programmable latent kernel
     -> fixed-point residual
     -> inward residual contraction
     -> core correction
     -> outward prolongation
     -> residual acceleration
     -> decode
     -> reconstruction contrast
     -> verify/correct
     -> Omega memory update
     -> 64-panel cluster reductions
     -> 100-cluster global reduction
     -> runtime policy adaptation
     -> recur

Inward hierarchy:

    6400 -> 800 -> 100 -> 18 -> 4 -> 1

The correction returns outward through the inverse hierarchy.

## Build

From the repository root:

    cmake -S cpp_runtime -B build/cpp-runtime -DJARVISX_BUILD_GL_VISUALIZER=OFF
    cmake --build build/cpp-runtime --target jarvisx-dr-moagi-6400x1000 -j

Canonical 1000-slot run:

    ./build/cpp-runtime/DrMoagi-6400x1000-3D --cycles 4 --iterations 1000

Short smoke run:

    ./build/cpp-runtime/DrMoagi-6400x1000-3D --cycles 1 --iterations 64 --workers 2

## Runtime receipts

Each macrocycle reports:

- logical: total programmable slots;
- executed: actual kernel operations after activity/convergence gates;
- active: panels requiring refinement;
- exec_fraction: executed/logical;
- mse: global reconstruction mean-squared error;
- fp: fixed-point residual before inward correction;
- corrections: verification corrections;
- relax: adaptive refinement relaxation;
- rho: recurrent-memory coefficient;
- checksum: aggregate reconstruction checksum;
- ms: measured wall time.

## Mathematical core

Neighbour message:

    N_i = mean_{j in N6(i)} Z_j

Encoding:

    Z_i^0 = tanh(0.58 X_i + 0.24 Omega_i + 0.18 N_i)

Representative programmable update:

    z_k <- z_k + lambda [
        tanh(0.48 z_k + 0.18 z_j + 0.14 n_k + 0.11 omega_k + 0.09 x_k) - z_k
    ]

Fixed-point residual and inward correction:

    Q = F(Z) - Z
    Delta Z = P_up C_core R_down(Q)

Reconstruction and memory:

    Xhat_i = tanh(0.90 Z_i + 0.10 Omega_i)
    R_i = X_i - Xhat_i
    Omega_i' = rho Omega_i + (1-rho)(Z_i + 0.25 R_i)

## Boundary

The runtime keeps latent states bounded and verifies finiteness before promotion. Runtime adaptation changes bounded policy parameters, not arbitrary source code.

Logical iteration compression and wall-clock acceleration are separate metrics. A 1000x physical speedup is not assumed; it must be established by benchmark against a defined baseline on the same hardware.
