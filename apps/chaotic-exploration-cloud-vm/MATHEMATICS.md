# JARVIS X Phi3D — exact state-transition specification (v2)

## Operational claim and its limits

The implemention in `vm.cpp` computes a deterministic, bounded **application-level** 3D state transition on the CPU. Its fixed bytecode is:

`INGEST 10; LORENZ 20; ATTEND_GRU 30; FOLD_3D 40; PLAN 50; VERIFY 60; HALT ff`.

`INGEST` executes **once**. Every subsequent iteration executes the remaining stages. `ATTEND_GRU` is a legacy instruction name: the operation is a hand-written scalar attention/gated map and has **no trained weights, attention heads, GRU parameter matrices, or backpropagation**. It is not a pretrained AI model, cryptographic processor, general-purpose virtual-machine hypervisor, or distributed ANN swarm.

## 1. State and domain

For side `N in [4,32]` and periodic coordinates `i=(x,y,z)` in the discrete 3-torus `T_N^3=(Z/NZ)^3`, define:

- `U_k(i)` — voxel scalar, initialized from a deterministic seeded SplitMix64 sequence or accepted explicit user-supplied voxels (`N<=16`) in `[-1,1]`.
- `q_k=(qx,qy,qz)` — Lorenz oscillator state, initially `(0.1,0,0)`.
- `p_k in {0,1,2,3}` — selected path index.
- `T_k` — trace of measured residuals, energy, mean, path cost and invariant errors.

The full state is `S_k=(U_k,q_k,p_k,T_k)`. The interpreter executes `S_{k+1}=Phi_theta(S_k)` with the following exact phases.

## 2. Lorenz flow and spatial forcing

The Lorenz vector field is `f(q)=(10(qy-qx), qx(28-qz)-qy, qx*qy-(8/3)qz)`.

`q_{k+1}=RK4(q_k, f, h=0.005)` (classical fourth-order explicit Runge–Kutta).

With decay `rho in [0,1]` and scale `chi in [0,0.2]`, define:

`a_k=chi * rho^k * tanh(q_{k+1,x}/20)`.

`psi(x,y,z)=sin(2*pi*(x+0.5)/N) * cos(2*pi*(y+0.5)/N) * sin(2*pi*(z+0.5)/N)`.

`V_k(i)=tanh(U_k(i)+a_k*psi(i))`.

The spatial forcing is genuinely coordinate-dependent and bounded. `rho=1` is permitted, and does **not** imply that the external forcing vanishes.

## 3. Scalar attention and gated recurrence

With normalized exponential attention weights:

`a_i=exp(V_k(i)) / sum_j exp(V_k(j))`, `c=sum_i a_i V_k(i)`.

`g_i=sigmoid(V_k(i)+c)`, `sigmoid(v)=1/(1+exp(-v))`.

`W_k(i)=(1-g_i)*V_k(i)+g_i*tanh(0.75*V_k(i)+0.25*c)`.

This is an explicit deterministic nonlinear map, not a learned statistical prediction model.

## 4. Toroidal 3D folding and its invariant

Six-neighbour periodic Laplacian:

`Delta_3 W(i)=W(x-1,y,z)+W(x+1,y,z)+W(x,y-1,z)+W(x,y+1,z)+W(x,y,z-1)+W(x,y,z+1)-6*W(i)`.

Let `m=(1/N^3)*sum_i W(i)` and choose `alpha>=0`, `beta>=0`, `6*alpha+beta<=1`.

`U_{k+1}(i)=W(i)+alpha*Delta_3 W(i)-beta*(W(i)-m)`.

Equivalently, `U_{k+1}(i)=(1-6*alpha-beta)*W(i) + alpha*sum_neighbors W(j) + beta*m`.

**Boundedness proof**: `V in [-1,1]` because of tanh; `c` is a convex combination of `V`; the gated recurrence forms a convex combination of `V` and another tanh result; and the final fold is a convex combination of bounded `W` values and their mean, since all coefficients are non-negative and sum to one. By induction `|U_k(i)|<=1` in exact arithmetic for all finite steps. The runtime checks tolerance `1e-10`.

**Mean preservation of the folding stage**: periodic neighbours are a permutation of the lattice. Thus `sum_i Delta_3 W(i)=0`, and `sum_i (W(i)-m)=0`. Therefore `mean(U_{k+1})=mean(W)` in exact arithmetic. The program checks the absolute numerical error at each step (`<=1e-10`). This does NOT imply that the preceding nonlinear attention/forcing stages preserve mean.

The inequality `6*alpha+beta<=1` is a **sufficient bound for the folding operation**; it does not prove total-system contraction, Lorenz boundedness for all time, or asymptotic convergence of the composite nonlinear map.

## 5. Discrete path search

For directions `d0=(1,0,0), d1=(0,1,0), d2=(0,0,1), d3=(1,1,1)`, center `c0=(floor(N/2),floor(N/2),floor(N/2))`, planning horizon `H in [1,32]`:

`J(d,U)=(1/H) * sum_{t=0}^{H-1} [U(c0+t*d mod N)^2 + 0.01*abs(U(c0+t*d mod N))]`.

`p_{k+1}=argmin_{d in {0,1,2,3}} J(d,U_{k+1})`, lowest index wins a tie.

This is a measured geometric quadratic-cost search. It is NOT a neural policy trained with reinforcement learning, nor is it an arbitrary decision-making agent.

## 6. Verification, stopping and evidence

`r_k = U_{k+1}-U_k`. `RMS_k=sqrt(sum_i r_k(i)^2 / N^3)`.

`Linf_k=max_i |r_k(i)|`; `E_k=sum_i U_{k+1}(i)^2 / N^3`.

At every step: finite fields and metrics, voxel range, and mean preservation of the fold are verified. Every trace row includes RMS, Linf, energy, mean, max absolute voxel value, context, forcing, path, path cost, fold mean error. `stationary_observed` is set only if `RMS_k <= epsilon` **and** `|a_k| <= epsilon` with epsilon strictly positive. This is an *empirical termination test only*. The program explicitly returns `asymptotic_convergence_proven=false`.

`fingerprint64` is a **non-cryptographic** FNV-1a digest of the final state quantized to 1e-4; collisions are possible. The `audit.py` and `batch.py` SHA-256 values hash serialized evidence and are not signed attestations or proof of tamper resistance.

### Space/time cost

Let `M=N^3`, iterations `K`, planning horizon `H` (4 candidate paths). State arrays store ~`3*M*8` bytes plus trace. Each iteration computes O(M+H) arithmetic; the sin/cos forcing and exp attention raise operation costs but not asymptotic order. The output with full field is limited to `N<=16`; larger runs export the central XY slice, finite telemetry and deterministic fingerprint.

### End-to-end chain

`validated client JSON -> native CLI options + optional stdin voxels -> ingest -> RK4 Lorenz -> coordinate-dependent forcing -> gated nonlinear update -> stable 3D fold -> deterministic path search -> invariant verification -> JSON voxel field/trace -> native/HTTP unit tests -> isolated GitHub Actions CPU runner -> optional authenticated Cloud Run service or Cloud Run Job`.

## Scientific sources

- Lorenz equations: https://mathworld.wolfram.com/LorenzEquations.html
- Google Cloud Run container runtime contract: https://docs.cloud.google.com/run/docs/container-contract
- Cloud Run IAM authentication: https://docs.cloud.google.com/run/docs/authenticating/overview
- Cloud Run jobs: https://docs.cloud.google.com/run/docs/overview/what-is-cloud-run

The boundedness and fold-mean derivations above follow directly from the specified coefficients and toroidal topology; they are claims about this implementation, not experimental evidence of a self-improving AI system.
