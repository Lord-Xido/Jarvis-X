# HGA Proof-Gated Geometric VM

A runnable reference emulator for the **Hyper-Geometric Assembly (HGA)** model as a self-referential, proof-gated geometric virtual machine.

The bytecode is the serialized control description. The actual machine is the closed state operator

\[
S_{k+1}=\mathcal G_{\Gamma_k}\left[S_k,(\mathcal K_{\rm MDL}\circ\mathcal B_G\circ\mathcal U_{B_k}\circ\mathcal F_{\rm in}\circ\mathcal P_G)(S_k)\right].
\]

This implementation is deliberately bounded and executable. It does **not** claim general program equivalence, absolute theorem proving, literal time collapse, or physical ontological equivalence; those are represented as constrained, testable operators.

## Implemented

- 4-byte HGA bytecode decoder/interpreter.
- Runtime state `S=(pc,B,Gamma,G,Psi,Z,V,Omega,Theta,Pi,M)`.
- Minimal `Cl(1,3)` Clifford geometric product and rotor action.
- Semi-implicit damped latent kinetic evolution.
- Inward encode/decode/error-memory correction loop.
- Bounded parameter learning with gradient clipping.
- Trivalent proof gate: `PROVED`, `UNKNOWN`, `DISPROVED`.
- Transactional commit/preserve/rollback semantics.
- Proof-carrying self-optimization for a concrete safe transform: NOP elimination.
- MDL-style executable score.
- Dependency-graph block scheduler.
- TIME_BLOCK multiple-slice Jacobi relaxation.
- Deterministic telemetry and SHA-256 certificates.
- Unit tests covering mechanics and failure modes.

## Install

```bash
cd apps/hga-vm
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Run end-to-end

```bash
python hga_vm.py run --cycles 8
```

Machine-readable telemetry:

```bash
python hga_vm.py run --cycles 8 --json
```

Inspect bytecode and block schedule:

```bash
python hga_vm.py inspect
```

Exercise TIME_BLOCK directly:

```bash
python hga_vm.py time-block
```

Run the verification suite:

```bash
python -m unittest -v test_hga_vm.py
```

## Default bytecode

Each instruction is four bytes `[opcode, a, b, c]`.

| Opcode | Instruction | Operational meaning |
|---:|---|---|
| `0x00` | `NOP` | No state change |
| `0x01` | `AXIOM_LOAD` | Extend `Gamma` with a registered proof namespace |
| `0x10` | `GEOM_INIT` | Initialize `Cl(1,3)` metric |
| `0x20` | `PERMEATE` | Advance latent kinetic state |
| `0x30` | `FOLD_INWARD` | Encode, reconstruct, correct, learn |
| `0x40` | `VERIFY` | Evaluate admissibility certificate |
| `0x50` | `SELF_EXEC` | Treat bytecode as bounded data and inspect/hash it |
| `0x60` | `MDL_SCORE` | Compute bounded description/resource score |
| `0x70` | `TIME_BLOCK` | Solve multiple latent temporal slices by relaxation |
| `0x80` | `OPTIMIZE` | Propose + prove + commit/rollback bytecode transform |
| `0xF0` | `HALT` | End one bytecode pass |

## Mechanical cycle

One `closed_cycle()` executes:

1. clone the committed state;
2. interpret the HGA bytecode;
3. load logical namespaces and geometry;
4. evolve `Z,V`;
5. fold `Psi -> Z -> Psi_hat -> e -> Omega -> Psi_corrected`;
6. update `Theta`;
7. verify the candidate;
8. self-inspect bytecode;
9. compute MDL score;
10. solve a time block;
11. propose a bounded bytecode optimization;
12. post-verify;
13. commit on `PROVED`, preserve on `UNKNOWN`, rollback on `DISPROVED`.

The proof gate checks bytecode well-formedness, finite numerical state, reconstruction bounds, energy/resource bounds, a local contraction proxy, and required logical namespaces. `UNKNOWN` is intentionally distinct from `DISPROVED`.

## Self-modification boundary

The emulator does not attempt undecidable general semantic equivalence. Instead `verify_refinement()` proves a narrow executable contract for the current optimizer:

\[
\operatorname{signature}(B')=\operatorname{signature}(B),\quad |B'|\le |B|,\quad B'\text{ well formed}.
\]

The current transform removes only NOP instructions. Any future transform must add its own refinement certificate before commit.

## Geometry and kinetics

The Clifford basis obeys

\[
e_\mu e_\nu+e_\nu e_\mu=2g_{\mu\nu},\qquad g=\operatorname{diag}(1,-1,-1,-1).
\]

The latent kinetic update is the executable semi-implicit form

\[
V_{k+1}=\frac{V_k-\Delta\tau M^{-1}\nabla U+\Delta\tau M^{-1}F}{1+\Delta\tau M^{-1}\Gamma_d},\qquad Z_{k+1}=\operatorname{Exp}_{Z_k}(\Delta\tau V_{k+1}).
\]

The reference emulator currently uses an identity latent manifold, so `Exp_Z(delta)=Z+delta`. The interface is isolated so a curved manifold implementation can replace it.

## Proof semantics

- `PROVED`: all executable invariants are certified.
- `UNKNOWN`: no hard invariant is falsified, but at least one is not certified.
- `DISPROVED`: at least one hard invariant is falsified.

The closed machine therefore implements commit/preserve/rollback rather than treating proof failure as falsity.

## Scope

This app is a mechanistic reference implementation of the HGA equations. Formal proof assistants, nonlinear manifolds, GPU kernels, learned multimodal encoders, and external-world observation operators can be integrated behind the same state and gate interfaces without changing the transactional machine contract.
