# DM-vΩΞ⁺ Permeation Transport Core

Operational packaging of the uploaded Meta AI React artifact **DM-vΩΞ⁺ PERMEATION TRANSPORT CORE — TOPO. TOKAMAK CONTROL // VER 2.4.1**.

## What is deployed

- `index.html` — a clean deployable UI reconstructed from the uploaded artifact’s implemented recurrence and controls, wired to the readable fixed-step kernel.
- `core.mjs` — readable, deterministic kernel for the particle transport recurrence found in the uploaded artifact.
- `test_core.mjs` — invariants covering particle-count mapping, seeded reproducibility, finite/bounded state, refresh-rate-independent fixed-step execution, and permeation disablement.
- `.nojekyll` — keeps GitHub Pages serving the static app as-is.

## Model boundary

This is an interactive **stochastic cylindrical transport analogue**. It is not an MHD, PIC, Maxwell, Grad–Shafranov, collision, or experimentally calibrated tokamak solver. The production contract is therefore simulation/runtime correctness and reproducibility of the implemented recurrence, not plasma-physics validation.

## Runtime state

Each particle carries:

```text
(r, theta, z, vr, vtheta, vz, shell, q)
```

with shell boundaries `0.33, 0.66, 1.0` and equilibrium radii `0.165, 0.495, 0.83`.

Default control parameters are:

```text
omega=1.8  kr=0.85  Ares=0.62  c=0.12  mu=0.38  D=0.28  rho=1.0
```

Transport regimes:

```text
laminar:   alpha=0.15 gamma=0.22 beta=1.7
turbulent: alpha=1.35 gamma=0.85 beta=3.2
quantum:   alpha=0.02 gamma=0.03 beta=0.6
```

## Verification

```bash
node --test apps/permeation-transport/test_core.mjs
python3 -m http.server 8080 --directory apps/permeation-transport
```

Then open `http://localhost:8080`.

## Production rule

UI labels are descriptive. `core.mjs` and the executable artifact recurrence define the software behavior. Claims of physical validity require separate empirical validation.
