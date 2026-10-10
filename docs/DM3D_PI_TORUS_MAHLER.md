# DM3D–π: verified three-torus Mahler numerical research layer

## Provenance and epistemic boundary

This module implements **established** mathematical identities as bounded research software. It is not a new theorem, does not supersede established Mahler-measure generalizations, and should not be used to claim physically executing 10^72 operations. The π-periodic parameterization is exactly equivalent to the standard unit-torus measure.

This is an **independent numerical adapter** adjacent to the existing 3D encoding/decoding, toroidal swarm and virtual-address research engines. It is **not** an ANN trained across media and does not change any hardware/control-plane authority.

## Coordinates and exact integer addressing

Let S = 10^24, Lambda = (Z/SZ)^3 and |Lambda| = 10^72 *virtual* addresses. Each integer coordinate k in [0,S) induces u=πk/S and phase exp(2iu). Distinguishing neighbouring indices near S requires arbitrary precision, so Python uses mpmath for exact-integer-to-high-precision-angle evaluation. Ordinary float64 is insufficient to resolve 10^-24 phase differences.

## Mahler measure and Jensen reduction

For a three-variable Laurent polynomial P and positive radii r=(r_x,r_y,r_z), define

m_r(P) = π^-3 ∫_[0,π)^3 log |P(r_x e^(2iu), r_y e^(2iv), r_z e^(2iw))| du dv dw.

If **P=A(x,y)+z B(x,y)** (only z exponents 0 and 1), Jensen's formula gives

m_r(P) = π^-2 ∫_[0,π)^2 log max( |A(r_x e^(2iu),r_y e^(2iv))|,
                                    r_z |B(r_x e^(2iu),r_y e^(2iv))| ) du dv.

Reason: for fixed (u,v), Jensen's formula gives the circular average of
log |a+b exp(iw)| = log max(|a|,|b|). This is an **exact analytic elimination**
of the third angular variable. Subsequent 2D *numerical quadrature remains approximate*. The module rejects polynomials with z exponents other than 0 and 1.

## Smyth benchmark

For P = 1+x+y+z, classical evaluation m(P) = 7 ζ(3)/(2π²) ≈ 0.4262783988175058.

At N=32 midpoint resolution:
- Direct 3D midpoint: N³ = **32,768** source-polynomial evaluations; estimate **0.4274239963474219**, error ≈ 0.0011456.
- Jensen 2D midpoint: N² = **1,024** reduced-integrand evaluations; estimate **0.4262780321368381**, error ≈ 0.000000367.
- Evaluation-count ratio = **32×** at this matched grid resolution. These two evaluations have different per-point costs; this is **not** an asserted 32× CPU speedup.
- Adaptive resolution compares consecutive quadrature grids; this is a convergence heuristic, not a rigorous generic error certificate.

For a+bX+cY+dZ on radii r, when |a| > |b|r_x+|c|r_y+|d|r_z, the Mahler measure is log|a|. The dominant test uses a=1.85, b=1, c=0.5, d=0.2 and r_i=1.

## Fourier control variate (distinct experiment)

After Jensen reduction set f(u,v)=log max(|A|,|r_z B|). Fit a finite-order Fourier surrogate g_theta(u,v) on *pilot* samples, choose the order on distinct *validation* samples, and evaluate the correction on *independent* Monte Carlo residual samples:

E[f] = θ_0 + E[f-g_theta];  estimate = θ_0 + mean_{held-out}(f-g_theta).

All nonconstant Fourier modes integrate to zero over the π-periodic torus. The identity is exact, and the stochastic estimator is unbiased conditional on independently fitted coefficients and integrability. A finite-sample estimated standard error is not a certified deterministic integration bound.

Example seed 42, 384 pilot + 128 validation + 2048 held-out = 2560 source evaluations:
- Adaptive Fourier residual SE ~0.00060217.
- Plain independent Monte Carlo SE ~0.00740007 at the same 2560 source evaluations.
- Ratio of **estimated sampling variances** ~151.0 for this example.
- Fourier basis construction/least-squares fitting cost time and memory. **No 151× wall-clock improvement is claimed.**
- This fitting process selects spectral coefficients and order; it is **not a trained 3D ANN**.

## Reproduction and CI

From repository root:

```bash
python -m pip install -r apps/dm3d-pi-mahler/requirements.txt
cd apps/dm3d-pi-mahler
python -m unittest -v test_pi_mahler.py test_fourier_feedback.py
python pi_mahler.py demo --n 32 --out demo_audit.json
python fourier_feedback.py --case smyth --seed 42 --out fourier_audit.json
```

Open `apps/dm3d-pi-mahler/pi_torus_monitor.html` in a modern browser (offline). The graph renders a two-angle torus projection, with the third angle conceptual, rather than purporting to embed T³ into ordinary R³.

The dedicated GitHub Actions workflow reruns numerical regression tests and validates auditable reports on Python 3.10/3.12/3.13. Interpret pre-existing reports as historical run artifacts, not automatically refreshed benchmark measurements.
