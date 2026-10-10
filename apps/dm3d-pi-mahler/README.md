# DM3D–π | Septillion³ Virtual Three-Torus Mahler Emulation

An **offline, bounded, reproducible** experiment that maps arbitrary 80-digit-safe *integer* addresses to high-precision phases on a π-periodic 3-torus. It compares direct 3D midpoint cubature against exact Jensen dimension elimination for a useful polynomial family.

## Run

```bash
python -m pip install numpy mpmath
python pi_mahler.py demo --n 32 --tol 1e-6 --out demo_audit.json
python pi_mahler.py phase --coordinate 999999999999999999999999
python -m unittest -v test_pi_mahler.py
```

Open `pi_torus_monitor.html` in your browser. It contains all display code and works offline.

## Canonical mathematics

Let `S=10^24`, and `Λ = (Z/SZ)^3` with `|Λ|=10^72` logical positions. Map `(x,y,z)` to phases `(exp(2πix/S), exp(2πiy/S), exp(2πiz/S))`. To distinguish adjacent 24-digit indices, `phase` uses **mpmath arbitrary precision**; float64 generally cannot retain differences of order `10^-24` around the far corner.

For any Laurent polynomial whose exponents in the third variable are restricted to 0 and 1,

```
P(X,Y,Z) = A(X,Y) + Z B(X,Y)
m_{r1,r2,r3}(P)
  = (1/π²)∫_[0,π)² log(max(|A(r1 e^(2iu),r2 e^(2iv))|,
                           r3 |B(r1 e^(2iu),r2 e^(2iv))|)) du dv.
```

This is an **existing Jensen formula**, not an original extension or a replacement for Mahler's/generalized Mahler measure. It is mathematically exact even if `P` vanishes at isolated points on the original 3-torus, provided the measure is integrable.

For `P=1+X+Y+Z`, the known exact evaluation by Smyth is

```
m(P) = 7 ζ(3)/(2π²) ≈ 0.4262783988175058.
```

For `P=a+bX+cY+dZ` with `|a|>|b|r1+|c|r2+|d|r3`, the logarithmic Mahler measure is `log|a|`. For the midpoint `N^3` grid, the alias terms vanish below total degree N. Let `q=(|b|r1+|c|r2+|d|r3)/|a|<1`. Then the **worst-case** discrete-grid alias error satisfies:

```
|m_N(P) - log|a|| ≤ q^N / (N (1-q)), N ≥ 1.
```

The bound derives from the absolutely convergent series for `log(1+w)`, with no need to allocate the virtual `10^72` cells. It is an established style of Fourier/analytic aliasing argument, not a claim of new Mahler theory.

## Actual benchmark and evidence boundary

The included `demo_audit.json` was produced on this runtime for the 32-resolution `1+x+y+z` benchmark:

- Direct midpoint `32^3 = 32,768` point evaluations.
- Jensen midpoint `32^2 = 1,024` point evaluations, **32× fewer**.
- Known exact numerical target: `0.4262783988175058`.
- Both approaches have cubature error; the 2D result is much closer to the benchmark here, but **2D cannot always replace 3D**, only when the algebraic form permits it.
- Adaptive `N` refinement compares consecutive grids. Its stopping test is **heuristic**; it does not yield a general certified numerical integration error bound.
- CPU timings are machine-dependent and are not asserted to be a constant speedup across hardware.

## Connection to DM3D

The earlier learned depth router optimizes the number of spatial folding steps per finite 8³ tile. This new numerical layer is a complementary toroidal calculation: `symbolic coordinates → exact algebraic reduction → finite 2D quadrature → residual check → adaptive grid refinement`. The two engines have **not** been fused into a trained Mahler-integral network. A proposed next phase is to learn a control variate and evaluate its residual on independent samples, preserving the exact integral by correction.

**No mathematically novel result or supersession of established Mahler generalizations is claimed.**

## Learned spectral surrogate and residual correction (separate experiment)

```bash
python fourier_feedback.py --case smyth --seed 42 --out fourier_audit.json
python -m unittest -v test_fourier_feedback.py
```

Build a finite-order, data-fitted Fourier approximation

```
model_theta(u,v)=theta_0 + Σ_(i,j) [a_ij cos(2(iu+jv))+b_ij sin(2(iu+jv))]
E_pi[model_theta] = theta_0
m(P) = theta_0 + E_pi[f_P-model_theta]
```

The **training data, order-selection validation data, and final residual samples are disjoint**, so the independently corrected Monte Carlo estimate is unbiased conditional on the fitted coefficients (under standard integrability assumptions). The adaptive order-selection gate prefers a lower *held-out residual variance*, independently from training. For the seed-42 Smyth example, the residual method's **estimated standard error** was about `0.000602` vs `0.007400` for plain Monte Carlo with equal **2,560 calls to the source integrand**. That is about a 151× estimated *sampling variance* improvement in this particular experiment, **not a measured 151× wall-time speedup**; Fourier features and least-squares training cost CPU time. This is computational approximation and correction, not a new Mahler theorem.