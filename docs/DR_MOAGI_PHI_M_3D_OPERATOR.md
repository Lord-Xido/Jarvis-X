# Phi_M: bounded three-dimensional geometric operator

**Status:** opt-in, candidate-only mathematical reference module. No production, firmware, banking, medical, robotics, or vehicle connections are changed.

## Original equation and explicit modeling assumptions

The source image contains

\[
\Phi_M(\mathbf{x},t)
=\sigma\left(\alpha_M\cos(\omega_M\|\mathbf{x}\|)\mathbf{x}
+\int_0^t\nabla z(\tau)\,d\tau\right).
\]

An executable vector map requires resolving the unspecified scalar field and activation. This reference chooses componentwise tanh and a moving Gaussian potential

\[
z(\mathbf{x},t)=A(1+\beta\sin(\nu t))
\exp\left[-\|\mathbf{x}-(\mathbf c_0+\mathbf vt)\|^2/(2s^2)\right].
\]

The term inside the integral is defined as the spatial gradient at the same fixed spatial coordinate x throughout the time integral:

\[
\nabla_{\mathbf{x}}z = -z(\mathbf{x},t)(\mathbf{x}-\mathbf c(t))/s^2.
\]

The Hessian used for the spatial Jacobian is

\[
H_z=z\left[(\mathbf{x}-\mathbf c)(\mathbf{x}-\mathbf c)^T/s^4-I/s^2\right].
\]

Gauss-Legendre quadrature approximates the time integrals. The analytic Jacobian includes the radial derivative, the integrated Hessian and the componentwise tanh derivative. The temporal derivative equals

\[
\partial_t\Phi_M=(1-\Phi_M^2)\odot\nabla_{\mathbf{x}}z(\mathbf{x},t)
\]

up to the temporal quadrature approximation.

These choices do not establish invertibility, an autoencoder, lossless compression, a trained ANN, global contraction, or a speedup claim.

## Run

Requires Python 3.10+ and optional NumPy; SciPy enables accelerated analytical integration. From the repository root:

    pip install -e '.[graphics]'
    python -m jarvisx.dr_moagi_phi_m_3d --self-test
    python -m jarvisx.dr_moagi_phi_m_3d --time 1.5
    python -m pytest -q -o addopts='' tests/test_dr_moagi_phi_m_3d.py

The independent numerical test compares the 3x3 Jacobian against coordinate-wise finite differences, the time derivative against time-wise finite differences, and a stationary Gaussian case against its closed-form integral. It checks origin continuity, t=0, tensor shapes, input validation, resource budgets, topology preservation, feature preservation, and rejection without commit.

## Integration with existing Jarvis-X research graph runtime

Import PhiMConfig, phi_m, spatial_jacobian, temporal_derivative and graph_candidate from jarvisx.dr_moagi_phi_m_3d. The existing geometric_diffusion_runtime.GeometricGraph remains the source of truth.

The optional graph_candidate function:
1. Maps existing 3D node positions through Phi_M, without modifying their features or edges.
2. Uses the existing project_candidate projection to limit displacement relative to the input graph.
3. Calls the existing graph_rms metric and fails closed if the configured RMS limit is exceeded.
4. Returns an independent candidate graph; no runtime state is committed or overwritten.

Independent validation and explicit authority boundaries are still required before promotion. The module is not automatically invoked from any VM, scheduler or graph runtime.

## Bounds and verification

Point count is capped at 4096 and the Gauss-Legendre order at 64; times and parameters must be finite and within declared ranges. The dense quadrature temporary work is O(NQ), including a (Q,N,3,3) Hessian. Invalid inputs fail before iteration. The reference is deterministic for fixed inputs but is not a general security sandbox.

Local numerical verification supports mathematical consistency of this selected model, not validation against an external physical system.


## Moving-source analytic integration (experimental acceleration)

The former evaluator used 32-point Gauss–Legendre quadrature for the moving Gaussian. The analytic path now evaluates three temporal moments in O(N) per point batch rather than O(NQ), using the complex scaled complementary error function available in SciPy.

Let d = x-c0, v be the source velocity and s its spatial sigma:

\[
a=\frac{\|v\|^2}{2s^2},\quad b=\frac{d\cdot v}{s^2},
\quad c=-\frac{\|d\|^2}{2s^2}.
\]

Define complex Gaussian temporal moments:

\[
I_k(\lambda)=\int_0^t\tau^k\exp(-a\tau^2+\lambda\tau)\,d\tau.
\]

For a>0, the stable base integral uses erfcx:

\[
I_0(\lambda)=\frac{\sqrt\pi}{2\sqrt a}
[\operatorname{erfcx}(-\lambda/(2\sqrt a))
-e^{-at^2+\lambda t}
\operatorname{erfcx}(\sqrt a t-\lambda/(2\sqrt a))].
\]

Recurrence gives

\[
I_1=\frac{\lambda I_0+1-e^{-at^2+\lambda t}}{2a},
\quad
I_2=\frac{I_0+\lambda I_1-t e^{-at^2+\lambda t}}{2a}.
\]

For k in {0,1,2}, define

\[
M_k=e^c[\Re I_k(b)+\beta\Im I_k(b+i\nu)].
\]

Then accumulated gradient and Hessian are

\[
G=-\frac{A}{s^2}(dM_0-vM_1),
\]

\[
H=A\left[
\frac{dd^TM_0-(dv^T+vd^T)M_1+vv^TM_2}{s^4}
-\frac{I M_0}{s^2}
\right].
\]

These equations implement the sinusoidally modulated, translating Gaussian source, *not* an arbitrary scalar field.

The exact zero-velocity case uses the elementary time integral of 1+beta*sin(nu*t). When v is extremely small relative to s/t, or moment evaluation becomes nonfinite, the implementation intentionally falls back to quadrature to avoid cancellation. Use method="analytic" or method="quadrature" explicitly for reproducibility. In method="auto", SciPy is used if installed; otherwise the earlier numerical evaluator remains available.

Install:

    pip install -e '.[phi_m]'
    python -m jarvisx.dr_moagi_phi_m_3d --self-test
    python -m pytest -q -o addopts='' tests/test_dr_moagi_phi_m_3d.py

## Performance evidence and nonclaims

Local Python 3.13.5 / NumPy 2.3.5 benchmark of 48-point quadrature vs analytic integration (median of 15 warm runs, time 1.5) measured roughly 8.3x at N=64, 9.3x at N=512, 10.8x at N=2048, and 9.6x at N=4096. Absolute differences: gradients ~4e-15 and Hessians ~6e-14 in these cases.

This is **not** evidence of 1000x speedup, global acceleration across all parameters, speed on GPU/C++, or a claim of universally exact floating-point output. These numbers apply only to this local configuration. Model verification is distinct from external physical validation.
