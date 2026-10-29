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

Requires Python 3.10+ and optional NumPy. From the repository root:

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
