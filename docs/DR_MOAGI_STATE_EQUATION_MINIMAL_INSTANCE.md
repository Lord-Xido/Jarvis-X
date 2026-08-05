# Dr. Moagi State Equation — Minimal Geometric Instance

**Status:** Pedagogical / demonstrative specialisation of the 3D autoencoding geometry.

This document records the exact low-dimensional instance that exhibits the core geometric behaviour of the engine: anisotropic inward contraction, time-variant oscillatory reconstruction, exact residual cancellation, particle homotopy, spherical inversion, and multimodal sonification.

It does not replace the high-dimensional sparse or billion-field specifications. It is the concrete “heartbeat” demonstration.

---

## 1. State Variables & Input Geometry

Let the input matrix be

$$
\mathbf{P} = [P_1 \mid P_2] \in \mathbb{R}^{3 \times 2},
\qquad
P_1 = (1,2,3)^\top,\;
P_2 = (4,5,6)^\top.
$$

Define the centroid (mean) and deviation (span):

$$
\mu = \frac{P_1 + P_2}{2} \in \mathbb{R}^3,
\qquad
\Delta = \frac{P_1 - P_2}{2} \in \mathbb{R}^3.
$$

Explicitly:

$$
\mu = (2.5,\; 3.5,\; 4.5)^\top,
\qquad
\Delta = (-1.5,\; -1.5,\; -1.5)^\top.
$$

---

## 2. The Encoding Operator (Inward Contraction)

The latent core is a non-uniform affine compression:

$$
\boxed{Z(t) = \mathbf{S} \cdot \mu}
$$

where

$$
\mathbf{S} = \operatorname{diag}(0.9,\; 0.8,\; 0.7)
$$

is the fixed compression matrix.

Geometrically this maps the input parallelogram inward toward the origin with anisotropic scaling along each Cartesian axis.

$$
Z = (2.25,\; 2.8,\; 3.15)^\top.
$$

---

## 3. The Decoding Operator (Outward Expansion with Oscillation)

Reconstruction is a time-variant affine expansion:

$$
\boxed{
P'_1(t) = \alpha(t)\cdot Z(t) + \Delta,
\qquad
P'_2(t) = \alpha(t)\cdot Z(t) - \Delta
}
$$

with dynamical drift scalar

$$
\alpha(t) = 1 + \varepsilon\sin(\omega_\alpha t),
\qquad
\varepsilon = 0.05,\;
\omega_\alpha = 0.3\;\text{rad/s}.
$$

This injects a persistent oscillatory drift, preventing exact convergence and producing a quasi-periodic training loop.

---

## 4. Reconstruction Loss (Geometric Error Field)

The instantaneous loss is the sum of Euclidean residuals:

$$
\boxed{\mathcal{L}(t) = \|P_1 - P'_1(t)\|_2 + \|P_2 - P'_2(t)\|_2}.
$$

Because the deviations cancel,

$$
P_1 - P'_1(t) = \mu - \alpha(t)Z,
\qquad
P_2 - P'_2(t) = \mu - \alpha(t)Z,
$$

the loss collapses exactly to

$$
\boxed{\mathcal{L}(t) = 2\,\|(I - \alpha(t)\mathbf{S})\mu\|_2}.
$$

The loss depends only on the distortion of the centroid; the span \(\Delta\) is invisible to the residual.

---

## 5. Spatiotemporal Particle Flow (Homotopy Transport)

Particles execute a continuous linear homotopy. For the \(i\)-th particle:

$$
\boxed{\gamma_i(t) = (1 - \tau_i(t))\cdot\Gamma_{\text{start}} + \tau_i(t)\cdot\Gamma_{\text{end}}}
$$

with smoothstep temporal kernel

$$
\tau_i(t) = 3s_i(t)^2 - 2s_i(t)^3,
\qquad
s_i(t) = \frac12\bigl(\sin(\omega_i t + \phi_i) + 1\bigr).
$$

This guarantees \(C^1\)-continuous, jerk-free motion.

- Encoder particles: \(\Gamma_{\text{start}}\in\{P_1,P_2\}\), \(\Gamma_{\text{end}}=Z(t)\).
- Decoder particles: the reverse path.

---

## 6. Topological Inversion (Inward Sphere)

Every point \(X\in\mathbb{R}^3\) is projected onto the inner surface of a sphere of radius \(R=5.8\):

$$
\boxed{\Phi_{\text{mirror}}(X) = R\cdot\frac{X}{\|X\|_2}
\quad\text{(clamped for \(\|X\|_2\le R\))}}.
$$

This radial normalisation converts Euclidean space into a bounded, non-Euclidean visual field that forces geometry to fold inward toward the observer.

---

## 7. Multimodal Audio Projection (Sonification)

The latent state is mapped into the auditory domain:

$$
\boxed{
\begin{aligned}
f_{\text{pitch}}(t) &= 180 + 15\cdot(Z_x + Z_y) &&\text{(Hz)}\\
G_{\text{gain}}(t) &= 0.08 + 0.5\cdot\mathcal{L}(t) &&\text{(amplitude)}\\
\Theta_{\text{pan}}(t) &= \operatorname{atan2}(Z_y,Z_x) &&\text{(azimuth)}\\
\Psi_{\text{elev}}(t) &= \operatorname{asin}\bigl(Z_z/\|Z\|_2\bigr) &&\text{(elevation)}
\end{aligned}
}
$$

Stereo panning uses HRTF interpolation on \((\Theta,\Psi)\).

---

## 8. Unified State Transition (ODE Form)

The complete continuous dynamics may be written

$$
\boxed{
\frac{d}{dt}
\begin{bmatrix}
\mu \\ \Delta \\ Z \\ \alpha \\ \mathcal{L} \\ \gamma_i
\end{bmatrix}
=
\begin{bmatrix}
0 \\
0 \\
0 \\
\varepsilon\omega_\alpha\cos(\omega_\alpha t) \\
\text{derivative of closed-form }\mathcal{L}(t) \\
\dot{\tau}_i(t)\cdot(\Gamma_{\text{end}}-\Gamma_{\text{start}})
\end{bmatrix}
}
$$

with initial conditions

$$
\mu(0)=\tfrac{P_1+P_2}{2},\;
\Delta(0)=\tfrac{P_1-P_2}{2},\;
\alpha(0)=1,\;
\mathcal{L}(0)=2\|\mu\|_2.
$$

---

## 9. Conservation Law (Autoencoding Invariant)

Despite the time-variance the linear span is conserved:

$$
\boxed{\operatorname{span}\{P'_1(t),P'_2(t)\} = \operatorname{span}\{Z(t),\Delta\}
\quad\forall t.}
$$

The decoded points always lie on a plane parallel to the original input span. The engine never invents new directions; it only scales the compressed centroid under persistent oscillation.

---

## 10. Closed-Form Loss Trajectory (“Heartbeat”)

Substituting the explicit \(\alpha(t)\) yields the exact temporal signature:

$$
\boxed{
\mathcal{L}(t) = 2\sqrt{
(1-0.9\alpha(t))^2\mu_x^2 +
(1-0.8\alpha(t))^2\mu_y^2 +
(1-0.7\alpha(t))^2\mu_z^2
}
}
$$

where

$$
\alpha(t) = 1 + 0.05\sin(0.3t).
$$

This is the deterministic, bounded, quasi-periodic “heartbeat” of the minimal Dr. Moagi engine.

---

## Capability Boundary

- This instance is low-dimensional and pedagogical.
- It demonstrates residual cancellation, anisotropic compression, non-convergent oscillation, particle transport, spherical inversion and sonification.
- It does **not** claim lossless high-dimensional compression, unique fixed-point convergence, or production-scale sparse allocation.
- All numerical claims are exact for the stated two-point geometry.

**Classification:** Reference demonstration of the geometric core.
