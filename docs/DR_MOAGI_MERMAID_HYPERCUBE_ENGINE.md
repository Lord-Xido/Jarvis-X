# The Dr Moagi 3D Auto-Encoding & Decoding Equation

**Fully operational formulation of the Mermaid Hypercube Engine**

**Status:** Specification — fixed-point latent refinement with Anderson acceleration and implicit differentiation.

This document defines the inward-spiral autoencoder that maps a bounded 64³ voxel patch into a 512-dimensional latent state, iteratively refines that state to a fixed point, and reconstructs the patch. It sits alongside the DMSO-3D PDE, the minimal geometric instance, and the billion-field references as a concrete, trainable operator.

---

## 1. Core Autoencoder Structure

Let the input be a 3D voxel patch

$$
\mathbf{X} \in \mathbb{R}^{64 \times 64 \times 64}.
$$

The encoder $\mathcal{E}$ projects it to an initial latent vector:

$$
\mathbf{z}_0 = \mathcal{E}(\mathbf{X}) \in \mathbb{R}^{512}.
$$

The decoder $\mathcal{D}$ reconstructs the patch from a converged latent state $\mathbf{z}^*$:

$$
\hat{\mathbf{X}} = \mathcal{D}(\mathbf{z}^*) \in \mathbb{R}^{64 \times 64 \times 64}.
$$

---

## 2. The Inward-Spiral Fixed-Point Dynamics (Dr Moagi Recurrence)

Instead of a single forward pass, the latent state is repeatedly refined by a 3D convolutional cell $\mathcal{F}$ that operates on a $4 \times 4 \times 4$ spatial grid (projected from the 512-D latent). The recurrence is

$$
\mathbf{z}_{t+1} = \mathcal{F}(\mathbf{z}_t) + \mathbf{z}_0,\qquad t = 0,1,2,\dots
$$

where

$$
\mathcal{F}(\mathbf{z}) = \operatorname{vec}\Big(\operatorname{Conv3D}\big(\operatorname{reshape}(\mathbf{z}, 4,4,4,C)\big)\Big) \in \mathbb{R}^{512}
$$

with a residual skip connection via the fixed $\mathbf{z}_0$.

The equilibrium state $\mathbf{z}^*$ satisfies the fixed-point equation:

$$
\boxed{\mathbf{z}^* = \mathcal{F}(\mathbf{z}^*) + \mathbf{z}_0}
$$

This is the Dr Moagi Equation for the latent embedding.

---

## 3. Anderson-Accelerated Solver

To solve for $\mathbf{z}^*$ efficiently we use Anderson acceleration with memory $m$:

$$
\mathbf{z}^{(k+1)} = \sum_{j=0}^{m} \alpha_j^{(k)} \,\mathbf{z}^{(k-m+j)},
$$

where

$$
\boldsymbol{\alpha}^{(k)} = \arg\min_{\sum \alpha_j = 1}\left\|
\sum_{j=0}^{m}\alpha_j\big(\mathcal{F}(\mathbf{z}^{(k-m+j)}) + \mathbf{z}_0 - \mathbf{z}^{(k-m+j)}\big)
\right\|_2^2.
$$

Convergence is declared when $\|\mathbf{z}^{(k+1)} - \mathbf{z}^{(k)}\|_2 < \varepsilon$.

---

## 4. Loss Function and Training Objective

The autoencoder is trained end-to-end by minimising the combined loss

$$
\mathcal{L}
=
\underbrace{\|\mathbf{X}-\hat{\mathbf{X}}\|_2^2}_{\text{Reconstruction MSE}}
+
\beta\underbrace{D_{\mathrm{KL}}\big(\mathcal{N}(\boldsymbol{\mu},\boldsymbol{\sigma}^2)\,\|\,\mathcal{N}(0,1)\big)}_{\text{KL regularisation (VAE variant)}}
+
\gamma\underbrace{\|\nabla_{\mathbf{z}}\mathcal{L}_{\mathrm{recon}}\|_2^2}_{\text{Gradient penalty for smoothness}},
$$

where $\beta$ and $\gamma$ are weighting hyperparameters.

---

## 5. The Complete Dr Moagi Operator

Combining all stages, the full forward pass is

$$
\boxed{
\hat{\mathbf{X}}
=
\mathcal{D}\!\left(
\lim_{t\to\infty}\big(\mathcal{F}(\mathbf{z}_t)+\mathcal{E}(\mathbf{X})\big)
\right)
}
$$

with the understanding that the limit is obtained via the accelerated fixed-point iteration.

---

## 6. Backward Pass (Implicit Differentiation)

To avoid unrolling the recurrence we use the implicit function theorem to compute gradients of the loss with respect to the parameters $\theta$ of $\mathcal{F}$ and $\mathbf{z}_0$:

$$
\frac{\partial\mathcal{L}}{\partial\theta}
=
\frac{\partial\mathcal{L}}{\partial\mathbf{z}^*}
\cdot
\left(\mathbf{I}-\frac{\partial\mathcal{F}}{\partial\mathbf{z}^*}\right)^{-1}
\cdot
\frac{\partial\mathcal{F}}{\partial\theta}.
$$

This is the Dr Moagi gradient backpropagation rule, enabling infinite-depth backpropagation with constant memory.

---

## 7. Summary in Compact Form

Let the whole system be denoted by the operator $\mathcal{M}_\theta$:

$$
\mathcal{M}_\theta(\mathbf{X})
=
\mathcal{D}_\theta\!\left(
\operatorname{solve}\big(\mathbf{z}=\mathcal{F}_\theta(\mathbf{z})+\mathcal{E}_\theta(\mathbf{X})\big)
\right).
$$

Then the Dr Moagi Equation for the entire auto-encoding–decoding loop is simply

$$
\boxed{
\mathbf{z}^* = \mathcal{F}_\theta(\mathbf{z}^*) + \mathcal{E}_\theta(\mathbf{X}),
\qquad
\hat{\mathbf{X}} = \mathcal{D}_\theta(\mathbf{z}^*)
}
$$

with $\theta$ updated via the implicit gradient to minimise $\mathcal{L}(\mathbf{X},\hat{\mathbf{X}})$.

---

## Capability Boundary

- Input extent is a fixed $64^3$ patch; it does not claim dense allocation of larger virtual lattices.
- The $4\times4\times4$ reshape of the 512-D latent is a design choice that must be matched by the channel count $C$ of the Conv3D cell.
- Anderson acceleration requires a stable residual history and a well-conditioned least-squares solve; failure modes must be handled by fallback fixed-point iteration or early termination.
- Implicit differentiation assumes the Jacobian $\mathbf{I}-\partial\mathcal{F}/\partial\mathbf{z}^*$ is invertible at the fixed point; a damping or regularised inverse may be required in practice.
- This specification does not establish consciousness, unrestricted self-modification, or production-scale training performance.

**Classification:** Operational fixed-point autoencoder specification for the Mermaid Hypercube Engine.
