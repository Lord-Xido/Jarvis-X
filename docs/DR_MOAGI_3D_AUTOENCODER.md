# Dr Moagi 3‑D Auto‑Encoder / Decoder Core

This document formalises the bijective 3‑D latent mapping used by the Jarvis‑X runtime.  It contains the exact analytical encoder \(\mathcal E\), decoder \(\mathcal D\) and the latent‑space evolution operator \(\mathfrak P\).

```math
\boxed{\begin{aligned}
\text{Encoder } \mathcal{E} &: \mathbb{R}^9 \;\longrightarrow\; \mathbb{R}^3\\[4pt]
\begin{bmatrix} X \\ Y \\ Z \end{bmatrix} &=
\begin{bmatrix}
\displaystyle \frac{8}{1+L/10}-2\\[6pt]
\displaystyle 4-6\,\Omega\\[6pt]
\displaystyle 6-2(\alpha+\beta+\gamma)
\end{bmatrix}\\[15pt]
\text{Decoder } \mathcal{D} &: \mathbb{R}^3 \;\longrightarrow\; \mathbb{R}^9\\[4pt]
\hat{\mathbf s} &=
\begin{bmatrix}
\displaystyle 50-0.4X-3Y-0.2Z\\[4pt]
\displaystyle \frac{8}{X+2}-10\\[4pt]
\displaystyle \frac{4-Y}{6}\\[4pt]
\displaystyle 50+10\sin\!\bigl(\tfrac{X+Y}{2}\bigr)\\[4pt]
\displaystyle 1.0+0.3\,(1-\hat\Omega)\\[4pt]
\displaystyle \frac{6-Z}{4}-0.5\,(\beta+\gamma)\\[4pt]
\displaystyle 0.2+0.1\,(1-\hat\Omega)\\[4pt]
\displaystyle 0.3+0.2\,(1-\hat\Omega)\\[4pt]
\displaystyle 0.5+0.3\cos\!\bigl(\tfrac{X+Y}{2}\bigr)
\end{bmatrix}\\[15pt]
\text{Evolution } \mathfrak P &: \qquad \mathbb P_{t+1}=\mathbb P_t+\eta\,\nabla_{\mathbf z}F\bigl(\mathcal D(\mathbf z)\bigr)\,\mathbb I_{\|\mathbf s-\hat{\mathbf s}\|<\varepsilon}
\end{aligned}}
```

*All variables follow the ranges and semantics defined in previous design notes.  The fidelity gate \(\varepsilon\) is currently set to 0.05 in production.*
