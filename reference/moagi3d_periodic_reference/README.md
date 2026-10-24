# Moagi3D periodic numerical reference (public, non-authoritative)

This is a bounded, dependency-light **numerical** example of 3D periodic diffusion, coupled memory and residual accumulation. It is independent of the private DM-PCS DOCX runtime, archive formats, model weights, IP dossier and operational authority. No such private artifacts are committed here.

## Model

On a periodic cubic lattice, let `L` be the six-neighbour discrete Laplacian and `R` the order-four permutation `(x,y,z) -> (z,y,-x mod side)`. For `alpha=.99`, `gamma=.05`, `D=.10`:

```
q        = v + D * L(v)
residual = alpha * (m - q)
v_next   = R(q + gamma * residual)
m_next   = R(alpha * m + (1-alpha)*q)
e_next   = R(e + abs(residual))
```

The conserved scalar is `(1-alpha)*sum(v) + gamma*alpha*sum(m)`. The asymptotic **v,m** state is uniform with value `c = ((1-alpha)*mean(v0)+gamma*alpha*mean(m0))/((1-alpha)+gamma*alpha)`. `finite_value_memory_fft(...)` produces a finite-step **v,m** prediction using Fourier block exponentiation and a final `R**(steps % 4)`; it does not calculate the driven `e` state.

The spatial mean of `e` is monotonic, while its spatial arrangement can have a period-four asymptotic orbit. This is not physical energy, neural training, encryption, compression, exact byte reconstruction or an automatically executing Word document.

## Reproduce

```bash
python -m pip install 'numpy>=1.24,<2.3'
cd reference/moagi3d_periodic_reference
python -m unittest -q test_engine.py
```

The public reference enforces a power-of-two cubic side from 2 to 64, finite `float64` data, and 0–1,000,000 bounded analytic steps. Inputs are never evaluated as source code.

## Integration boundary

This is a standalone research example, **not imported by the production `jarvisx` package**. It does not alter any VM instruction set, transactional authority boundary, DOCX parser, production service, financial functionality, or deployment workflow. Promote only after independent numerical, security, licensing and integration review. Never execute code embedded in untrusted documents.
