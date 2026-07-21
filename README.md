# Jarvis-X

Jarvis-X is a deterministic virtual machine and a transactional sparse-field
runtime with auto-encoding, residual correction, policy projection, and replayable
journaling.

## Sparse tetration field

The operational field is indexed over the symbolic universe

\[
T_1=1000,\qquad T_{k+1}=1000^{T_k},\qquad
\mathcal U_k=\{0,\ldots,T_k-1\}^3.
\]

No dense matrix of \(|\mathcal U_k|=T_k^3\) cells is allocated. Physical state is

\[
\Sigma_t=(\mathcal A_t,B_t,Z_t,\Omega_t,J_t),
\]

where \(\mathcal A_t\) contains at most `max_active_bricks` materialised brick
addresses. Each brick is

\[
B_t(\mathbf r)\in\mathbb R^{3\times4\times4\times4}
\]

and therefore contains 192 scalar values.

The default `--tower-height 2` gives an axis of \(1000^{1000}\). Higher towers
remain symbolic. Because a completely arbitrary raw coordinate at height three
or above would itself require an impractical number of bits, executable addresses
use a finite chart identifier plus exact signed local offsets.

## Operational cycle

```text
resolve symbolic address
→ hash into a collision-chained sparse directory
→ apply the active mask by directory membership
→ project all 192 brick values through W_enc
→ condition the latent state with W_omega Ω
→ softmax route to one expert (top-1)
→ apply the selected latent expert
→ reconstruct all 192 values through W_dec
→ calculate E = decoded - observed
→ update Ω' = ρΩ - ηE
→ evaluate the six-face Laplacian across brick boundaries
→ project each value into [16, 235]
→ verify budgets and finiteness
→ atomically commit or roll back
→ append the deterministic SHA-256 journal link
```

The explicit diffusion condition is enforced:

\[
\Delta tD\le\frac16,
\]

and persistent memory requires

\[
0<\rho<1.
\]

The physical cost is tied to the materialised frontier:

\[
T_{cycle}=O\!\left(M_t(192d+d^2+192\cdot6)\right),
\qquad M_{memory}=O(M_t),
\]

not to \(|\mathcal U_k|\).

## Install

```bash
git clone https://github.com/Lord-Xido/Jarvis-X.git
cd Jarvis-X
pip install -r requirements.txt
pip install -e .
```

## Run

```bash
jarvisx universe --tower-height 2
jarvisx universe --tower-height 4
jarvisx automaton --steps 20 --tower-height 2
jarvisx automaton --steps 20 --tower-height 4 --max-active 128 --json
python -m jarvisx automaton --steps 20
```

## Bytecode VM

```bash
jarvisx run program.jx
```

## API

```bash
jarvisx api
```

Endpoints:

- `GET /health`
- `GET /universe?tower_height=4`
- `POST /run`
- `GET /field` and `GET /automaton`
- `POST /field/step` and `POST /automaton/step`

Scalar brick stimulus:

```bash
curl -X POST http://localhost:8080/field/step \
  -H 'content-type: application/json' \
  -d '{"injections":[{"chart":"origin","x":0,"y":0,"z":0,"value":24.0}]}'
```

A full observation may instead provide `values` with exactly 192 numbers.

## Test

```bash
pytest
```

The regression suite covers collision chaining, full encoder and decoder
projection, softmax/top-1 routing, cross-brick diffusion, stability constraints,
sparse frontier budgets, exact replay, projection bounds, and rollback.

See [`docs/SPARSE_3D_AUTOMATON.md`](docs/SPARSE_3D_AUTOMATON.md) for the complete
operational mathematics.
