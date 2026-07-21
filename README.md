# Jarvis-X

Jarvis-X is a deterministic, auditable virtual machine with a reflex control
layer, policy gate, and an operational sparse 3-D auto-encoding/decoding
simulation automaton.

## Operational 3-D automaton

The automaton exposes a virtual address universe of

\[
(1000^{1000})\times(1000^{1000})\times(1000^{1000})=10^{9000}
\]

cells without allocating a dense tensor. Exact arbitrary-precision coordinates
identify the universe; untouched cells are reconstructed procedurally, while
only the bounded causal frontier is physically materialised.

Each transaction performs:

```text
activate
→ encode the local 3-D neighbourhood
→ evolve the latent ANN state
→ decode the proposed field
→ calculate residual error
→ update Ω correction memory
→ apply diffusion and local automaton dynamics
→ project through Λ validity constraints
→ commit or roll back atomically
→ append the deterministic journal hash
```

A bounded mechanics optimiser can shadow-test declared parameter candidates and
adopt one only when it is valid and improves the measured objective. It does
not perform unrestricted source rewriting.

## Install

```bash
git clone https://github.com/Lord-Xido/Jarvis-X.git
cd Jarvis-X
pip install -r requirements.txt
pip install -e .
```

## Run the sparse universe

```bash
jarvisx universe
jarvisx automaton --steps 20 --side 3
jarvisx automaton --steps 20 --side 3 --auto-optimize --json
```

Module execution is equivalent:

```bash
python -m jarvisx automaton --steps 20
```

## Run the bytecode VM

```bash
jarvisx run program.jx
```

## Run the API

```bash
jarvisx api
```

The service exposes:

- `GET /health`
- `POST /run`
- `GET /automaton`
- `POST /automaton/step`

Example automaton transaction:

```bash
curl -X POST http://localhost:8080/automaton/step \
  -H 'content-type: application/json' \
  -d '{"injections":[{"x":0,"y":0,"z":0,"value":1.0}]}'
```

## Test

```bash
pytest
```

The reference tests verify deterministic replay, procedural reconstruction,
sparse frontier budgeting, autoencoder dimensions, transactional commits,
journal chaining, and rollback on failed validity checks.

See [`docs/SPARSE_3D_AUTOMATON.md`](docs/SPARSE_3D_AUTOMATON.md) for the
operational mathematics and execution model.
