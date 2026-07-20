# Jarvis-X

Jarvis-X is a deterministic, auditable virtual machine with a reflex
control layer and policy gate.

## Install

```bash
git clone https://github.com/Lord-Xido/Jarvis-X.git
cd Jarvis-X
pip install -r requirements.txt
pip install .
```

## Run the 800-instance reference swarm

```bash
jarvisx swarm 1
```

Run ten cycles without inward ROM mutation:

```bash
jarvisx swarm 10 --no-mutate
```

The reference runtime implements corrected Q8.8 arithmetic, the 128-bit SVI,
4 KiB ROM geometry, protected shadow mutation, the 800 -> 80 -> 8 hierarchy,
and cadence-based fusion. See
[`docs/JARVIS_X_800_INSTANCE_RUNTIME.md`](docs/JARVIS_X_800_INSTANCE_RUNTIME.md)
for the exact operational contract and limitations.

## Run the bounded Self-Evolving ROM

```bash
jarvisx ser 8
```

The SER runtime profiles adjacent bytecode, proposes `LDC` and `DSM` macro-op
fusions, replays baseline and candidate programs from the same deterministic
snapshot, and publishes a parent-linked ROM version only when machine-state
semantics are exactly preserved and execution cost is reduced. See
[`docs/DR_MOAGI_SELF_EVOLVING_ROM.md`](docs/DR_MOAGI_SELF_EVOLVING_ROM.md).

## Run the AEDSIE-Sigma virtual engine

```bash
jarvisx aedsie 4
```

Disable the bounded inward mechanics proposal:

```bash
jarvisx aedsie 4 --no-inward
```

The AEDSIE reference engine auto-executes deterministic synthetic RF ingestion,
DDC and channelisation, 3D tensorisation, the Dr Moagi differential operator,
residual encoding and decoding, Omega correction, positive metric evolution,
nine-expert routing, angle estimation, SHA3 provenance, and transactional
commit. See
[`docs/AEDSIE_SIGMA_VIRTUAL_ENGINE.md`](docs/AEDSIE_SIGMA_VIRTUAL_ENGINE.md)
for the mathematical contract and scope boundary.
