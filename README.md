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
