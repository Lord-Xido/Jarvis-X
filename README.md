# Jarvis-X

Jarvis-X is a deterministic, auditable virtual machine with a reflex control layer, policy gate, persistent Ω ledger, transactional hierarchical cognitive kernel, and a 3D geometric reasoning-visualisation runtime.

## Install

```bash
git clone https://github.com/Lord-Xido/Jarvis-X.git
cd Jarvis-X
pip install -r requirements.txt
pip install .
```

## Assembly VM

```bash
jarvisx run program.jx
```

The existing VM supports deterministic bytecode decoding, register execution, Λ policy checks, explicit reflex stabilisation, tracing, sandbox limits, and ledger recording.

## Hierarchical cognitive kernel

The operational kernel executes the canonical cycle:

```text
Encode Q3
  -> Hierarchical Condensation
  -> Predict
  -> Compare / Residual
  -> Update cumulative Ω memory
  -> Project through Λ constraints
  -> Decode
  -> Commit or Roll Back
```

Run a cycle directly:

```bash
jarvisx cognitive 3 1 -1 -3
```

Or from Python:

```python
from jarvisx.core import CodexVM

vm = CodexVM()
result = vm.cognitive_cycle([3, 1, -1, -3])

print(result.committed)
print(result.hierarchy)
print(result.metrics)
print(vm.regs.snapshot())
```

### Operational properties

- Signed three-bit latent domain: `Q3 = {-4, -3, -2, -1, 0, 1, 2, 3}`.
- Deterministic integer/fixed-ratio prediction and memory updates.
- Configurable branching factor and bounded hierarchy depth.
- Cumulative Ω correction memory with retention and learning ratios.
- Λ gates for residual, active-node, memory, and quantisation bounds.
- Atomic commit or rollback: rejected candidates never mutate committed state.
- SHA-256 state-hash chain for replay and provenance.
- Bounded in-memory journal of committed and rejected cycles.
- Existing Greek register bridge:
  - `Ξ`: encoded input aggregate
  - `Ψ`: condensed root
  - `Φ`: prediction aggregate
  - `Λ`: commit gate
  - `Ω`: cumulative memory aggregate
  - `𝒮`: residual magnitude
  - `Π`: decoded aggregate

### Measured outputs

Each cycle reports raw and condensed bit counts, hierarchy depth, condensation ratio, active-node fraction, residual magnitude, cumulative memory, reconstruction error, and provenance hashes.

## 3D geometric multiparallel feedback loop

The reasoning-visualisation shell can turn its committed public output inward:

```text
Public shell values
  -> signed-3-bit 3D voxel lattice
  -> identity / diffusion / memory / hybrid lanes in parallel
  -> 2x2x2 multiresolution condensation
  -> top-down geometric decoding
  -> reconstruction scoring
  -> Λ projection
  -> atomic commit or rollback
  -> committed output becomes the next input
```

Run four inward cycles:

```bash
jarvisx geometry3d --cycles 4 3 1 -1 -3
```

The command emits browser-safe JSON containing each cycle, the geometric hierarchy, all four lane candidates, reconstruction and memory metrics, Λ decisions, public shell events, the final state, and the projected register bank.

Python API:

```python
from jarvisx.core import CodexVM

vm = CodexVM()
cycles = vm.geometric_feedback([3, 1, -1, -3], cycles=4)

for cycle in cycles:
    print(cycle.selected_lane)
    print(cycle.output)
    print(cycle.events)
```

Open `geometric-rvis-shell.html` directly in a browser. It provides a rotatable and zoomable 3D lattice, encoded/evolved/decoded stage switching, cycle replay, lane telemetry, Λ status, and the public event timeline. Paste or load JSON produced by the CLI to replay a real execution.

The full design is documented in `docs/JX_RVIS_3D_GEOMETRIC_LOOP.md`.

## Tests

```bash
pytest
```

The suites cover signed-three-bit quantisation, hierarchical condensation, Ω learning, deterministic replay, Λ rollback, exact ISA behavior, ledger integrity, 3D address bijection, multiparallel lane determinism, inward feedback, browser event generation, and register integration.
