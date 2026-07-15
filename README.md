# Jarvis-X

Jarvis-X is a deterministic, auditable virtual machine with a reflex control layer, policy gate, persistent Ω ledger, and a transactional hierarchical cognitive kernel.

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

The existing VM supports deterministic bytecode decoding, register execution, Λ policy checks, reflex stabilisation, tracing, sandbox limits, and ledger recording.

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

Each cycle reports:

- raw and condensed bit counts;
- hierarchy depth and total nodes;
- condensation ratio;
- active-node fraction;
- prediction residual magnitude;
- cumulative memory magnitude;
- reconstruction error;
- previous, candidate, and committed state hashes.

## Tests

```bash
pytest
```

The cognitive tests cover signed three-bit quantisation, hierarchical condensation, Ω learning, deterministic replay, Λ rollback, and register integration.
