# Jarvis-X 30D Auto-Encoding/Decoding Engine

## Runtime identity

The 30D engine is a deterministic cognitive subsystem embedded beneath the
Jarvis-X VM control plane. It does not allocate a dense 30-dimensional tensor.
It materialises only coordinates selected by deterministic content routing.

The closed cycle is:

```text
Observation
  -> deterministic sparse routing
  -> latent encoding
  -> next-state prediction
  -> prediction residual
  -> persistent omega-memory update
  -> latent correction
  -> decoding
  -> coherence projection
  -> transactional commit or rollback
```

## State

Each active coordinate contains:

```text
Cell30D = (latent, prediction, residual, omega, decoded, visits)
```

A coordinate is a tuple of exactly 30 bounded integers. The global physical
state is proportional to the active-cell count rather than the theoretical
manifold volume.

## Determinism contract

Given the same configuration, prior committed manifold state, and input, the
engine emits the same:

- active coordinates;
- latent transitions;
- decoded output;
- residuals;
- omega-memory update;
- coherence score.

Python's process-randomised `hash()` is not used. Routing uses BLAKE2b over a
canonical numeric representation.

## VM integration

`CodexVM.step()` retains its existing authority chain:

```text
decode -> LambdaShield -> execute -> ledger -> trace -> reflex -> sandbox
```

After reflex stabilisation, the committed opcode and register snapshot are
passed to `ThirtyDAutoEncodingEngine.observe_vm_state()`. The 30D subsystem
observes the transition but cannot directly mutate the VM register file.

An explicit application-level cycle is available as:

```python
result = vm.cognitive_cycle(observation)
```

## Sparse memory boundary

`Engine30DConfig.max_active_cells` is a hard upper bound. When the bound is
reached, the least-visited coordinate is evicted deterministically, with the
coordinate tuple acting as the tie-breaker.

## Lambda-style coherence projection

A candidate state is rejected when it contains non-finite values, exceeds the
configured magnitude boundary, or violates the active-cell budget. Failed
candidates restore their prior touched-cell state.

## Operational capabilities

The initial implementation provides:

1. text, byte, and numeric observation normalisation;
2. deterministic 30-axis sparse addressing;
3. bounded virtual manifold storage;
4. latent encoding and decoding;
5. predictive residual calculation;
6. persistent omega-memory;
7. recurrent latent correction;
8. per-instruction VM observation;
9. deterministic execution traces;
10. transactional candidate validation.

It intentionally excludes autonomous external actions, unrestricted code
rewriting, network access, and direct mutation of authoritative VM state.
Those capabilities must remain behind explicit policy, sandbox, provenance,
and review boundaries.

## Master transition

For each active coordinate `q`:

```text
z_t        = encode(x_t, omega_t)
p_t        = predict(z_(t-1), omega_t)
e_t        = x_t - p_t
omega_(t+1)= retain(omega_t) + learn(e_t) + retain_latent(z_t)
z_t+       = correct(z_t, e_t, omega_(t+1))
y_t        = decode(z_t+, omega_(t+1))
```

The candidate is committed only after coherence validation.
