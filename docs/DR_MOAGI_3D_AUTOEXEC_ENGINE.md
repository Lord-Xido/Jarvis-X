# Dr Moagi 3D Auto-Encoding / Decoding Auto-Execution Engine

## Status

This is a bounded reference integration track for Jarvis-X. It composes existing sparse 3D field execution with a deterministic sparse parser, a block autoencoder, verify-before-commit execution, hash-chained audit records, and a finite self-optimization loop.

It does **not** claim unrestricted self-modification, arbitrary code rewriting, general intelligence, or lossless compression of arbitrary data.

## End-to-end kinetic loop

```text
raw sparse 3D field
      |
      v
SparseParser3D
validate -> clamp -> prune -> budget
      |
      v
SparseBlockCodec3D
encode active coordinates into quantized block latents
      |
      v
DrMoagiFieldRuntime
snapshot -> decode requested support -> residual -> field law
      |
      v
candidate state
      |
      v
validator / Pi_Lambda
MSE + finite telemetry + active-cell budget
      |
      +---- reject ----> rollback
      |
      v
commit
      |
      v
bounded policy search
      |
      v
promote only if measured objective improves
      |
      v
permeate policy coherently across parser + codec + runtime
      |
      v
SHA-256 journal -> next cycle
```

The executable state transition remains the existing field law:

```text
dPsi/dt =
    -alpha (I - D o E)[Psi]
    + lambda * Laplacian((I - D o E)[Psi])
    + eta * G_moagi * Psi
```

The new integration layer supplies concrete `E` and `D` operators and makes their execution contract explicit.

## Sparse representation

The engine never allocates `side ** 3` resident cells. A field is represented as:

```python
{(x, y, z): value, ...}
```

Only active coordinates are materialized. The parser validates coordinates and values before they enter the runtime and enforces `max_active_cells`.

## 3D autoencoder

`SparseBlockCodec3D` partitions the logical lattice into cubic blocks of edge length `block_size`.

For block `b`, its latent value is the quantized mean of active values in that block:

```text
z_b = Q( (1 / n_b) * sum(x_i for i in b) )
```

The latent payload stores:

- block coordinate;
- quantized latent value;
- active population;
- block energy.

Decoding only materializes coordinates explicitly requested in the runtime support closure. It never expands the complete logical volume.

## Auto-execution and verification

Each cycle executes transactionally:

1. freeze the current sparse state;
2. encode it;
3. decode only requested support;
4. compute reconstruction residual;
5. evaluate closure, residual diffusion and Moagi glyph terms;
6. project candidate values into configured bounds;
7. reject if the reconstruction error, numeric checks or active-cell budget fail;
8. otherwise commit atomically;
9. journal the cycle;
10. optionally evaluate bounded policy candidates for the next cycle.

A rejected candidate leaves the authoritative sparse state unchanged.

## Bounded self-optimization

The adjustable policy is deliberately small:

```text
pi = (block_size, quantization, prune_epsilon)
```

The optimizer searches only the immediate finite neighbourhood of the current policy. Candidate scoring is deterministic:

```text
J = w_f * fidelity
  + w_c * compression_gain
  + w_e * execution_saving
```

with:

```text
fidelity         = 1 / (1 + reconstruction_mse)
compression_gain = 1 - latent_cells / active_cells
execution_saving = 1 - retained_cells / active_cells
```

A candidate is promoted only when:

```text
J(candidate) > J(current) + min_policy_improvement
```

This is a bounded parameter search, not arbitrary runtime self-rewriting.

## Permeation

Within this implementation, **permeation** has a precise operational meaning: a verified policy promotion is applied consistently to every layer that depends on the policy.

```text
verified policy
   |----> parser prune threshold
   |----> codec block/quantization policy
   `----> runtime projection prune threshold
```

This prevents parser, latent representation and execution projection from drifting onto incompatible policies.

## Audit chain

Every cycle emits a canonical JSON record into a SHA-256 hash chain:

```text
H_t = SHA256(prev_hash || canonical_json(record_t))
```

An optional JSONL journal can be persisted and reloaded. The journal verifies the complete chain before continuing.

This provides integrity evidence and deterministic traceability. It is not encryption and does not by itself provide secrecy.

## CLI

Install the package in development mode:

```bash
python -m pip install -e ".[dev]"
```

Run the built-in sparse 3D demonstration:

```bash
jarvisx-dr-moagi --cycles 8 --pretty
```

Persist the audit chain:

```bash
jarvisx-dr-moagi \
  --cycles 8 \
  --block-size 2 \
  --quantization 0.01 \
  --journal state/dr-moagi-autoexec.jsonl \
  --pretty
```

Disable bounded self-optimization:

```bash
jarvisx-dr-moagi --no-auto-optimize --pretty
```

### Input format

```json
{
  "field": [
    {"x": 10, "y": 10, "z": 10, "value": 1.0},
    {"x": 11, "y": 10, "z": 10, "value": 0.75}
  ]
}
```

Run it with:

```bash
jarvisx-dr-moagi --input field.json --side 64 --cycles 4 --pretty
```

## Python API

```python
from jarvisx.dr_moagi_autoexec import DrMoagiAutoExecutionEngine

engine = DrMoagiAutoExecutionEngine()
engine.load({
    (32, 32, 32): 1.0,
    (33, 32, 32): 0.75,
})

reports = engine.run(4)
assert engine.journal.verify()
print(engine.status())
```

## Operational boundaries

The current reference implementation prioritizes correctness, bounded resource use and auditability over learned reconstruction quality or throughput.

Production-scale follow-on work should benchmark alternative codecs, add accelerator-specific kernels behind the same `FieldCodec` boundary, and preserve the same support, validation and transactional invariants.
