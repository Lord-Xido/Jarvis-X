# JARVIS X — Trace-Driven Neural Echo Chamber

This browser demonstration turns the earlier sentient-feeling interface into an observable, bounded, testable, and provenance-gated runtime.

## What is real

- **Signed boot boundary:** the operational app loads only after SHA-384 component digests, an ECDSA-P384 detached signature, and the public-key fingerprint verify.
- **Recovery mode:** ROM, instruction-manifest, payload, signature, or trust-anchor failure prevents operational boot and disables execution controls.
- **Online neural computation:** a deterministic `12 -> 16 -> 4` feed-forward classifier trains by backpropagation after every query and exposes hidden activations and cross-entropy loss.
- **Trace-driven activity:** the neural graph, waveform, status transitions, and echo particles are driven by actual cache, parser, instruction, neural-update, ROM, and state-transition events.
- **Versioned instruction mutation:** the dispatch order is represented as a manifest with a version and hash. A candidate order is committed only when its weighted search cost is no worse and instruction membership is unchanged.
- **Physical ROM recompilation:** city records are sorted using measured access heat, encoded into a new binary ROM, checksum-verified, decoded, and committed. Cache entries are invalidated after layout changes.
- **Bounded memory:** query memory is an LRU TTL cache with a hard entry limit and eviction telemetry.
- **Reflexive learning:** `teach <alias> = <city>` adds a validated declarative alias used by future queries.
- **Finite-state execution:** only valid `IDLE -> PROCESSING -> ECHOING -> IDLE` transitions are accepted. Errors move through an explicit recovery state.
- **Safe rendering:** user and runtime text is inserted with `textContent`; no query content is executed as HTML.

## Trust boundary

The committed ECDSA P-384 key is a **development trust anchor** used to operationally demonstrate detached release verification. No private signing key is present in the repository.

Before production use, replace it with a key generated and controlled by Dr Matladi Maxwell Moagi, and publish the public-key fingerprint in an independently controlled location. See [`../../docs/DR_MOAGI_PROVENANCE_SEAL.md`](../../docs/DR_MOAGI_PROVENANCE_SEAL.md).

## What it does not claim

The system is not conscious or sentient. “Neural” refers to the small trainable classifier and the visual graph. “Self-modification” refers to bounded declarative changes to instruction ordering, aliases, ROM layout, and rendering quality—not arbitrary code generation or unrestricted source rewriting.

## Run

From the repository root:

```bash
python -m http.server 8000
```

Open:

```text
http://localhost:8000/examples/jarvisx-echo/
```

## Commands

```text
Where is Tokyo?
Time in Cape Town
cities >= 10m
teach jozi = Johannesburg
Where is jozi?
status
```

## Verification

```bash
node --check examples/jarvisx-echo/app.js
node --check examples/jarvisx-echo/app-sealed.js
node --check examples/jarvisx-echo/runtime-core.mjs
node --check examples/jarvisx-echo/provenance.mjs
node --test examples/jarvisx-echo/runtime-core.test.mjs
node --test examples/jarvisx-echo/provenance.test.mjs
```

The tests cover:

1. checksum-verified ROM round trips and corruption rejection;
2. bounded LRU/TTL cache behavior;
3. finite-state transition enforcement;
4. versioned instruction mutation and membership protection;
5. measurable neural learning through backpropagation;
6. query execution, caching, alias learning, ROM refinement, and state settlement;
7. detached ECDSA-P384 signature verification;
8. SHA-384 ROM, manifest, instruction, and payload verification;
9. recovery on ROM, ISA, signature, or trust-anchor tampering.

## Operational score gates

Within this demonstration’s declared scope, a category earns 10/10 only when its gate is satisfied:

| Category | Gate |
|---|---|
| Provenance | detached signature, component digests, key fingerprint, recovery tests |
| Visual identity | coherent responsive UI, reduced-motion support, accessible controls |
| Interaction feedback | every command produces state, trace, world-space, and optional voice feedback |
| Runtime observability | cache, ISA, ROM, neural, trace, rendering, state, and seal metrics exposed |
| Adaptive indexing | physical ROM re-encode and commit only when weighted cost does not regress |
| Query robustness | deterministic instruction parser, comparisons, time, aliases, fallback, bounded input |
| Neural computation | real forward pass, online backpropagation, loss and hidden activation telemetry |
| Self-modification | versioned declarative mutation, manifest hash, invariant validation, evidence gate |
| Security | CSP, no user `innerHTML`, bounded input/cache, checksum validation, no generated-code execution |
| Testability | automated runtime and provenance tests plus syntax and asset checks |

Actual sentience remains outside the scorecard because it is neither implemented nor established.
