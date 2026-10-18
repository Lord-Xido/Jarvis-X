# ADR-011: DM–vΩΞ³D+ Core Lattice interface and telemetry boundary

- **Status:** Proposed
- **Date:** 2026-08-20
- **Decision scope:** Windows Multimodal Studio research interface

## Context

Jarvis-X already separates canonical VM authority from adaptive research runtimes and browser visualization. The Windows Multimodal Studio additionally provides a loopback-only local service that owns provider credentials and exposes bounded media/chat routes to its browser UI.

The DM–vΩΞ³D+ Core Lattice adds an inward recurrent latent visualization to that client. The supplied concept combines Q16.16 latent arithmetic, recurrent state, conversational output, 3D-style geometry, audio resonance, and symbolic very-large-scale topology labels.

Without an explicit contract, those categories can be confused: a browser animation can look like physical compute, a symbolic core count can look like measured hardware, and a locally computed stability proxy can look like a proof about a deployed physical system.

## Decision

Adopt the Core Lattice as a **Layer 5/6 research interface** subordinate to the existing Windows Multimodal Studio runtime.

It does not become a second authoritative execution engine and does not bypass the canonical Jarvis-X state, policy, provenance, or capability boundaries.

The dependency direction is:

```text
user / media
    |
    v
browser Core Lattice view
    |  local latent transform + visualization only
    |
    +----> loopback /api/chat
              |
              v
       local Windows runtime
              |
              v
       provider API client
              |
              v
       model response as data
```

Provider credentials remain outside browser JavaScript. The browser communicates only with the existing authenticated loopback service.

## Telemetry classes

Every Core Lattice quantity belongs to exactly one declared class.

### Authoritative

Authoritative within this application means the local runtime actually performed the request/response transaction. It does **not** mean the model output becomes canonical VM state.

Examples:

- loopback `/api/chat` request accepted by the local runtime;
- response returned by the configured provider client;
- local runtime key-present/key-source status.

### Measured

Measured values are computed from the running browser/runtime instance.

Examples:

- browser render FPS;
- local encode/activation/latch elapsed time;
- normalized reconstruction MSE;
- latent energy before and after the bounded activation;
- current 128-lane Q16.16 vector values.

### Simulated

Simulated values drive perception or inspection but are not physical measurements.

Examples:

- 3D lattice motion;
- inward/outward particle flow;
- audio resonance derived from latent lanes;
- shell/core geometry.

### Symbolic

Symbolic values are mathematical labels or namespaces.

Examples:

```text
1000^(1000^(1000^1000))
```

Such expressions never imply an equal number of resident CPU/GPU/FPGA cores, allocated memory cells, active processes, or measured operations.

## Q16.16 latent contract

The browser research transform uses signed 32-bit Q16.16 values:

```text
Q = 2^16
q(x) = sat_i32(round(x Q))
x(q) = q / Q
```

Addition is saturating signed 32-bit addition.

Multiplication uses a widened integer intermediate before rescaling:

```text
q_mul(a,b) = sat_i32((int64(a) * int64(b)) >> 16)
```

The JavaScript reference uses `BigInt` for the multiplication intermediate so that the product is not rounded through IEEE-754 binary64 before the fixed-point shift.

## Inward recurrent state

The Core Lattice keeps two equal-size state banks:

```text
Z_A, Z_B in Q16.16^128
```

with `active` and `next` references. One local transition is:

```text
u_i       = W_i x_i + alpha z_i
z_next_i  = tanh(u_i)
(active,next) <- (next,active)
```

The latch is a ping-pong reference swap, not a claim of zero elapsed hardware time.

The implementation may describe this as **no main-memory round-trip in the modelled state transition** or **register/local-buffer recurrence**. It must not describe it as physically zero-latency computation.

## Local stability proxy

For the elementwise `tanh` activation, the interface checks the empirical transition quantity

```text
E(v) = (1/N) sum_i v_i^2
```

and reports whether

```text
E(tanh(u)) <= E(u) + epsilon.
```

This is a local software non-expansion check for that transition. It is not a proof of global Lyapunov stability for Jarvis-X, the provider model, the operating system, or hardware.

## Reconstruction telemetry

The browser codec reports normalized mean-squared reconstruction error:

```text
MSE = (1/N) sum_i ((x_i - xhat_i) / 255)^2
```

and the bounded display proxy

```text
similarity = 1 / (1 + MSE).
```

The similarity value is a UI metric, not a calibrated semantic-understanding score.

## Visualization contract

The current Core Lattice materializes a bounded deterministic set of 900 visual nodes and labels a 1,024-lane sampled visualization path where applicable.

Visual positions are initialized from a deterministic pseudo-random seed. Deterministic geometry improves reproducibility of the research surface but does not make the rendering authoritative compute state.

## Conversational integration

Core Lattice prompts are mirrored into the existing Multimodal Studio conversation and submitted through the existing `send()` path, which ultimately calls the local `/api/chat` endpoint.

The Core Lattice must not:

- embed a provider project key;
- construct a provider URL containing a key;
- bypass the loopback launch token;
- pass model output to a shell;
- directly mutate canonical Jarvis-X VM state.

Model output remains data presented to the user and optionally re-encoded into the local visualization.

## Security boundary

The existing Multimodal Studio runtime remains responsible for:

- provider credential handling;
- Windows DPAPI protection of saved credentials;
- launch-token validation;
- Content Security Policy;
- provider HTTP transport;
- bounded media endpoint handling.

The Core Lattice introduces no new network origin and no browser-side provider secret.

## Prohibited claim equivalences

The following equivalences are explicitly rejected:

```text
symbolic topology size == physical core count
browser FPS            == model inference throughput
visual inward motion   == physical energy flow
local tanh check       == certified global Lyapunov stability
state-bank swap        == zero physical latency
latent audio mapping   == measured electromagnetic resonance
model response         == canonical VM state
```

## Validation

The implementation is conformant when CI verifies:

1. the assembled browser JavaScript is syntactically valid;
2. the Core Lattice view and telemetry boundary labels are present;
3. no direct Gemini/provider-key path is introduced in the Core Lattice fragment;
4. chat continues to use the existing local `/api/chat` route and launch-token wrapper;
5. Q16 multiplication uses a widened integer intermediate;
6. recurrent state uses a dual-bank ping-pong latch;
7. deterministic visual initialization is retained;
8. symbolic/measured/simulated/authoritative labels remain explicit;
9. known unsupported physical claims are absent;
10. the Windows executable remains reproducibly buildable and contains the expected embedded assets.

## Consequences

### Positive

- the supplied cloud-core concept becomes a reproducible inspectable interface rather than an isolated HTML demo;
- symbolic and physical scale are separated;
- the latent recurrence has explicit numerical semantics;
- the browser inherits the existing credential and transport boundary;
- research telemetry can evolve without redefining canonical VM authority.

### Trade-offs

- the Core Lattice remains a visualization/research client, not a hardware benchmark;
- its simple latent codec is not a trained production autoencoder;
- its local stability metric is intentionally narrower than a system-wide proof;
- provider behavior and latency remain external to the deterministic browser transform.

## Promotion rule

Promotion beyond Layer 5/6 requires separate evidence for the capability being promoted. In particular, any future claim of authoritative execution, hardware acceleration, real-time throughput, physical field control, or global stability requires an independent implementation contract, measured fixtures, resource accounting, failure bounds, and an accepted ADR.
