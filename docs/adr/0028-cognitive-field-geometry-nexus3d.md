# ADR-028: Computational cognitive field geometry and NEXUS-3D projection

- **Status:** Proposed
- **Date:** 2026-09-21
- **Applies to:** Dr Moagi cognitive-engine latent geometry, CTR evidence, fixed-point verification, NEXUS-3D, and future geometry-aware backends
- **Extends:** ADR-015, ADR-016, ADR-017, ADR-026

## Context

Jarvis-X already treats cognition as a bounded state-transition system with
explicit latent state, residuals, temporal memory, candidate-first adaptation,
CTR evidence, Pi_Lambda admissibility and commit/rollback.

A useful systems analogy exists between that architecture and constrained
geometric field systems: state-dependent geometry, local transport,
path-dependent inconsistency, source/geometry coupling, constraint monitoring
and verified evolution.

That analogy becomes useful only when converted into software-defined quantities
with an explicit evidence boundary. Reusing physical general-relativity symbols
without redefining them would create an unacceptable ambiguity between
computational abstractions and physical tensors.

The supplied NEXUS-3D interface also needs a trustworthy integration boundary.
A browser UI must not expose provider credentials and must not become an
alternate authoritative runtime path.

## Decision

Jarvis-X adds two cooperating, non-authoritative layers.

### A. Computational cognitive field geometry

The reference implementation is:

- `docs/DR_MOAGI_COGNITIVE_FIELD_GEOMETRY.md`
- `src/jarvisx/cognitive_field_geometry.py`
- `tests/test_cognitive_field_geometry.py`

It defines:

1. a symmetric positive-definite three-axis latent metric;
2. a symmetric computational curvature proxy;
3. a symmetric informational source tensor;
4. a computational field residual;
5. a loop-transport holonomy residual;
6. reconstruction and fixed-point evidence;
7. a non-authoritative geometry receipt.

The reference field residual is

[
mathcal R^{(Z)}
=
mathcal G^{(Z)}
+
Lambda_Z g^{(Z)}
-
kappa_Zmathcal T^{(mathrm{info})}.
]

Every symbol in this equation is software-defined.

### B. NEXUS-3D GUI/media projection

The reference surface is:

- `apps/nexus-3d/index.html`
- `src/jarvisx/nexus3d_api.py`
- `tests/test_nexus3d_api.py`

The browser performs visualization, interaction, image attachment, provider
request initiation, media playback, audio-spectrum measurement and 3D gallery
projection.

Provider credentials remain server-side. The browser calls only same-origin
endpoints:

```text
/api/nexus3d/chat
/api/nexus3d/image
/api/nexus3d/tts
```

The backend may call the configured provider, but it does not mutate
authoritative Jarvis-X runtime state.

## Geometry authority boundary

The geometry verifier does not commit state.

Its output is evidence that may be supplied to CTR / Pi_Lambda:

```text
candidate latent state
  -> geometry/source projection
  -> field residual
  -> transport holonomy residual
  -> geometry receipt
  -> CTR
  -> Pi_Lambda
  -> COMMIT or ROLLBACK
```

A geometry receipt therefore has the same authority class as another bounded
verification receipt: informative but non-authoritative until the enclosing
transaction commits.

## Verification law

For configured thresholds, the local geometry gate accepts only when

[
|mathcal R^{(Z)}|_Fleepsilon_G,
]

[
h_Zleepsilon_H,
]

[
e_{m recon}leepsilon_R,
]

and

[
delta_{m fp}leepsilon_F.
]

The combined action proxy

[
A_Z=
sqrt{
|mathcal R^{(Z)}|_F^2
+h_Z^2
+e_{m recon}^2
+delta_{m fp}^2
}
]

is telemetry only. Acceptance is based on the individual declared limits so one
small term cannot conceal another out-of-bounds term.

## Structural relation to general relativity

General relativity supplies structural inspiration only.

The following identities are explicitly rejected:

```text
latent metric == spacetime metric
informational stress == physical stress-energy
computational curvature == Riemann/Einstein curvature
kappa_Z == 8*pi*G/c^4
Lambda_Z == cosmological constant
```

A small computational residual does not establish a physical law, consciousness,
truth, external correctness or general intelligence.

## NEXUS-3D provider boundary

The GUI SHALL NOT contain provider API keys.

Provider calls SHALL:

- originate from the same-origin Jarvis-X backend;
- enforce request-size bounds;
- enforce provider-response-size bounds;
- use explicit timeouts;
- return sanitized source URLs;
- surface provider failures rather than silently fabricating output;
- keep model identifiers configurable on the server.

The browser MAY use local Web Speech Recognition if the browser supports it.
This is a browser capability, not a Jarvis-X authoritative input channel.

## Measured versus visual telemetry

NEXUS-3D SHALL distinguish actual measurements from visual effects.

The audio ring is driven from Web Audio analyser bins while audio is playing.
Particle motion, core rotation, glow, hologram orbit and other scene effects are
visualization state and are not runtime-performance evidence.

The footer FPS value is browser render-loop telemetry only.

## GUI command law

ADR-026 remains authoritative for operational GUI commands.

NEXUS-3D is currently a media/observation adapter. If future controls propose
runtime mutations, they must use:

```text
immutable snapshot
  -> epoch-targeted bounded command
  -> capability/bounds checks
  -> candidate
  -> CTR / Pi_Lambda
  -> COMMIT or ROLLBACK
```

Direct browser mutation of authoritative runtime memory is prohibited.

## Consequences

### Positive

- turns the GR/cognitive analogy into testable computational diagnostics;
- introduces a concrete path-consistency/holonomy signal;
- preserves the candidate-first transaction boundary;
- gives NEXUS-3D a secure same-origin provider architecture;
- prevents browser exposure of provider credentials;
- replaces simulated random audio bars with measured spectrum data;
- keeps physical and cognitive claims evidence-gated.

### Costs

- the geometry layer adds another verification receipt and configuration surface;
- the current reference is intentionally three-axis and is not a general
  differential-geometry engine;
- provider-backed NEXUS features require server configuration and network access;
- the browser visualization remains dependent on external CDN assets unless
  those assets are vendored later.

## Validation

Conformance tests SHALL cover:

- metric symmetry and positive definiteness;
- computational field residual construction;
- zero and nonzero loop-holonomy cases;
- acceptance/rejection under independent geometry thresholds;
- claim status fixed to `computational_geometry_only`;
- provider key non-exposure;
- same-origin browser API routes;
- image/TTS inline-media extraction;
- grounding source filtering;
- invalid base64 rejection;
- absence of direct provider URLs/API-key literals from the browser bundle.

## Provenance

This ADR belongs to the Dr Moagi family and inherits the canonical attribution
and provenance policy defined by ADR-015 and
`docs/attribution/PERMEATION_MANIFEST.md`.
