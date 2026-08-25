# DMVANN Chat — DM-vΩΞ⁺ Neural Interface

DMVANN Chat is a first-class Jarvis-X application that separates a deterministic local Ψ–Φ–Λ–Ω–Θ control field from optional generative-model inference.

The repository implementation is **not** asserted to be source-equivalent to the externally hosted Manus deployment. The external endpoint remains registered for provenance and comparison only.

## Deployments

### External reference

- URL: https://dmvannchat-utfg9t9j.manus.space/
- Observed interface name: `DM–vΩΞ⁺ Neural Interface`
- Provider: Manus
- Source equivalence with this repository: **unverified**

### Jarvis-X implementation

This directory contains an independently verifiable implementation with two execution modes:

1. **Static / GitHub Pages mode** — renders the DMVANN neural-field interface and executes deterministic local field transforms. No model credentials or generative inference are used.
2. **Node runtime mode** — serves the same UI, exposes health/runtime endpoints, and can proxy an OpenAI-compatible chat backend using server-side environment variables after an explicit operator opt-in.

The interface always distinguishes deterministic local control-plane output from remote model output.

## Operational state

For each accepted text input, the local field updates a bounded state

```text
M_t = (Ψ_t, Φ_t, Λ_t, Ω_t, Θ_t)
```

using deterministic, testable transforms implemented in `core.mjs`:

```text
Ψ_t = normalized input information magnitude
Φ_t = tanh(coupling · Ψ_t + 0.25 Θ_{t-1})
residual_t = |Φ_t - Ψ_t|
Ω_t = decay · Ω_{t-1} + (1-decay) · residual_t
Λ_t = 1 / (1 + Ω_t)
Θ_t = clamp(Θ_{t-1} + learningRate · (Φ_t - Ψ_t), -1, 1)
```

These quantities are operational control/telemetry variables. They are not presented as evidence that the browser itself is a trained neural network or that the equations reproduce the internal state of an external LLM.

## Files

```text
apps/dmvann-chat/
├── index.html       # responsive neural-field + chat UI
├── app.mjs          # browser state, rendering and runtime selection
├── core.mjs         # deterministic Ψ–Φ–Λ–Ω–Θ transforms and validation
├── server.mjs       # Node static server + server-side chat proxy
├── test_core.mjs    # invariant + endpoint tests
├── package.json     # Node 22 scripts
├── Dockerfile       # portable runtime image
├── deployment.json  # external/reference deployment metadata
└── README.md
```

## Run locally

```bash
cd apps/dmvann-chat
npm test
npm run check
npm start
```

Default address:

```text
http://localhost:8787/
```

Health endpoint:

```text
GET /healthz
```

Runtime metadata:

```text
GET /api/runtime
```

Chat endpoint:

```text
POST /api/chat
```

## Configure generative chat

The Node runtime expects an OpenAI-compatible upstream at `/v1/chat/completions`. Remote inference is disabled unless `DMVANN_ENABLE_REMOTE=1` is explicitly set.

```bash
export DMVANN_UPSTREAM_URL="https://your-trusted-model-gateway.example"
export DMVANN_UPSTREAM_API_KEY="server-side-secret"
export DMVANN_MODEL="your-model-id"
export DMVANN_ENABLE_REMOTE="1"
node server.mjs
```

`DMVANN_UPSTREAM_API_KEY` is read only by the Node process and is never emitted by `/healthz`, `/api/runtime`, or browser code. Client requests cannot override `DMVANN_MODEL`.

If no upstream is configured, `/api/chat` fails closed with `503 UPSTREAM_NOT_CONFIGURED`. If an upstream URL exists but remote inference has not been explicitly enabled, it fails closed with `503 REMOTE_MODEL_DISABLED`. In both cases the browser uses a clearly labelled deterministic local fallback.

## Container

From this directory:

```bash
docker build -t dmvann-chat .
docker run --rm -p 8787:8787 \
  -e DMVANN_UPSTREAM_URL \
  -e DMVANN_UPSTREAM_API_KEY \
  -e DMVANN_MODEL \
  -e DMVANN_ENABLE_REMOTE=1 \
  dmvann-chat
```

## Security and reliability boundaries

- request body bounded to 1 MiB;
- conversation history and per-message lengths bounded in the core;
- upstream request timeout enforced;
- unsupported message roles normalized;
- API keys remain server-side;
- upstream model selection is server-controlled;
- remote inference requires explicit operator enablement;
- upstream error bodies are not reflected to browser clients;
- same-origin browser API surface by default;
- CSP, frame denial, no-sniff, referrer and permissions headers emitted by the Node server;
- remote failure degrades to disclosed deterministic local mode instead of fabricating a model response;
- no claim of byte-for-byte identity with the Manus deployment.

A public Node deployment should still sit behind authentication/rate limiting before attaching a billable or sensitive upstream model service.

## Verification

```bash
node --test apps/dmvann-chat/test_core.mjs
node --check apps/dmvann-chat/core.mjs
node --check apps/dmvann-chat/app.mjs
node --check apps/dmvann-chat/server.mjs
```

The current suite contains seven tests covering bounded message normalization, deterministic field evolution, state bounds, fingerprints, chat request validation, disclosure of local fallback behavior, health endpoint semantics, and fail-closed remote inference behavior.

## Release criterion

The integration follows the Jarvis-X progression:

```text
working → robust → portable → elegant → advanced
```

The current GitHub implementation reaches **portable**: it is runnable locally, testable, static-hostable, containerizable, and can attach to a server-side model gateway without exposing provider credentials. Further advancement should benchmark latency, add authenticated multi-user sessions, persistence, streaming inference, model-routing policy, observability and deployment provenance before stronger production claims are made.
