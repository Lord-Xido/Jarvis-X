# ANN OS Cloud Studio — bounded cloud integration contract

## Purpose
Provide a cloud-capable multimodal chat and media generation GUI, retaining the
user's original React/WebGL ANN OS 3D Inward Loop Emulator as an inspectable,
byte-for-byte preserved legacy view.

The implementation lives in apps/ann-os-cloud-studio and does not replace the
existing C++ ANN runtime, CodexVM, Dr Moagi IDE, or Windows Multimodal Studio.

## Operational data plane
Browser → authenticated same-origin HTTP → bounded Node.js request parsing
→ cloud model-provider API → responses → UI and session storage.

The MP4 renderer is local FFmpeg motion graphics, not learned video synthesis.
3D toroidal visualization and inward loop emulator are simulated views.
Cloud inference becomes live only after provider credentials are configured.

## Security and authority
- Cloud provider keys remain server-side.
- Production mode refuses startup without APP_ACCESS_TOKEN.
- Model responses are displayed, not executed as shell commands or VM opcodes.
- Input media sizes are bounded; this is not yet a multi-tenant SaaS.
- JSON-file sessions are suitable for a single writable server instance.
- Configure TLS, per-user auth, quotas, managed databases and object storage
  before public multi-user operation.

## Measured versus symbolic
Actual telemetry: HTTP requests, errors, cloud response latency, and provider
token usage when available. 3D field motion is an illustration of the recurrence,
not validation of local ANN training.

  X_t → E → Z_t → cloud inference → Y_t → D → Xhat_t
                ↑                     |
                +--- verify/feedback --+

E, D and verification are architectural abstractions. This does not claim
that the browser trains a local neural network or achieves AGI.

## Validation
Node 22 CI uses a local mocked cloud API so automated tests require no paid
credits. Tests cover authentication, text/vision/voice routes, session storage,
a real FFmpeg MP4 and the original emulator's exact Git blob SHA. Docker build
is a separate CI job. Mocked provider calls do not establish real provider access.

## Deployment
Use Docker with HTTPS and a secrets manager for OPENAI_API_KEY and
APP_ACCESS_TOKEN. Set DATA_DIR to a durable volume or replace JSON sessions
with a managed data store.
