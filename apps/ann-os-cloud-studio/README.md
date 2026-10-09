# ANN OS Cloud Multimodal Studio

A runnable, cloud-deployable Node.js application that wraps the original uploaded React / Three.js **ANN OS 3D Inward Loop Emulator** as a preserved interactive component and adds a new, responsive chat-based multimodal interface.

## Capabilities and evidence boundaries

| Capability | Implementation | Credentials |
|---|---|---|
| Chat and vision | OpenAI Responses API: text plus images, PDF, text files and sampled video frames | `OPENAI_API_KEY` |
| Prompt-to-image | OpenAI Images API (`gpt-image-1.5` by default), downloadable image | `OPENAI_API_KEY` |
| Voice input | Browser microphone (MediaRecorder) → OpenAI transcription → editable prompt | `OPENAI_API_KEY` plus microphone permission |
| Voice output | OpenAI speech synthesis → downloadable MP3 | `OPENAI_API_KEY` |
| Motion video | Server-side FFmpeg generates a **real 6-second MP4** from a fractal motion background or animated image; does **not** generate model-native photorealistic video | Local FFmpeg (already in Dockerfile) |
| 3D visualization | Animated mathematical 3D toroidal projection plus preserved original React/WebGL simulator | None |
| Real telemetry | Request count, errors, last completed request latency, model output token count | None (tokens only with AI requests) |
| Session persistence | Browser localStorage plus optional server JSON sessions in `DATA_DIR` | None; use an access token if deployed publicly |

**Do not interpret the original emulator's simulated loss, virtual Cubes, CUBEFS ONLINE labels, operation modes or performance counters as evidence of a functioning trained ANN.** The new GUI separates simulated graphics and actual HTTP metrics.

**Important 2026 video limitation:** OpenAI Sora 2 and the OpenAI Video API were shut down on **2026-09-24** according to the official current OpenAI developer reference. The Motion mode produces procedural/animated MP4 content using FFmpeg, not Sora footage.

## Start locally

Prerequisites: Node.js 20+ and FFmpeg for MP4 renders. No `npm install` is required.

```bash
cd ann-os-cloud-studio
export OPENAI_API_KEY='your_server_side_key'
# Optional but strongly recommended for any internet-facing deployment:
export APP_ACCESS_TOKEN='a_long_random_private_passphrase'
node server.mjs
```

Open **http://localhost:3000**. If an `APP_ACCESS_TOKEN` is configured, use the settings icon in the GUI to provide it. This token is held in tab-scoped sessionStorage. The OpenAI provider API key is **never entered in the browser**. `OPENAI_API_KEY` missing? 3D visualization, session interface and local FFmpeg MP4 rendering still work; AI operations will clearly report that cloud inference is not configured.

## Docker / cloud

```bash
docker build -t ann-os-cloud-studio .
docker run --rm -p 3000:3000 \
  -e OPENAI_API_KEY="$OPENAI_API_KEY" \
  -e APP_ACCESS_TOKEN="$APP_ACCESS_TOKEN" \
  -v annos-data:/app/data ann-os-cloud-studio
```

Set the same environment variables in Render, Fly.io, Railway, Replit or another Node 22 service and bind the HTTP service to `PORT` (default `3000`). Do not commit `.env` or provider secrets. Use HTTPS and set `APP_ACCESS_TOKEN` before exposing endpoints to other people. The server returns the original emulator at `/legacy/ann3d.html`.

## Verification

```bash
npm test
```

The tests assert routing, validation, security boundaries, mocked OpenAI calls, and video renderer output. Tests **cannot** prove real provider access without a configured working API key. The default test suite does not spend API credits.

## Architectural evolution for multi-user production

This reference is deployable on a **single Node process**, not an auto-scaling, multi-tenant, production SaaS. For hardened multi-instance operations: replace JSON-session persistence with managed Postgres, place video work in an authenticated async queue, add an object store for long-lived binary assets, account-scoped authentication, token/cost quotas, request cancellation, structured tracing, encrypted secrets, retention/deletion controls, and infrastructure rate limiting. Configure proper access controls and content limits for user media.

The source file is preserved **without rewriting its minified React bundle**; this avoids falsely presenting its animations as a trained model. The neural math animation in the new panel is rendered locally and labeled as such. All model inferences route through the server.
