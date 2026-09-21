# NEXUS-3D

NEXUS-3D is the multimodal GUI/media projection surface for Jarvis-X.

It provides:

- Three.js 3D core and hologram gallery;
- text + image chat requests;
- optional provider search grounding;
- provider-backed image generation;
- provider-backed TTS;
- browser speech-to-text when supported;
- measured browser FPS;
- measured Web Audio spectrum visualization.

## Authority boundary

NEXUS-3D is not an authoritative Jarvis-X state engine.

```text
browser interaction
  -> bounded same-origin request
  -> media/provider adapter
  -> response / visualization
```

Future controls that mutate Jarvis-X runtime state must use ADR-026:

```text
immutable snapshot
  -> epoch-targeted proposal
  -> bounds/capability checks
  -> CTR / Pi_Lambda
  -> COMMIT or ROLLBACK
```

## Provider credentials

Do **not** put a provider API key in `index.html`.

Configure the server process instead:

```bash
export GEMINI_API_KEY="..."
jarvisx-nexus3d
```

`GOOGLE_GEMINI_API_KEY` is also accepted.

The browser only calls same-origin Jarvis-X endpoints:

```text
/api/nexus3d/chat
/api/nexus3d/image
/api/nexus3d/tts
```

## Model configuration

The model identifiers supplied with the original workspace are defaults, not
hard-coded assumptions. Override them without editing browser code:

```bash
export JARVISX_NEXUS3D_TEXT_MODEL="gemini-3-flash-preview"
export JARVISX_NEXUS3D_IMAGE_MODEL="gemini-3.1-flash-image"
export JARVISX_NEXUS3D_TTS_MODEL="gemini-2.5-flash-preview-tts"
```

Provider base URL and request limits are likewise configurable:

```bash
export JARVISX_NEXUS3D_PROVIDER_BASE_URL="https://generativelanguage.googleapis.com/v1beta"
export JARVISX_NEXUS3D_PROVIDER_TIMEOUT_SECONDS="45"
export JARVISX_NEXUS3D_MAX_PROVIDER_RESPONSE_BYTES="25165824"
export JARVISX_NEXUS3D_MAX_IMAGE_BYTES="8388608"
```

## Run from the repository

```bash
python -m pip install -e .
jarvisx-nexus3d --host 127.0.0.1 --port 8787
```

Then open `http://127.0.0.1:8787/`.

For a custom static bundle:

```bash
jarvisx-nexus3d --static-dir /path/to/nexus-static
```

## Security boundary

The backend enforces:

- server-side credentials;
- prompt and upload bounds;
- upstream timeout;
- upstream response-size bound;
- HTTP(S)-only grounding source URLs;
- same-origin browser provider calls;
- CSP, referrer and content-type security headers.

The service does not expose arbitrary URL fetching, arbitrary shell execution,
source rewriting, or direct authoritative runtime mutation.

## Telemetry boundary

The 3D scene is a visualization. Core rotation, particles, glow and hologram
orbit are not compute-performance evidence.

The FPS counter measures the browser render loop.

While generated speech is playing, spectrum bars are driven by Web Audio
analyser data. They are no longer random simulated frequency values.

## Cognitive geometry relationship

ADR-028 connects NEXUS-3D to the Dr Moagi cognitive-field geometry work as a
projection surface only:

```text
candidate cognitive state
  -> computational geometry receipt
  -> CTR / Pi_Lambda
  -> authoritative state
  -> immutable GUI/media projection
  -> NEXUS-3D
```

NEXUS-3D does not turn computational latent geometry into a physical
general-relativity claim.
