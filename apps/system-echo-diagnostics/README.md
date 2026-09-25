# Jarvis X System Echo Diagnostics

A state-reactive Three.js/WebGL HUD for the Jarvis-X research platform.

## Implemented browser capabilities

- ANN UI state machine: `IDLE -> ENCODING -> PROCESSING -> DECODING`;
- state-driven 3D rotation, reverse decoding rotation, pulse amplitude, emissive color and bloom strength;
- Three.js icosahedral wire shell, solid physical-material core and procedural particle field;
- Unreal Bloom post-processing through `EffectComposer`;
- real microphone acquisition after explicit browser permission, analyzed through Web Audio FFT;
- command input and multimodal decoder surface;
- execution-stream logging and program-counter visualization;
- local browser clock;
- synthetic CPU/GPU load meters whose distribution changes with ANN UI state.

## State-reactive geometry

The holographic core is a visualization of the current UI state:

| State | Geometry response |
|---|---|
| `IDLE` | cyan, baseline rotation, low-amplitude pulse |
| `ENCODING` | light blue, 4x rotation, stronger pulse |
| `PROCESSING` | red, 10x rotation, high pulse and bloom |
| `DECODING` | purple, reversed 3x rotation, medium pulse |

The particle field counter-rotates relative to the core.

## Microphone path

Press **INITIATE SENSOR LINK** to request `getUserMedia({ audio: true })`. Once permission is granted, the resonance bars display frequency-domain data from a Web Audio `AnalyserNode`.

Before permission is granted, the bars are synthetic animation. The UI identifies the sensor state explicitly.

## External multimodal generation

No API credential is committed to this public repository.

The page reads an optional runtime value from:

```js
window.JARVIS_X_GEMINI_API_KEY
```

and otherwise leaves external text/image/TTS generation disabled.

A browser-visible API key is not a secret. For any public or production deployment, route external model calls through a server-side Jarvis-X endpoint or another credential-holding proxy rather than embedding a provider key in this page.

The model identifiers and provider endpoints in the browser client are experimental integration points and should be validated against the provider API before production use.

## Operational boundary

The following are real browser operations when available:

- Three.js/WebGL rendering;
- post-processing/bloom;
- geometry transforms and material mutation;
- microphone capture after explicit permission;
- Web Audio FFT;
- DOM command handling;
- HTTP requests when a runtime API credential has been supplied.

The following remain synthetic diagnostic telemetry:

- CPU percentage;
- GPU percentage;
- the displayed program-counter address sequence;
- execution-stream module/address labels;
- ANN state itself unless connected to an authoritative backend runtime.

The UI must not be interpreted as measuring native Jarvis-X VM execution, host CPU/GPU utilization, or deployed ANN internals merely because the visualization changes state.

## Run

Open `index.html` in a modern browser with network access to the Tailwind, Google Fonts, Three.js, and Three.js post-processing CDNs. Microphone capture requires a secure browser context such as HTTPS or localhost.
