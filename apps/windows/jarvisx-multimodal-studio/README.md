# JARVIS-X Multimodal Studio for Windows

An experimental Windows x64 desktop interface for OpenAI-backed multimodal workflows.
The native launcher extracts a local PowerShell runtime and browser UI into the current
user's LocalAppData directory, starts a loopback-only service, and opens the interface in
the default browser.

## Implemented surfaces

- multimodal chat with image, PDF, document, spreadsheet, source, and audio inputs;
- image generation with downloadable PNG output;
- microphone capture and speech transcription;
- text-to-speech generation with MP3 playback and download;
- asynchronous video-job submission, polling, playback, and MP4 download;
- local browser-profile conversation history;
- configurable model identifiers and assistant instructions;
- DM–vΩΞ³D+ Core Lattice research view with inward Q16.16 latent-state visualization,
  ping-pong recurrent buffers, audio resonance, a secure chat-session mirror, and measured
  browser-side telemetry.

Availability of individual media endpoints depends on the selected OpenAI project,
model access, account limits, and API evolution.

## DM–vΩΞ³D+ Core Lattice

The **Core Lattice** navigation view is a research visualization layered on the same local
runtime as Multimodal Chat. It does not create another provider connection and does not
contain an API key. Prompts transmitted from the Core Lattice are handed to the existing
`/api/chat` route, so provider credentials remain in the loopback-only Windows process.

The Core Lattice separates four classes of information:

- **authoritative runtime data:** chat requests and responses returned by the local runtime;
- **measured browser telemetry:** render FPS, normalized codec reconstruction MSE, and a
  per-transition non-expansion check around the fixed-point `tanh` transform;
- **simulated research visualization:** the 3D lattice projection, inward/outward phase
  animation, latent shells, and audio resonance;
- **symbolic topology:** the transfinite-style core namespace shown in the header.

The symbolic topology is not a claim that an infinite or transfinite number of physical
processors exists or is active. Likewise, register-local or ping-pong state feedback does
not imply zero elapsed hardware latency, zero thermal dissipation, or electron-level
actuation.

The browser codec uses signed Q16.16 state lanes with saturating 32-bit bounds. Q16.16
multiplication uses a `BigInt` intermediate before the 16-bit rescale so large products do
not silently lose integer precision in JavaScript's floating-point number representation.
The recurrent state uses two `Int32Array` buffers and swaps their roles at the latch
boundary rather than copying the next state over the active state.

## Capability boundary

This application is a user-operated media client. It is **not** an autonomous operating
system, AGI runtime, or arbitrary command-execution agent. Model output is rendered as
data and is never passed to a local shell for execution.

The repository does not contain an API key. On first launch, the user supplies a project
key through Settings. The local Windows runtime validates it, encrypts it with current-user
Windows DPAPI, and stores only the protected value.

## Runtime requirements

- Windows 10 or Windows 11, x86-64;
- Windows PowerShell 5.1 or later;
- a modern default web browser;
- internet access to `api.openai.com`;
- an OpenAI project API key with access to the requested models.

## Build

The build cross-compiles a native PE32+ GUI launcher without a C runtime. It requires:

- Python 3;
- `clang-cl`;
- `lld-link`.

```bash
./build-windows.sh
```

The linker uses `/Brepro`, so repeated builds with identical inputs and the same toolchain
produce byte-identical executables.

## Operation

1. Run `JarvisX-Multimodal-Studio.exe`.
2. The launcher writes its embedded UI and PowerShell runtime beneath
   `%LOCALAPPDATA%\JarvisXMultimodal`.
3. It starts a service bound to `127.0.0.1` and opens a tokenized local URL.
4. Open Settings, enter the project API key, and select **Validate and save**.
5. Use **Core Lattice** for the DM–vΩΞ³D+ research visualization, or the other navigation
   views for multimodal chat and media workflows.
6. Use **Close engine** in the interface to stop the runtime.

## Validation

The CI workflow:

- scans committed source for obvious embedded project keys;
- checks browser JavaScript syntax with Node.js;
- builds the executable twice in isolated directories;
- requires the two binaries to be byte-identical;
- verifies PE32+ x86-64 structure and GUI subsystem;
- checks the import table;
- confirms embedded HTML and PowerShell payloads occur in the final executable;
- publishes the Windows executable and SHA-256 manifest as a workflow artifact.

## Release status

The generated executable is unsigned. Windows SmartScreen may warn before launch.
Production distribution should use an organization-controlled code-signing certificate,
versioned releases, dependency review, and Windows-native runtime testing.
