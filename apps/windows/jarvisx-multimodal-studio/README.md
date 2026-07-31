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
- configurable model identifiers and assistant instructions.

Availability of individual media endpoints depends on the selected OpenAI project,
model access, account limits, and API evolution.

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
5. Use **Close engine** in the interface to stop the runtime.

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
