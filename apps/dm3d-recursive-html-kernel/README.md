# DM3D Ω³ recursive self-hosting HTML kernel

This directory contains the **exact single-file browser application** developed in the DM3D multimodal virtual-kernel prototype. Run `index.html` directly in a JavaScript-enabled browser. No CDN or application server is required for the core workload.

## Capabilities

- 30-bit-per-axis virtual 3D coordinates with reversible 90-bit Morton addressing (sparse backing, **not** 2^90 bytes of RAM).
- Adaptive exact byte codec, binary archive export/import, CRC/integrity verification.
- Generate/process text, images, WAV audio, video-frame sequences and arbitrary binary data in a browser sandbox.
- Trainable compact 64→12→64 ANN autoencoder; **approximate neural generation is separate from lossless media reconstruction**.
- KASM program compiler/interpreter, cooperative scheduling, bounded execution, virtual files and IndexedDB persistence where permitted.
- Million-cycle encode/decode/verify benchmark with actual work per recorded cycle.
- Recursive kernel self-capture, exact encode/decode/reconstruction, isolated `iframe sandbox="allow-scripts"` boot probe and bounded scheduler-genome proposals.

## Run

1. Open `index.html` locally. On browsers restricting `file:` storage, some persistence features fall back to session memory.
2. Use the kernel console to compile and run the bundled program.
3. Use **Section 11 — Recursive HTML/JS Kernel** to snapshot the current document, or import the original file for byte-exact source verification.
4. Start verified inward rounds; export a clone, archive or JSON audit.
5. To exercise the million-cycle run, select its sample program and compile/schedule it.

## Reproducible checks

From the repository root:

```bash
python -m pip install playwright==1.57.0
python -m playwright install chromium
python apps/dm3d-recursive-html-kernel/tests/test_kernel.py
```

CI additionally checks inline JavaScript syntax with Node.js. The tests use Chromium under an isolated page origin, which does not demonstrate network deployment or full IndexedDB persistence.

## Scope and security

- This is an **unprivileged, local-first browser VM emulator**, not a native OS kernel, physical GiB³ RAM, cloud supercomputer, or general autonomous coding agent.
- The self-revision search mutates only documented scheduler tuning fields and accepts candidates subject to local integrity, regression and performance gates. Faster measured scheduling is not proof of general algorithmic improvement.
- Sandbox candidate execution does not grant host filesystem or network privileges. It is not a substitute for a formal security audit of untrusted imported HTML.
- Imported media, original source and exported archives remain the user's responsibility; do not import untrusted executable HTML without reviewing it.

## Provenance

Baseline artifact SHA-256: `c103226b811467ba0a84eb0dac183d8edeeb90cca76bd0fbcc6f2490c27e8ed6` (151,091 bytes).
Exact Git blob: `4d0fb942f4a142c4513f6446e119c85c85e906aa`.
Previous standalone Chromium regression (21 November 2026) verified boot, all five modalities, exact self-copy, tamper detection, sandbox rejection and 1,000,000 codec cycles. GitHub CI is an independent check and may report separately.

Author / architecture: Dr. Matladi Maxwell Moagi. Integration: Jarvis-X DM3D research prototype.
