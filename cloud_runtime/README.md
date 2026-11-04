# JX3D Native Cloud Execution Gate

This directory holds the **exact C++17 sparse 3D reversible ANN bytecode core** from the locally verified JARVIS-X multimodal v2 package (the core's CLI identifies itself as v1).

## What the cloud workflow executes

GitHub-hosted Ubuntu runners compile the native C++ source and execute a 64-tile bounded run:

1. Generate 64 active 16×16×16 3D byte tiles (262,144 voxel bytes).
2. Train the fixed-point causal predictor on synthetic sample data.
3. Encode the sparse tile stream with the neural predictor and bytecode.
4. Decode and require byte-for-byte reconstruction with identical SHA-256.
5. Deliberately corrupt an encoded byte and require a nonzero decoder exit code.
6. Print runner architecture and stored payload sizes for feedback.

The native CLI reads and writes files. Coordinates address up to 10^12 positions per axis but physical storage is sparse. The workflow is a **remote CI proof of execution**, not an always-on cloud service, production object-store integration, or a demonstration of processing a 10^36-byte dataset.

## Deployment boundary

The Python v2 multimedia generator, wrapper optimizers, FastAPI application, and generated PNG/GIF/MP4/WAV assets are **not yet synchronized to this branch**. They remain locally tested with a separate 12-test suite. This cloud gate proves only the exact native C++ streaming codec and checks, not production multimedia cloud operation.

Replit app creation was denied by the connected account service (active subscription required). Production HTTP routing would require a provisioned container/VM service, registry, authenticated job queue, persistent object storage, rate limits, TLS and monitoring. No cloud endpoint or GPU service is claimed here.

## Native commands

The CI job uses these five commands: `generate`, `train`, `encode`, `decode`, `verify`. It exits nonzero when the decoder accepts a corrupted stream.
