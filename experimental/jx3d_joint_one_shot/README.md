# JARVIS X | jointly trainable 3D one-shot codec

Isolated, headless C++17 experiment. It trains an 8-phase encoder and 224-parameter spatial nonlinear decoder on six synthetic periodic 3D fields using Adam with a straight-through estimator for the 8-bit latent. It tests two held-out fields and two out-of-distribution fields, executes bounded feedback, and verifies exact byte recovery from four competing self-contained packet methods.

## Build and test

Requires C++17, CMake, libzstd headers and library (Ubuntu: libzstd-dev).

    cmake -S experimental/jx3d_joint_one_shot -B build/jx3d_joint -DCMAKE_BUILD_TYPE=Release
    cmake --build build/jx3d_joint --parallel 2
    ctest --test-dir build/jx3d_joint --output-on-failure
    build/jx3d_joint/jx3d_joint --epochs 85 --budget-ms 1000 --out /tmp/jx3d
    build/jx3d_joint/jx3d_joint --decode /tmp/jx3d_encoded.jx3d --out /tmp/jx3d_recovered.raw
    cmp /tmp/jx3d_original.raw /tmp/jx3d_recovered.raw
    python3 experimental/jx3d_joint_one_shot/audit.py /tmp/jx3d

Outputs: PREFIX_metrics.csv, PREFIX_heldout.csv, PREFIX_original.raw, PREFIX_encoded.jx3d, and the decoded 4096-byte raw volume from the standalone decoder. All four candidate packets are individually decoded and checked before choosing the minimum.

## Scope and claim limits

16^3 physical uint8 input voxels, 8^3 latent uint8 cells, 232 trainable parameters. The neural candidate stores all 928 bytes of model parameters, 512 latent bytes, compressed 16-bit residuals, and a 20-byte common packet header; no hidden model is assumed. The other candidates are raw, Zstandard level 3 on raw voxels, and mean-pooling-plus-residual Zstandard. CRC32 catches accidental corruption, not deliberate tampering; the decoder depends on floating-point math and has no cross-platform fixed-point conformance guarantee.

Default local experiment (85 epochs) completed in approximately 329-355 ms including packet verification and file export, excluding compilation. Two unseen smooth fields improved; random out-of-distribution reconstruction worsened. On blocks, raw Zstandard wins; on random voxels, raw wins. These synthetic results are not proof of universal compression, clinical applicability, general intelligence or a deterministic one-second guarantee across hardware. Feedback is a contracting interpolation, not a learned recurrent policy. The packet protocol is a research prototype.

See MATHEMATICS.md for exact equations and packet specification. Scoped only to this experimental directory.
