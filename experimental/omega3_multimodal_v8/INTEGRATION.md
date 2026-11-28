# Repository integration — 2026-11-28

Experimental CPU reference, isolated from the existing runtime. Source files and the original recorded verification report are imported from the v8 release; hybrid.py adds input contiguity normalization after a locally reproduced failure on strided arrays.

## Reproduce without external fixtures

From this directory:

    python -m pip install numpy pillow
    g++ -std=c++17 -O3 -Wall -Wextra core.cpp -lz -o core
    python smoke.py --iterations 1000000

Requires a C++17 compiler and zlib headers. The smoke test runs one million blocks under each native policy, independently decodes the first 256 blocks of each, checks sample packet equality and full-run hash/statistics equality, tests five synthetic modality-labelled tensors (including non-contiguous arrays), and rejects truncation. It does not test real media container I/O or model training. See smoke_report.json for local measured results (not GitHub Actions results).

For a new bounded neural generation checkpoint, install PyTorch CPU and FFmpeg with libx264rgb, then:

    python neural.py generator.pt preview --train --frames 60
    python generate_and_pack.py output --model generator.pt --frames 240

The trained-from-scratch checkpoint need not reproduce the original bytewise checkpoint across environments. Full v8 regeneration/refinement and the original 11-media regression require original v7 legacy_packets/ and checkpoint files. These binary assets, examples and output streams are deliberately not tracked in Git; retrieve them from the original downloadable release archive and place them beside the scripts as described in README.md. Keep the tracked original verification.json intact by running the full reproduction in a disposable working directory.

Original release archive: JARVIS-X_Omega3_Multimodal_v8.zip, SHA-256 7471e1397106a3c22b92a0e367330b8f3b14d148beba873693000c7a4548e616.
generator.pt SHA-256 301ac8116bab03c03c3430b5457179c137712b6fb35b2fdce2cbf58ffcac7513.
generator_refined.pt SHA-256 211c1d1353b6c1ef7d3a4df4e4d7c2c125578751c70ff6c5166ee55f9b20bf3b.

The original report records prior bounded tests, not new remote CI execution. No photorealism, GPU real-time, arbitrary external bytecode, global halo exchange, or general cross-modal intelligence is claimed. This is not a hardened untrusted-input decoder.
