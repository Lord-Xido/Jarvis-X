# NEURAL ARENA ΩΞ v1.3 — A17 Pro device bitwise conformance

This package contains an iOS Metal application that **runs actual arithmetic parity checks on-device** once built, signed, and launched on an iPhone 15 Pro Max. The package is not proof of physical device execution until its report is exported from that hardware.

## Tests

- The original v1.2 signed Q12 six-neighbor operator: `average = round_symmetric(sum / valid_neighbors)`, `next = clamp(round_symmetric((17*current + 3*average)/20),[-4096,4096])`. Singleton fields use the current value for an absent-neighbor average.
- 3 channels per latent site, `x`-major, 16-bit signed little-endian values.
- 9 cases × 3 iterations = 27 SHA-256 known-answer tests.
- Input dimensions include `1×1×1`, `1×9×5`, `5×7×3`, `8×8×4`, `9×10×5`, `17×15×9`, and `64×64×32`.
- Two physically distinct Metal implementations: global-memory stencil and threadgroup-memory cooperative-halo stencil. The halo tile is 10×10×6 with three channels and all padded lanes synchronized.
- CPU mathematical oracle vs each GPU implementation **every iteration**, and CPU vs independently generated C++ / NumPy golden SHA-256 values.
- Mismatch evidence includes spatial coordinates, channel, expected/actual signed values, XOR mask and changed-bit count.
- Report includes `hw.machine` model identifier, GPU name, `supportsFamily(.apple9)`, simulator flag, command-buffer timing, pass/fail and per-step SHA-256 digests.
- Only `iPhone16,2` (iPhone 15 Pro Max) with Apple9 and successful all-step comparisons sets `deviceBitwiseValidated: true`.

## Compile/run on the real iPhone

1. On a Mac with Xcode and iOS platform components installed, extract this directory and open `NeuralArenaBitwise.xcodeproj`.
2. Configure your Apple Development team under **Signing & Capabilities**. If prompted, install the Xcode Metal Toolchain (`xcodebuild -downloadComponent metalToolchain`).
3. Connect your iPhone 15 Pro Max and select it as the run destination, then press **Run**.
4. Tap **Run A17 Pro GPU Bitwise Validation**. The two GPU kernels execute on the phone, and the app computes all three SHA-256 hashes for each test stage.
5. If all 27 comparisons match, the display says **PASS — A17 Pro hardware verified**. Export the JSON using **Export device evidence JSON**.
6. Inspect the JSON's `deviceBitwiseValidated`, `hardwareModel`, `casesPassed`, `casesTotal`, `tests`, and `exceptions` fields. A compile-only or simulator report must **not** be recorded as an A17 Pro hardware pass.

## Cloud check

The included GitHub Actions workflow provides cloud **iOS Metal compilation** (macOS Xcode), not physical iPhone execution. A GitHub-hosted Mac is not evidence of A17 Pro timing, thermal behaviour, or device bitwise parity.

## Local numerical check

```
python3 reference/oracle_numpy.py
```

This compiles independent C++20 code, generates full binary outputs for each case, and computes NumPy output independently. Every byte is compared; `reference/GOLDEN.json` records authoritative SHA-256 hashes. The iOS app has those expected hashes embedded.

## Boundaries

This suite validates the v1.2 **3-channel latent-field refinement** on the GPU, not the full 8→3→8 encoder, neural weight training, PNG decoding or a 1 MiB lossless archive. `64×64×32` has **131,072 latent sites**, each with 3×Int16 = 786,432 bytes per field, corresponding to the 131,072 patches of the 1,048,576-byte original source. It does not require or allocate the user’s much larger virtual-address cube.

No telemetry is uploaded, no paid GPU instance is provisioned, and evidence remains on the phone until exported.
