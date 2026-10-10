# ANN OS — verified C++17 user-space prototype

**This is not a bootable kernel and does not perform arbitrary autonomous code synthesis.**
It is a conventional executable demonstrating real C++17 ANN training on byte tiles from its own source file, a reversible 3D byte representation, transactional validation-driven optimization, a target-aware scheduler, and an interactive shell. Its ANN learned reconstruction is lossy; the CodeCube byte representation is lossless.

## Build

```sh
g++ -O3 -std=c++17 -Wall -Wextra -pedantic ann_os.cpp -o ann_os
./ann_os --self ann_os.cpp
```

Or `cmake -S . -B build && cmake --build build && ctest --test-dir build --output-on-failure`.

## Commands

- `ls` list virtual files
- `stats` show validation MSE and timing
- `step` perform one transactional training/verification update
- `optimize` perform five verified updates
- `generate app hello` or `generate app xor` write deterministic C++ templates to `app/main.cpp`
- `debug` compile the generated template (requires `g++`)
- `doc` write a Markdown manual (`os_manual.md`), **not a PDF**
- `image` write a true SVG architecture diagram (`os_arch.svg`), **not a PNG**
- `replicate` save an exact source snapshot to `app/replica.cpp`
- `self-evolve` enable a built-in WAV synthesis routine and run one optimization update; **not self-modifying machine code**
- `audio` write a valid mono 8 kHz PCM WAV to `app/tone.wav`
- `schedule x y z` enqueue latent vector with priority based on Euclidean distance to `[0,0,0]`
- `run` dequeue and dispatch a vector (prints it; no arbitrary task execution)
- `save` save ANN checkpoint to `app/autoencoder.ann17`
- `exit` quit

## ANN training

- Topology: `8 -> 12 -> 3 -> 12 -> 8` using tanh activations.
- Input: 8 spatially adjacent voxels (actual `2x2x2` neighborhoods) from the source cube, mapped into `[-1,1]`.
- Objective: MSE reconstruction using mini-batch SGD.
- Held-out split: every fifth spatial tile; deterministic sampling of up to 1000 tiles.
- Transaction: copy network, train candidate, evaluate on validation, accept only if validation MSE improves by >1e-12. Otherwise drop candidate, retaining all original weights.
- **Training is not proof of increased speed or generalized software intelligence.**
- Compiler, `g++`, on host required only for `debug`. No third-party C++ dependencies.

## Limitations

`CodeCube` uses an axis-aligned 3D array, not a true periodic 3-torus. Text file bytes are mapped to voxels; there is no language model, code reasoning model, kernel scheduler, or bootloader. The source is limited to 64^3 bytes. Latent scheduler tasks are priority/dequeue demonstrations. This sample runs one process and does not require OpenMP. No network access or autonomous code execution is performed.

## Integrated Jarvis-X build

Standalone: `cmake -S cpp_runtime/ann_os -B build/ann-os && cmake --build build/ann-os && ctest --test-dir build/ann-os --output-on-failure`.

Parent build: `cmake -S cpp_runtime -B build/cpp-runtime && cmake --build build/cpp-runtime --target jarvisx-ann-os jarvisx-ann-os-tests && ctest --test-dir build/cpp-runtime -R '^ann-os-' --output-on-failure`.

The CLI and the generated-template compiler operate from the build directory because CMake copies `ann_os.cpp` and `ann.hpp` into that directory. CI checks GCC, Clang with sanitizers, and MSVC, then uploads platform-specific executable artifacts. No host kernel, autonomous compiler, or AI operating-system privileges are implied.
