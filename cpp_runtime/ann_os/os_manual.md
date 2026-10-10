# ANN OS — User-Space C++17 Runtime

This is a normal process with an ANN learning loop, not a bootable operating system.

## Architecture

Source bytes -> lossless CodeCube (3D voxel array) -> spatial 2x2x2 sample dataset -> 8-12-3-12-8 autoencoder -> reconstruction MSE -> train candidate -> compare held-out MSE -> keep or roll back.

The goal-aware scheduler dispatches a 3-element latent vector closest to an explicit goal. This is a demonstration scheduler, not kernel task scheduling.

## Building

Standalone: `cmake -S cpp_runtime/ann_os -B build/ann-os && cmake --build build/ann-os && ctest --test-dir build/ann-os --output-on-failure`.

Integrated parent: `cmake -S cpp_runtime -B build/cpp-runtime && cmake --build build/cpp-runtime --target jarvisx-ann-os jarvisx-ann-os-tests`.

Run the executable from its build directory with `./ann_os --self ann_os.cpp`, or use `--batch` to run without the interactive shell.

## Scope

Generation supports only known-safe, deterministic `hello` and `xor` C++ templates. The `self-evolve` command registers built-in audio output and performs one held-out optimization step; it does not modify or recompile its own binary. `doc` generates Markdown and `image` generates SVG; neither produces a PDF or PNG.

The candidate's measured improvement is reconstruction MSE, not execution speed.
