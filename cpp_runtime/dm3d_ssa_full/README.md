# DM3D SSA multi-type code generator

A bounded, deterministic, testable C++17 implementation of the DM3D compiler loop. It preserves G = Z8 × Z8 × Z12 as a toroidal code genome, but adds explicit typed SSA intermediate representation and a self-contained C++17 translation-unit emitter.

## Capabilities

- Value types: Scalar, Vector[n], Matrix[m,n], Tensor3D[d,h,w].
- SSA nodes: Input, Constant, Copy, Add, Multiply, Relu, Conv3D, Pool2, Up2, Mean, Broadcast, Matmul, Dot, Reshape, Mix.
- Static graph validation: topological operand constraints, type/shape inference, divisibility for 3D pooling, matrix inner dimensions, reshape element count and finite parameters.
- Interpreter: per-element toroidal 7-point 3D stencil, average pool, nearest-neighbor upsample and tensor/scalar feedback.
- Compiler: emits standalone C++17 source containing the runtime and a fixed, typed program graph, compiles and executes it, and checks its entire output tensor against the interpreter.
- Evolution: shortest toroidal displacement, bounded mutation, elitism, adaptive mutation pressure. Candidates that violate task tolerance are discarded before performance scoring.

## Build / execute

    cmake -S cpp_runtime/dm3d_ssa_full -B build/dm3d-ssa -DCMAKE_BUILD_TYPE=Release
    cmake --build build/dm3d-ssa --config Release --parallel 2
    ctest --test-dir build/dm3d-ssa -C Release --output-on-failure
    ./build/dm3d-ssa/dm3d_ssa_codegen 8 16 8 ./build/dm3d-ssa/output

Arguments: generations, population, even-volume-side, output-directory.

Generated files: best_kernel.cpp, best_kernel executable, best_kernel.bin (full float32 tensor), evolution.csv.

## Verification

1. SSA shapes and operand legality.
2. Finite output; task max-element-error <= 0.055 before fitness scoring.
3. Generated executable output matches all elements of interpreter output within 1e-6.
4. Generated executable also meets task-error tolerance.

## Limits

This is a reference-grade proof of architecture, not a general-purpose optimizing compiler or autonomous programmer. The evolutionary grammar is constrained; code emission specializes the graph but reuses reference operator implementations rather than fusing native kernels. Timing fitness is hardware dependent and noisy. No CUDA or sandbox for untrusted arbitrary C++ code is included.
