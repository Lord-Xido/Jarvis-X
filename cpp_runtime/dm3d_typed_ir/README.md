# DM3D Typed 3D IR Synthesizer

Graph-level successor to the DM3D kernel autotuner. It evolves a computational graph, verifies task error and numerical stability, uses a multi-objective score, adapts mutation pressure, emits a generated C++ translation unit, compiles it, executes it, and verifies the lowered program against the in-process IR interpreter.

## Operator vocabulary

The current Volume3D<float> -> Volume3D<float> IR implements DMA, CONV3D, ENCODE, DECODE, MATMUL, ATTENTION, FFT, REDUCE, and FIX_POINT.

## Closed loop

graph genome -> typed IR -> execute -> task verify -> multi-objective score -> select -> mutate -> adapt search policy -> emit C++ -> compile -> execute -> lowering verify -> recur

## Verification

1. Graph size and output finiteness.
2. Hard task constraint: max_error(candidate,target) <= 0.065.
3. Lowering invariant: the generated program checksum must agree with the interpreter within 1e-5 and the emitted program must remain inside the task-error bound.

Performance and complexity are optimization objectives only after hard constraints pass.

## Search-policy adaptation

Improvement contracts mutation pressure; stagnation expands it. This is a bounded outer-loop search-policy adaptation, not a claim of autonomous self-modification.

## Build and run

    cmake -S cpp_runtime/dm3d_typed_ir -B build/dm3d_typed_ir -DCMAKE_BUILD_TYPE=Release
    cmake --build build/dm3d_typed_ir --parallel 2
    ./build/dm3d_typed_ir/dm3d_typed_ir 8 16 10 build/dm3d_typed_ir/generated/typed_ir_kernel.cpp

A deterministic local validation evolved the redundant seed toward a smaller graph while preserving the hard task bound, then compiled and ran the generated translation unit with an interpreter-vs-lowered checksum difference around machine precision. Exact runtime and selected graph can vary by compiler and hardware.

## Next extension

Expand the value system into SSA-style Tensor3D, Scalar, Matrix, and TokenSet types with explicit shape inference, legality checks, backend-specific lowering, and top-K compile/run benchmarking during evolution.
