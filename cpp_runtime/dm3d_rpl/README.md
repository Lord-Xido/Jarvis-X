# DM3D-RPL

DM3D-RPL is the programmable C++20 frontend for the native DM3D bytecode runtime.

It provides:

- a small multimodal processing language;
- compilation to fixed-width 64-bit instructions;
- a tensor-register VM;
- video/audio/image/text/depth stream primitives;
- multimodal encode/fuse operations;
- implicit 3D geometric mapping;
- an LLM-style attention reference primitive;
- recursive inward folding;
- decode/residual/Omega feedback;
- bounded fixed-point refinement.

## Execution model

```text
media streams
  -> modal encode/embed
  -> shared latent fusion
  -> map3d
  -> llm_attend
  -> fold3d
  -> decode
  -> residual
  -> Omega memory
  -> fold3d ...
```

The example declares a 1,000,000 x 1,000,000 logical fabric, i.e. 10^12 logical positions. It remains sparse/virtual; the VM allocates only the tensors declared by the active program.

## Language

```text
fabric <side>
stream <name> <video|audio|image|text|depth|metadata|tokens> <elements>
zeros <name> <elements>
encode <src> -> <dst> <elements>
embed <src> -> <dst> <elements>
fuse <a> <b> -> <dst>
map3d <src> -> <dst> scale <float>
llm_attend <src> -> <dst> heads <n>
fold3d <src> <omega> -> <dst> lambda <float>
decode <src> -> <dst> <elements>
residual <target> <reconstruction> -> <dst>
omega <state> <residual> beta <float>
repeat <n> until <tensor> rms <epsilon> { ... }
emit <tensor>
halt
```

## Build

```bash
cmake -S cpp_runtime/dm3d_rpl -B build/dm3d-rpl -DCMAKE_BUILD_TYPE=Release
cmake --build build/dm3d-rpl --parallel
ctest --test-dir build/dm3d-rpl --output-on-failure
```

## Run source

```bash
build/dm3d-rpl/dm3d_rpl run cpp_runtime/dm3d_rpl/examples/multimodal.dm3d
```

## Compile and execute bytecode

```bash
build/dm3d-rpl/dm3d_rpl compile cpp_runtime/dm3d_rpl/examples/multimodal.dm3d /tmp/pipeline.bc
build/dm3d-rpl/dm3d_rpl exec /tmp/pipeline.bc
```

## Boundary

The language/compiler/VM is executable. Stream generation and `llm_attend` are reference kernels, not claims of production codec integration or a trained LLM. Production adapters can replace those boundaries with FFmpeg/libavcodec, tokenizer/model weights, CUDA/HIP/Metal/Vulkan compute, sparse 3D tensors and distributed scheduling without changing the DSL's control model.
