# G3M 3D Bytecode Execution Map — proposed ISA v1

This package documents an **experimental instruction-plan encoding** for the independently built G3M LibTorch point-cloud autoencoder. The bytecode here is not yet executed by the LibTorch neural runtime. The C++17 disassembler validates the fixed pipeline, operand/dataflow types, bounds, CRC32, and shape transitions; it deliberately **does not train an ANN or run numerical tensor operations**. Use the separate G3M LibTorch engine for actual geometry training/inference.

## Program binary layout

All multi-byte integers are little-endian. No alignment padding. File size exactly `32 + instruction_count * 16`.

Header, 32 bytes:

| Offset | Size | Field |
|---:|---:|---|
| 0 | 8 | Magic ASCII `G3MBC001` |
| 8 | 2 | Version = 1 (u16) |
| 10 | 2 | Instruction width = 16 (u16) |
| 12 | 4 | Instruction count (u32) |
| 16 | 4 | Flags = 1 (LE) |
| 20 | 4 | Maximum input/output points (u32), max 1,000,000 |
| 24 | 4 | CRC32 of instruction body only (u32) |
| 28 | 4 | Reserved = zero (u32) |

Instruction, 16 bytes:

| Offset | Size | Field |
|---:|---:|---|
| 0 | 1 | opcode (u8) |
| 1 | 1 | flags (u8), must be zero |
| 2 | 2 | destination register (u16) |
| 4 | 2 | source register A (u16) |
| 6 | 2 | source register B (u16) |
| 8 | 4 | immediate (u32) |
| 12 | 4 | auxiliary (u32) |

`0xFFFF` means unused register. Register names are logical *typed tensors*, not 64-bit physical GPU registers. The fixed 15-instruction program contains a six-epoch `TRAIN_ADAM` macro-op and two refinement steps. Backend numerical implementation would map those macro-ops to the existing LibTorch functions, including Autograd. `aux=2000` in training indicates learning rate 2000/1e6=0.002; `aux=500` in attention means alpha 0.5; `aux=10` in refinement means max geometry error degradation 1%.

## Build and verify

```sh
python3 emit_plan.py
c++ -std=c++17 -O2 -Wall -Wextra -pedantic g3m_disasm.cpp -o g3m_disasm
./g3m_disasm g3m_pipeline.g3mb
```

A corrupted instruction byte will fail CRC32. Recomputed CRC32 with an illegal opcode, source register, output size, or wrong instruction ordering will fail semantic validation. CRC32 here is an accidental-corruption detector, **not authenticated code signing**.

## Geometry-memory semantics

Raw points are contiguous little-endian float32: `point[i].channel[c]` at offset `12*i + 4*c` for c=0,1,2. Shape `[1, 1_000_000, 3]` is 12,000,000 bytes. Support `[1,128,3]` is 1,536 float32 bytes. kNN `[1,128,12]` is 12,288 int64 bytes; attention `[1,128,128]` is 65,536 float32 bytes; latent `[1,512]` is 2,048 float32 bytes. Output is again 12,000,000 bytes. Tensor/allocator metadata and intermediate buffers are excluded.

## Correctness boundaries

- This is a *bytecode plan and disassembler*, not a runnable VM that performs neural inference. The G3M LibTorch project is the operational ANN component.
- `VERIFY` in this plan stands for shape/finite checks only, not geometric fidelity or exact reconstruction.
- `EVIDENCE` is a proposed provenance receipt. This disassembler emits no signed receipt and has no durable ledger.
- The ANN is lossy, the 512D latent does not preserve arbitrary million-point clouds, and no million-fold acceleration is established.
- For production deployment use a versioned ABI, tensor metadata, model weight hashes, authenticated envelopes, transaction boundaries, resource metering, and numerical conformance tests against the LibTorch backend.
