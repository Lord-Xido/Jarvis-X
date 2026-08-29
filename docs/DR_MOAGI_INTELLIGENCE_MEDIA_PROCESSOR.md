# Dr Moagi Intelligence Media Processor

> **Normative 8KB container contract:** the bounded recursive state, arithmetic, fixed-point, transactional microcode, semantic-gap, toroidal-topology, and hardware-claim semantics for this processor family are defined in [`DM_VOMEGAXI_8KB_CANONICAL_SPEC.md`](DM_VOMEGAXI_8KB_CANONICAL_SPEC.md). This document describes the currently implemented DM-IMP/VCL-BVM-8 reference layer; the canonical specification distinguishes implemented behavior from required conformance and future hardware targets.

## Operational contract

The Dr Moagi Intelligence Media Processor (DM-IMP) is a bounded 8-bit spatial media execution layer for Jarvis-X. It does not replace the global sparse JX3DVM1 address space. Instead, arbitrary input bytes are divided into `8 x 8 x 8 = 512` byte tiles and processed locally through VCL-BVM-8.

The execution hierarchy is:

```text
arbitrary media bytes
      |
      v
512-byte / 8x8x8 VCL tile
      |
      v
CONV_3D_INT8 -> VCL_GATE -> ENC_SPATIAL -> EVAL_ENTROPY
      |                                         |
      |                                         v
      |                                  PRUNE_LATTICE
      |                                         |
      +-----------------> AUTO_EVOLVE <---------+
                              |
                              v
                       DEC_BYTECODE
                              |
                              v
                         SYNC_LOCK
                              |
                              v
                 Lambda quality validation
                    /               \
                 accept            reject
                   |                  |
                   v                  v
              keep Theta/Omega   restore Theta/Omega
                   |                  |
                   +-------> output/fallback
```

The implementation maps the project stack to explicit machine state:

- **Psi**: the complete tile observation/transform path;
- **Phi**: `ENC_SPATIAL` plus `DEC_BYTECODE`, giving an executable description/reconstruction operator;
- **Lambda**: the outer MSE quality gate and the inner candidate comparison in `AUTO_EVOLVE`;
- **Omega**: an 8-element integer latent memory fused into the 2x2x2 core;
- **Theta**: 512 signed 8-bit adaptive node weights;
- **M-hat**: `logic_mask AND control_flag` at each node;
- **hbar_semantic**: normalized output RMSE, `sqrt(MSE) / 255`.

A rejected tile restores both Theta and Omega to the snapshot taken before execution. If fallback is enabled, the original input tile is emitted, so a failed candidate cannot silently degrade authoritative media bytes.

## VCL-BVM-8 wire format

Except for `SYNC_LOCK = 0xFF`, each one-byte instruction header is split as:

```text
7 6 5 4 | 3 2 1 0
 opcode | mode/reg
```

The current v1 implementation reserves the low nibble for future addressing/register modes and dispatches on the high nibble.

| High nibble | Mnemonic | Payload | Runtime behavior |
| --- | --- | --- | --- |
| `0x10` | `INGEST_RAW` | `z_slice, value` | Fill one Z plane with a byte value |
| `0x20` | `ENC_SPATIAL` | `z_start, z_end` | Average active nodes into a 2x2x2 core and fuse Omega |
| `0x30` | `VCL_GATE` | `gate, threshold` | AND/OR/XOR/NOT control gating |
| `0x40` | `CONV_3D_INT8` | `kernel_id, bias` | Signed INT8 3x3x3 convolution with defined boundary clamping |
| `0x50` | `EVAL_ENTROPY` | `shell_id, threshold` | Compute actual empirical byte entropy and raise prune request |
| `0x60` | `PRUNE_LATTICE` | `target_mask` | Apply requested low-information/control pruning |
| `0x70` | `DEC_BYTECODE` | `z_start, z_end` | Decode the fused core back into tile space |
| `0x80` | `AUTO_EVOLVE` | `learning_rate` | Stage residual weight update and retain it only if reconstruction MSE does not increase |
| `0xFF` | `SYNC_LOCK` | none | Commit Omega update and end the VCL cycle |

All instruction payload reads are bounds checked. Streams that end without `SYNC_LOCK`, contain an unknown opcode/kernel/gate/shell, address an invalid Z range, or exceed the instruction ceiling are rejected.

## Kernel bank

The initial deterministic convolution bank provides four signed integer kernels:

1. `0`: identity;
2. `1`: 27-point box smoothing;
3. `2`: 3D Laplacian-style edge response;
4. `3`: 6-neighbor sharpening.

Every convolution uses a 32-bit accumulator and requantizes into signed INT8. Unlike the earlier reference pseudocode, the kernel operand is therefore active and boundary cells, including Z=0, are defined.

## Inward encoding

The input tile contains 512 signed states. The latent core contains 8 signed states:

```text
512 -> 8
```

which is a geometric state ratio of `64:1`. Each core bucket accumulates all active nodes with the corresponding `(x mod 2, y mod 2, z mod 2)` parity and divides once after accumulation. This avoids the order-dependent repeated half-average of the original pseudocode.

The core then fuses bounded integer memory:

```text
core_fused = (3 * core_raw + Omega) / 4
Omega_next = (7 * Omega + core_raw) / 8
```

`Omega_next` becomes authoritative only when the enclosing tile survives the Lambda quality gate.

## Transactional auto-evolution

`AUTO_EVOLVE` is deliberately candidate-first:

```text
W_before -> shadow decode -> baseline MSE
         -> candidate residual update
         -> shadow decode -> candidate MSE
```

The candidate is retained only if:

```text
candidate_MSE <= baseline_MSE
```

The processor then applies a second outer gate:

```text
candidate_MSE <= max_output_mse
```

A failure at the outer gate restores the complete adaptive snapshot, including weights and Omega.

This is bounded model-state adaptation. It does not rewrite source code, execute host commands, or establish a claim of frontier/SOTA intelligence.

## Media interface

The processor accepts arbitrary binary input, so it can operate on encoded image, audio, text, video, container, model or generic byte streams without a codec dependency. At this layer media modality is an execution/telemetry classification rather than a promise that compressed file formats remain semantically decodable after lossy transformation.

For format-aware image/audio/video processing, codec-specific adapters should decode the source format into raw frames/samples before DM-IMP and re-encode after processing.

## CLI

CMake target:

```text
jarvisx-intelligence-media
```

Windows output name:

```text
DrMoagi-Intelligence-Media.exe
```

Process a file:

```powershell
.\DrMoagi-Intelligence-Media.exe `
  --input .\input.bin `
  --output .\processed.bin `
  --modality generic `
  --passes 2 `
  --max-mse 4096
```

Run the deterministic visual demo:

```powershell
.\DrMoagi-Intelligence-Media.exe `
  --demo-bytes 4096 `
  --modality visual `
  --passes 2 `
  --output .\demo-media.bin
```

Execute a custom raw VCL-BVM-8 program:

```powershell
.\DrMoagi-Intelligence-Media.exe `
  --input .\input.bin `
  --bytecode .\program.vcl8 `
  --output .\processed.bin
```

## Canonical default program

```text
40 00 00       CONV_3D_INT8 identity, bias 0
30 02 00       VCL_GATE OR, threshold 0
20 00 07       ENC_SPATIAL all Z slices
50 00 08       EVAL_ENTROPY whole tile, threshold 8
60 01          PRUNE_LATTICE gate-disabled nodes
80 08          AUTO_EVOLVE rate 8
70 00 07       DEC_BYTECODE complete tile
FF             SYNC_LOCK
```

## Verification

`intelligence_media_processor_tests.cpp` checks:

- signed byte centering round trips;
- defined convolution at the tile boundary;
- true entropy distinguishes uniform and varied tiles;
- default VCL execution reaches `SYNC_LOCK`;
- the corrected documented VCL cycle executes;
- truncated bytecode is rejected;
- Lambda fallback preserves original bytes and restores adaptive state;
- deterministic replay produces identical results.

CMake also registers a full executable smoke run so the CLI path is exercised by the existing cross-platform C++ workflow.
