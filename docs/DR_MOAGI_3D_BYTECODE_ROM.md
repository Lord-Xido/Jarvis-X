# Dr. Moagi deterministic 3D bytecode ROM

This directory records a compact 64-byte big-endian ROM descriptor for a proposed
Q16.16 3D auto-encoding/decoding cycle.

## Canonical image

- Magic: `MOAG` (`4D 4F 41 47`)
- Image size: 16 words / 64 bytes
- Body size: 15 words / 60 bytes
- Checksum: IEEE CRC-32 over bytes `0x0000–0x003B`
- Canonical checksum: `0x96FDFC2F`
- Grid side-length immediate: `64`
- Grid cardinality implied by the descriptor: `64³ = 262,144` spatial sites

Generate the raw binary image with:

```bash
python -m jarvisx.dr_moagi_bytecode --output build/dr_moagi_3d_aed_v1.rom
```

The generated file must be exactly 64 bytes and end in:

```text
96 FD FC 2F
```

## Verification correction

The submitted stream ended in `7E 3F 91 A2`. Standard IEEE CRC-32, as implemented
by `zlib.crc32`, evaluates the preceding 60 bytes to `96 FD FC 2F`. The submitted
stream is retained in `rom/dr_moagi_3d_aed_v1.supplied.hex` for provenance, while
the canonical reproducible stream is stored in `rom/dr_moagi_3d_aed_v1.hex`.

## Word and instruction boundary

The image contains 16 total words, not 16 executable instructions:

1. word 0: magic metadata;
2. word 1: format/length metadata;
3. words 2–14: operation slots;
4. word 15: CRC-32.

The words at `0x0034` and `0x0038` are both `0x00000000`. Therefore `NOP` versus
`HALT` cannot be inferred from the opcode bits alone. This reference treats them
as distinct only by their fixed slot positions. A future ISA revision should
assign a unique halt encoding if instructions may be relocated.

## Endianness

The binary wire format is big-endian. A C `uint32_t[]` is a logical word view,
not a portable raw ROM image: its memory representation is reversed per word on
little-endian machines. `include/dr_moagi_bytecode.h` therefore exposes both an
endian-stable `uint8_t[]` and a logical `uint32_t[]`.

## Capability boundary

The 64-byte image specifies opcodes and operands. It does not contain:

- encoder or decoder weights;
- a 256-dimensional latent vector;
- convolution/deconvolution kernels;
- a numerical curl implementation;
- thermodynamic parameters beyond the encoded immediate;
- GPU synchronization code or a hardware transport contract.

`execute_rom_step` is consequently a deterministic trace decoder. It advances the
program counter and reports fields but does not claim to execute the named
mathematical kernels. A host runtime must bind and test those semantics.
