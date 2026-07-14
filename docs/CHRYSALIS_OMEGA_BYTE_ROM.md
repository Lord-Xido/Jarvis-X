# Chrysalis-Ω Deterministic 3D Byte-ROM

## Status

Operational Jarvis-X bytecode storage and execution subsystem.

This implementation replaces the earlier untrained latent reconstruction concept with a deterministic, lossless byte pipeline built around the repository's real 64-bit Jarvis-X instruction set. It does not claim neural compression or unrestricted self-rewriting.

## Runtime pipeline

```text
Jarvis-X source
    -> Parser
    -> Assembler
    -> unsigned 64-bit instruction words
    -> JXBC bytecode image
    -> optional zlib compression
    -> fixed-capacity engine × X × Y × Z ROM cells
    -> JXROM3D file
```

The inverse path verifies the ROM, reconstructs the exact bytecode image, unpacks the instruction words, and only then loads the VM.

## Mathematical model

Let the assembled program be

\[
W=(w_0,w_1,\ldots,w_{n-1}),\qquad 0\le w_i<2^{64}.
\]

Each instruction is serialized in network byte order:

\[
B_i=\operatorname{BE64}(w_i).
\]

The bytecode image is

\[
P=\texttt{JXBC}\;\Vert\;v\;\Vert\;n\;\Vert\;B_0\Vert\cdots\Vert B_{n-1}.
\]

For a ROM geometry with \(G\) engines, dimensions \((X,Y,Z)\), and \(C\) bytes per cell, total capacity is

\[
K=GXYZC.
\]

The payload must satisfy

\[
|P_s|\le K,
\]

where \(P_s\) is either the raw bytecode image or its compressed representation.

The linear cell index is

\[
q=(((gX+x)Y+y)Z+z).
\]

The first byte of that cell is stored at body offset

\[
o=qC.
\]

A stored byte offset \(r\) maps back to its cell by

\[
q=\left\lfloor\frac{r}{C}\right\rfloor,
\qquad
u=r\bmod C,
\]

where \(u\) is the byte position inside the cell.

## Integrity contract

The ROM header stores:

- format magic and version;
- compression flags;
- engine and grid dimensions;
- bytes per cell;
- stored payload length;
- uncompressed payload length;
- SHA-256 of the uncompressed payload.

A ROM is accepted only when:

\[
V=V_{format}\land V_{capacity}\land V_{padding}\land V_{length}\land V_{sha256}.
\]

Non-zero padding, malformed sizes, unknown feature flags, decompression failure, and digest mismatch are rejected before execution.

## Binary formats

### JXBC bytecode envelope

All integers are big-endian.

| Field | Size |
|---|---:|
| Magic `JXBC` | 4 bytes |
| Version | 1 byte |
| Reserved | 3 bytes |
| Word count | 4 bytes |
| Instruction words | `count × 8` bytes |

### JXROM3D envelope

| Field | Size |
|---|---:|
| Magic `JXROM3D\0` | 8 bytes |
| Version | 1 byte |
| Flags | 1 byte |
| Reserved | 2 bytes |
| Engines, X, Y, Z | 8 bytes |
| Cell size | 4 bytes |
| Stored size | 8 bytes |
| Raw size | 8 bytes |
| SHA-256 | 32 bytes |
| Fixed-capacity ROM body | `G × X × Y × Z × C` bytes |

## CLI

Encode assembly source into a ROM:

```bash
jarvisx rom encode program.jx program.jrom \
  --engines 2 \
  --grid 4x4x4 \
  --cell-bytes 32
```

Let the runtime choose the minimum cell size:

```bash
jarvisx rom encode program.jx program.jrom --grid 4x4x4
```

Enable compression when it reduces the payload:

```bash
jarvisx rom encode program.jx program.jrom --compress
```

Inspect and verify:

```bash
jarvisx rom inspect program.jrom
jarvisx rom verify program.jrom
```

Execute only after verification:

```bash
jarvisx rom run program.jrom
```

Extract the exact JXBC bytecode envelope:

```bash
jarvisx rom extract program.jrom program.jxbc
```

## Bounded mutation

The mutation command changes only the 16-bit immediate field of one `SET` instruction:

```bash
jarvisx rom mutate program.jrom candidate.jrom \
  --word-index 0 \
  --delta 5
```

For an instruction word \(w\), the immediate is

\[
i=(w\gg8)\land 0xffff.
\]

The candidate is formed as

\[
w'=\left(w\land\neg(0xffff\ll8)\right)\lor((i+\Delta)\ll8).
\]

The operation is rejected unless:

- the selected word exists;
- its opcode is `SET`;
- the delta is an integer;
- the resulting immediate remains in `0..65535`;
- the candidate still fits the declared ROM capacity.

The output candidate receives a new SHA-256 and is independently verifiable. The command does not execute or promote the candidate automatically.

## Operational guarantees

The subsystem provides:

- exact 64-bit instruction round trips;
- deterministic big-endian serialization;
- fixed-address 3D cell layout;
- atomic file replacement on writes;
- checksum verification before VM execution;
- capacity enforcement;
- bounded field-level mutation;
- no PyTorch or external runtime dependency.

## Non-goals

This implementation does not provide:

- learned latent compression;
- arbitrary source regeneration;
- CPython bytecode storage;
- semantic equivalence proofs for mutated programs;
- automatic deployment of mutations;
- physical read-only hardware.

`ROM` denotes the persisted, fixed-layout runtime image. Immutability is enforced at the object and verification layers; the file can still be replaced by an authorized process.
