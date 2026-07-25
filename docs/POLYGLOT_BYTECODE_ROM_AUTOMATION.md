# Jarvis-X Polyglot Bytecode ROM Automation

## Purpose

This subsystem compiles multiple source representations into the existing Jarvis-X 64-bit instruction format, packages those words in a deterministic ROM image, verifies integrity during decoding, and automates repeatable builds through GitHub Actions.

```text
Jarvis Assembly ─┐
JSON IR ─────────┼─> Normalized AST ─> Existing Assembler ─> 64-bit words
Restricted Python┘                                      │
                                                        v
                                               JXROM autoencoder
                                                        │
                                                        v
                                     digest-verified ROM artifact
                                                        │
                                                        v
                                        autodecoder / decompiler
```

## Supported source languages

### Jarvis assembly

```text
SET A 7
SET B 5
ADD C A B
SUB D C B
HALT
```

### JSON intermediate representation

```json
[
  {"op": "SET", "args": ["A", 7]},
  {"op": "SET", "args": ["B", 5]},
  {"op": "ADD", "args": ["C", "A", "B"]},
  {"op": "SUB", "args": ["D", "C", "B"]},
  {"op": "HALT", "args": []}
]
```

### Restricted Python

```python
A = 7
B = 5
C = A + B
D = C - B
halt()
```

The Python front end intentionally supports only deterministic VM constructs:

- integer assignment maps to `SET`
- register addition maps to `ADD`
- register subtraction maps to `SUB`
- `halt()` maps to `HALT`

Register aliases such as `PSI`, `PHI`, `LAMBDA`, `OMEGA`, `THETA`, `SIGMA`, `XI`, and `PI` are mapped to the symbolic Jarvis-X registers.

## JXROM/1 format

Each ROM image contains:

| Field | Purpose |
|---|---|
| Magic | Identifies a Jarvis-X ROM image |
| Format version | Enables future evolution |
| Language identifier length | Locates the source-language label |
| Metadata length | Locates canonical JSON metadata |
| Instruction count | Determines payload size |
| Source SHA-256 | Binds the artifact to its source text |
| Payload SHA-256 | Detects bytecode corruption or tampering |
| Language identifier | Records the selected front end |
| Canonical metadata | Reproducible, sorted build metadata |
| 64-bit words | Big-endian Jarvis-X bytecode payload |

Metadata is serialized with sorted JSON keys and compact separators. The same source, language, and metadata therefore produce the same ROM bytes.

## Command line

Compile a source file:

```bash
python -m jarvisx.polyglot_cli compile \
  --language python \
  --input examples/polyglot/program.py \
  --output build/program.jxrom
```

Inspect a ROM:

```bash
python -m jarvisx.polyglot_cli inspect --input build/program.jxrom
```

Decode a ROM back to canonical assembly:

```bash
python -m jarvisx.polyglot_cli decode \
  --input build/program.jxrom \
  --output build/program.jxasm
```

## Build automation

`PolyglotRomAutomation` performs the closed loop:

```text
SourceUnit
  -> compile
  -> encode ROM
  -> decode and verify hashes
  -> decompile to canonical assembly
  -> compile replay
  -> compare canonical 64-bit words
  -> emit verified result
```

The dedicated GitHub Actions workflow:

1. tests Python 3.8, 3.10, and 3.12;
2. compiles the assembly, JSON IR, and Python examples;
3. confirms that all front ends emit identical instruction words;
4. writes a SHA-256 manifest; and
5. uploads the verified ROM images as a workflow artifact.

## Extension contract

A new language adapter must:

1. parse its source without executing arbitrary code;
2. lower into the normalized instruction arrays accepted by the existing assembler;
3. reject unsupported or ambiguous constructs;
4. append `HALT` when absent;
5. preserve deterministic output; and
6. pass word-equivalence and ROM integrity tests.

This keeps the language layer replaceable while preserving the Jarvis-X VM, ISA, ledger, policy gate, tracing, and execution semantics beneath it.
