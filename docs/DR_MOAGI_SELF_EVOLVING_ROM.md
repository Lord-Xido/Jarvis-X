# Dr Moagi Bounded Self-Evolving ROM

## Status

Executable reference implementation of a semantics-preserving ROM optimizer.

The runtime turns inward by analysing its own instruction stream, but it does
not permit arbitrary source rewriting or unverified firmware replacement.
Every accepted patch is drawn from a declared rule set, evaluated in a shadow
core from the same deterministic snapshot, and published as a new immutable ROM
version only when the final machine state is exactly equal and the instruction
cost is lower.

## Run

```bash
jarvisx ser 8
```

The command emits JSON containing:

- each optimization epoch;
- the accepted macro-op rule;
- baseline and shadow cycle counts;
- semantic-equivalence results;
- analysis-cycle share;
- the final ROM and hexdump;
- the parent-linked ROM version journal;
- the final deterministic state hash.

## 64-bit instruction format

```text
[ opcode:8 | rs:8 | rt:8 | ru:8 | immediate:16 | address:16 ]
```

Supported operations:

```text
NOP LOAD3D STORE3D ENC DEC MAC3D BUNDLE META LDC DSM HALT
```

`LDC` and `DSM` are macro-ops:

```text
LDC = LOAD3D + ENC
DSM = DEC + STORE3D
```

## Correct fusion conditions

A `LOAD3D` and `ENC` pair may be fused only when they are adjacent and the
encoder reads the same register written by the load:

```text
LOAD3D rs=a, addr=p
ENC    rs=a, rt=b
```

The replacement is:

```text
LDC rs=a, rt=b, addr=p
```

A `DEC` and `STORE3D` pair may be fused only when they are adjacent and the
store reads the decoder destination:

```text
DEC     rs=a, rt=b
STORE3D rt=b, addr=p
```

The replacement is:

```text
DSM rs=a, rt=b, addr=p
```

The originally proposed example attempted to fuse a `LOAD3D` with an `ENC`
across an intervening second load. That rewrite is not generally safe and is
intentionally rejected by this implementation.

## Shadow verification

For baseline ROM `B`, candidate ROM `B'`, and deterministic initial snapshot
`S0`, the optimizer executes:

```text
S_base   = EXECUTE(B,  S0)
S_shadow = EXECUTE(B', S0)
```

A patch is admissible only when:

```text
S_shadow == S_base
```

and:

```text
cycles(B') < cycles(B)
length(B') < length(B)
```

The reference backend uses exact deterministic tuple state, so equality is
byte-for-byte at the canonical serialized snapshot level.

## Analysis budget

The optimizer models the inward-analysis budget as the shadow-execution share:

```text
analysis_share = shadow_cycles / (baseline_cycles + shadow_cycles)
```

The default policy requires:

```text
analysis_share <= 0.5
```

Only one candidate is evaluated per epoch, keeping the reference controller
within the declared 50 percent analysis envelope.

## Versioned Delta-ROM publication

The active ROM is never modified without a version transition. Every committed
version records:

```text
version
parent_hash
manifest_hash
instruction_words
patch_rule
```

The manifest is SHA-256 over the parent manifest, patch-rule identifier, and
ordered 64-bit instruction words. This creates a causal ROM provenance chain.

## Fixed point

The engine reaches a fixed point when no declared fusion rule applies. For the
eight-instruction demo ROM, two verified rewrites reduce the program to six
instructions:

```text
8 instructions
  -> fuse LOAD3D + ENC
7 instructions
  -> fuse DEC + STORE3D
6 instructions
  -> no admissible rule
```

This is a fixed point relative to the current rule set and benchmark semantics.
It is not a proof of global optimality, universal computation in zero cycles, or
precomputed answers to arbitrary queries. A six-instruction program still
requires physical execution, memory movement, and finite clock cycles.

## Safety invariants

1. Only declared adjacent macro-op rules may modify the ROM.
2. Baseline and candidate execute from the same deterministic snapshot.
3. Candidate state must equal baseline state exactly.
4. Candidate instruction and dynamic-cycle costs must both decrease.
5. Analysis share may not exceed the configured bound.
6. Every accepted ROM version is immutable and parent-linked.
7. No wall-clock value or external randomness enters the state hash.
8. A rule-set fixed point is reported honestly as bounded convergence.
