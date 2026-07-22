# Jarvis X 800-Instance Runtime

This document describes the executable reference implementation in
`src/jarvisx/swarm800.py`.

## Scope

The runtime operationalises the corrected control-plane mathematics of the
800-instance swarm. It is a deterministic Python simulator for:

- signed Q8.8 arithmetic;
- the corrected 128-bit Spatial Virtual Instruction (SVI);
- a 4 KiB ROM containing 256 16-byte instructions;
- the 8 x 8 x 4 PC geometry;
- exact 128-dimensional instruction encoding and reconstruction;
- a 23 microsecond logical pipeline model;
- copy-on-write ROM mutation with protected fields;
- 800 instances arranged as 80 zones and 8 regions;
- zone, region, and global fusion cadences;
- sealed base-ROM manifests and dynamic hash chains;
- monotonic best-checkpoint loss.

It is not a WebGPU kernel, a real-time scheduler, a federated network, or a
trained 16,384-to-128 neural autoencoder. Those can be attached behind the
interfaces once the deterministic reference semantics are accepted.

## Run

```bash
pip install -e .
jarvisx swarm 1
```

Disable inward ROM mutation:

```bash
jarvisx swarm 10 --no-mutate
```

The command prints a JSON report containing the hierarchy counts, current
fusion metrics, accepted mutations, best loss, and sealed manifest hash.

## Canonical hierarchy

```text
800 instances
  = 80 zones x 10 instances
  = 8 regions x 10 zones x 10 instances
```

Fusion cadence:

| Level | Members | Cadence |
|---|---:|---:|
| Instance | 1 | every cycle |
| Zone | 10 instances | every 10 cycles |
| Region | 10 zones / 100 instances | every 100 cycles |
| Global | 8 regions / 800 instances | every 1000 cycles |

## Corrected SVI layout

| Field | Bits |
|---|---:|
| OPCODE | 8 |
| FLAGS | 8 |
| X | 12 signed |
| Y | 12 signed |
| Z | 12 signed |
| OPERAND | 32 |
| EDGE_FINGERPRINT | 32 |
| AGE_TIMER | 12 |
| **Total** | **128** |

The 32-bit edge field is explicitly a fingerprint, not a collision-free Morton
code for the complete signed 12-bit coordinate domain.

## ROM geometry

Each SVI occupies 16 bytes. A 4 KiB ROM therefore stores 256 instructions:

```text
4096 / 16 = 256 = 8 x 8 x 4
```

The byte-address PC mapping is:

```text
index = PC / 16
X = index & 7
Y = (index >> 3) & 7
Z = (index >> 6) & 3
```

The inverse is:

```text
PC = 16 x (X + 8Y + 64Z)
```

## Pipeline model

```text
FETCH 1 us -> ENCODE 5 us -> DECODE 5 us -> EXECUTE 2 us -> BACKPROP 10 us
```

Logical latency is the sum:

```text
1 + 5 + 5 + 2 + 10 = 23 us
```

Steady-state ideal throughput is bounded by the 10 microsecond bottleneck, not
by the reciprocal of 23 microseconds when stages overlap.

## Inward mutation transaction

Only packed bits 52 through 115 are mutable, covering the operand and edge
fingerprint. Opcode, flags, coordinates, and age are protected.

For each candidate:

1. choose three distinct mutable bits;
2. construct a shadow SVI;
3. validate its field ranges;
4. compare deterministic shadow fitness;
5. commit only if fitness improves by more than 0.1 percent;
6. append the committed cycle to the instance hash chain.

The immutable shared ROM is never rewritten. Every instance maintains a
copy-on-write patch table.

## Sealed invariants

The implementation enforces:

```text
8 x 10 x 10 = 800
PC mod 16 = 0
0 <= PC < 4096
-32768 <= Q8.8 register <= 32767
all SVI fields remain within their bit widths
all instances share the same base-ROM manifest hash
best_loss(t+1) <= best_loss(t)
```

## Validation

The test module `tests/test_swarm800.py` checks:

- Q8.8 conversion, multiplication, division, and divide-by-zero handling;
- exact 128-bit SVI packing and unpacking;
- exact 128-dimensional latent reconstruction;
- complete 4 KiB ROM coordinate round trips;
- 16-byte PC stride;
- protected-field mutation safety;
- the canonical 800 / 80 / 8 hierarchy;
- zone fusion after ten cycles;
- monotonic best-checkpoint loss.

Local validation used:

```text
8 passed
```

## Extension points

The deterministic `SVICodec` can be replaced by a learned encoder-decoder.
`SwarmInstance.step` can be backed by GPU kernels. `fuse_telemetry` can be
replaced by weighted federated aggregation. The present implementation fixes
the transaction semantics those backends must preserve.
