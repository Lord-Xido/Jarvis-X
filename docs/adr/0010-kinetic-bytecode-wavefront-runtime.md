# ADR-010: Kinetic Bytecode Wavefront Runtime

- **Status:** Proposed
- **Date:** 2026-08-16
- **Extends:** ADR-006, ADR-009
- **Decision scope:** bounded research/runtime semantics

## Context

ADR-006 defines Jarvis-X's accepted 3D geometric-diffusion kinetic boundary. ADR-009 defines
an accepted sparse native swarm ISA with deterministic fixed-width arithmetic and a bounded
resident working set. The missing link is an executable reference for the operational model in
which bytecode behaves as a moving wavefront:

```text
FETCH
-> DECODE
-> RESOLVE
-> ACTIVATE
-> MATERIALIZE
-> EXECUTE
-> PROJECT
-> VERIFY
-> ENCODE
-> COMMIT
-> EVICT
```

The declared virtual scale may be much larger than any physical allocation. In particular, one
requested research extent is

```text
N = 1,000,000 ^ 1,000,000 = 10 ^ 6,000,000
V = N ^ 3 = 10 ^ 18,000,000 virtual coordinates.
```

The repository must not turn that notation into a claim that the machine allocates or executes
all of those coordinates.

## Decision

Add `src/jarvisx/kinetic_bytecode_runtime.py` as a dependency-free deterministic reference
runtime for a sparse kinetic bytecode wave.

### Symbolic scale

The virtual axis extent is represented by `PowerExtent(base, exponent)`. The implementation
computes logarithmic scale and approximate address-bit requirements without evaluating or
allocating `base ** exponent`.

`canonical_million_power_space()` therefore represents

```text
1,000,000 ^ 1,000,000
```

per axis without constructing the full integer or a dense 3D array.

### Sparse region boundary

Physical work is identified by finite `RegionDescriptor` objects carrying 64-bit region
references. Runtime collections contain only active, resident, in-flight, and committed region
references.

The invariant is:

```text
physical_work(t) proportional to active/resident working set
physical_work(t) not proportional to declared virtual volume
```

### Kinetic pipeline

Each packet advances by at most one pipeline stage per logical tick. Multiple packets may occupy
different stages simultaneously, producing an observable wavefront rather than a monolithic
instantaneous operation.

The runtime records `(clock, packet_id, stage)` trace tuples so conformance tests can verify the
motion.

### Candidate-first execution

The reference `G3D` scalar projection is

```text
error     = observation - prediction
candidate = current + prediction - error + omega + immediate
```

where `current` is the last committed region value or the immutable observation for a new region.

The result does not become authoritative immediately. The pipeline applies:

```text
candidate
-> bounded projection
-> declared verification threshold
-> optional pure validator
-> deterministic quantization
-> research-state commit
```

A verification failure transitions to rollback and never updates the committed research state.

### Residency pressure

`max_resident_regions` is a hard physical-working-set bound. If a packet reaches
`MATERIALIZE` while the resident set is full, that packet stalls in place and increments
telemetry. It may proceed only after another packet reaches `EVICT`.

This gives the scheduler a concrete pressure/stall mechanism without pretending to materialize
the symbolic universe.

### Trust boundary

This runtime is a research-layer simulator. Its `committed` dictionary is local reference state;
it is **not** an authoritative `jarvisx.system_runtime` commit, tool authorization, external side
effect, or evidence that hardware executes the declared virtual volume.

The deterministic canonical VM and existing policy/transaction boundary remain authoritative.

## Consequences

1. The 64-bit canonical VM is unchanged.
2. The accepted native DM-vOmega-Xi swarm ISA is unchanged.
3. Kinetic wave semantics become executable and testable in Python.
4. Virtual scale is explicit but non-materializing.
5. Residency, stalls, projection, verification, rollback, and eviction become measurable.
6. Research-state commit remains below the authoritative system-control boundary.
7. Higher-dimensional or tensor payloads may replace the scalar fixture only if type, units,
   projection, verification, and resource bounds remain explicit.

## Validation

The focused test suite must verify:

- the `10 ^ 6,000,000` per-axis scale symbolically, with `10 ^ 18,000,000` virtual volume;
- full `G3D` packet traversal through the kinetic pipeline;
- projection before verification/commit;
- rollback on verification failure;
- bounded residency causing stalls rather than dense allocation;
- external validators being able to reject candidates deterministically.

## Promotion

Promote this ADR to **Accepted** only after the implementation PR passes the repository Python
matrix, formatting/lint/type checks, security/dependency checks, and focused kinetic-runtime
tests.
