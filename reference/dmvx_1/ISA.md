# DMVX-1 Reference ISA Contract

## 1. Scope

DMVX-1 is a bounded reference instruction set for the DM-vOmegaXi+ transactional auto-encoding firmware. It defines deterministic control flow and state transitions. It does **not** claim physical `1000^1000 GB` storage, zero latency, or guaranteed lossless reconstruction.

## 2. Machine model

- Little-endian.
- 32-bit addresses and 32-bit general registers `R0..R15`.
- `PC`, `SP`, `FP`, and `FLAGS` are architectural registers.
- Q16.16 is the canonical scalar fixed-point format.
- The stack grows downward and is checked against declared bounds.
- Persistent regions are modified only by `ATOMIC_COMMIT`.

## 3. Operand convention

The destination operand is always first:

```text
MOV destination, source
ADD destination, lhs, rhs
STORE address, source
LOAD destination, address
```

Instructions that can fail write a status code to their final register operand.

## 4. Instruction encoding

The normative source is assembly-like. A binary assembler may encode common instructions as one 32-bit word:

```text
31          24 23          16 15           8 7            0
+-------------+--------------+--------------+--------------+
| opcode      | operand A    | operand B    | operand C    |
+-------------+--------------+--------------+--------------+
```

Immediate, address, and variable-length data instructions may consume extension words. The source-level semantics, not a specific binary layout, are authoritative for this reference release.

## 5. Transaction invariants

For committed state `Omega_t` and candidate state `Omega~`:

```text
Omega_(t+1) = Omega~   when Lambda = 1
Omega_(t+1) = Omega_t  when Lambda = 0
```

The following invariants are mandatory:

1. Encoding and decoding operate on a candidate buffer.
2. Candidate data cannot alias committed persistent storage.
3. Failed validation clears the candidate buffer.
4. Retry count is bounded by `MAX_RETRIES`.
5. Every subroutine restores its stack frame on every return path.
6. Exact equality is never required after lossy quantization; a declared tolerance is used.
7. Every accepted or rejected transaction appends a provenance receipt.
8. Only ROM, stack, isolation, or atomic-commit integrity failures may halt the machine.

## 6. Core instruction semantics

### `WAIT_BUS destination`
Blocks until a payload handle is available. Writes `BUS_EMPTY` only when interrupted without a payload.

### `BEGIN_TX destination`
Creates a monotonically increasing transaction identifier.

### `ENCODE_LATENT input, theta, output, status`
Applies the declared deterministic encoder to `input` using modulation state `theta`.

### `CALC_FREE_ENERGY latent, destination, status`
Computes a normalized Q16.16 objective. The reference VM uses:

```text
F(z) = mean(z^2) + beta * mean(abs(z))
```

### `QUANTIZE_Q16 source, destination, status`
Quantizes each scalar into signed Q1.15 storage and projects back into Q16.16 semantics.

### `DECODE_LATENT latent, theta, output, status`
Decodes staged latent state. The decoder must be compatible with the encoder over the declared operating domain.

### `CALC_DISTANCE lhs, rhs, destination, status`
Computes normalized mean absolute error in Q16.16.

### `POLICY_CHECK candidate, theta, evidence, status`
Evaluates a declared policy predicate. It is an admissibility mechanism, not proof of moral correctness.

### `ATOMIC_COMMIT candidate, committed, status`
Replaces committed state as one transaction. On failure, committed state remains unchanged.

### `APPEND_RECEIPT log, tx, digest, energy, distance, outcome`
Appends an immutable transaction record containing identifiers, measurements, hashes, and outcome.

## 7. Virtual manifold rule

The manifold is a logical address domain backed by bounded active pages:

```text
A_t subset M
|A_t| <= MAX_ACTIVE_PAGES
```

`INIT_VMANIFOLD` initializes page metadata only. It never allocates an unbounded or continuous physical memory space.

## 8. Verification condition

A candidate can commit only when all of the following hold:

```text
finite(candidate)
free_energy <= FREE_ENERGY_LIMIT
reconstruction_distance <= RECON_TOLERANCE
bounds_valid(candidate)
budget_valid(candidate)
authorization_valid
audit_evidence_valid
```

