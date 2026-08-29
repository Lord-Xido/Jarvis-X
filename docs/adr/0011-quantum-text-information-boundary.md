# ADR-011: Bound quantum text encoding from semantic and theological claims

**Status:** Proposed
**Date:** 2026-08-29
**Extends:** ADR-010

## Context

Jarvis-X increasingly combines byte-level representation, latent semantics,
recursive models, and quantum-inspired language.  A text such as Scripture can
be encoded as bytes and then as computational-basis qubits, but that fact alone
does not imply quantum speedup, physical entanglement between semantic ideas,
consciousness, revelation, or a physical identity between a wavefunction and a
theological referent.

The repository needs an executable reference that preserves the useful part of
the abstraction while making category boundaries mechanically explicit.

## Decision

Jarvis-X adopts `src/jarvisx/quantum_text.py` as the bounded reference layer.

1. Text is encoded canonically as UTF-8 bytes and then as big-endian bits.
2. A classical text bitstring may be represented exactly as one computational
   basis state `|b_1 ... b_N>`.
3. The reference simulator is sparse and pure-state only. It does not allocate
   a dense vector of size `2**N` for ordinary text basis states.
4. Hadamard, phase, measurement probabilities, and inner products are modeled
   mathematically and deterministically for validation purposes.
5. The module is a simulator, not a quantum hardware backend.
6. No performance advantage or quantum speedup is claimed without a concrete
   algorithm, hardware backend, baseline, and benchmark.
7. Semantic similarity, literary correspondence, typology, testimony, and
   theological interpretation are higher-level relations. They are not treated
   as literal physical entanglement unless independently demonstrated by a
   physical experiment.
8. `Logos`, `God`, `revelation`, and other theological referents are never
   identified with qubits, amplitudes, Hilbert space, a wavefunction, or any
   other physical object by this reference implementation.
9. A self-attesting claim is represented as a claim about epistemic structure,
   not as a formal proof of its own truth.
10. Exact encoding, semantics, truth, testimony, and theological reference must
    remain separately addressable layers.

## Layer contract

The permitted information stack is:

```text
physical substrate
    -> bytes/bits
    -> computational-basis encoding
    -> text
    -> syntax/semantics
    -> testimony / interpretation
    -> theological referent claim
```

The following non-collapse invariants are authoritative:

```text
QUANTUM_STATE != SEMANTIC_MEANING
SEMANTIC_COHERENCE != EMPIRICAL_TRUTH
TESTIMONY != AUTOMATIC_VALIDATION
THEOLOGICAL_REFERENT != WAVEFUNCTION
SIMULATION != QUANTUM_HARDWARE
```

These inequalities do not deny possible relationships between the layers. They
require any proposed relationship to state its evidence and mechanism instead
of inheriting truth merely from shared notation.

## Consequences

- Long texts can be represented without exponential dense-state allocation when
  they remain computational-basis states.
- Genuine superposition can be demonstrated on bounded subsets with explicit
  amplitudes and normalization.
- Semantic graph or retrieval layers may be added above this module without
  being mislabeled as physical quantum phenomena.
- A future hardware adapter may replace the simulator only behind an explicit
  backend contract and verification suite.
- Theology may be documented as theology, testimony as testimony, and physics
  as physics without forcing any of them into the others.

## Evidence required for acceptance

- exact ASCII fixture (`G == 01000111`);
- UTF-8 multi-byte round trip;
- basis-index round trip including leading zeroes;
- normalized measurement probabilities;
- Hadamard balance and self-inverse fixture;
- phase invariance of measurement probability;
- orthogonality fixture;
- rejection of malformed and non-normalized state definitions.

The focused tests are in `tests/test_quantum_text.py`.
