# ADR-020: Canonical HyperFoldVME evolutionary lineage and systems boundaries

**Status:** Accepted  
**Date:** 2026-09-19  
**Applies to:** HyperFoldVME, Dr Moagi equations, Jarvis-X multimodal runtimes, sparse 3D engines, geometric bytecode, bitwise VME, state serialization, future CPU/GPU/WebGPU/distributed/hardware backends  
**Extends:** ADR-016, ADR-017, ADR-018 and ADR-019

## Context

The HyperFoldVME / Jarvis-X architecture has evolved across several abstraction layers: continuous field mathematics, recurrent latent dynamics, deterministic state-transition machines, sparse 3D geometry, multimodal autoencoding, candidate-first verification, geometric bytecode, bit-packed state, and portable machine-state serialization.

Without an explicit historical and systems-level decision record, later implementations may accidentally collapse distinctions that the architecture has progressively clarified, especially:

- logical extent vs physical residency;
- symbolic recursion vs physical iteration count;
- modeled equivalent throughput vs measured hardware throughput;
- latent fixed-point self-consistency vs external correctness;
- logical parameter address space vs trained model capability;
- visualization state vs underlying computational state;
- candidate adaptation vs committed authoritative state.

The repository now contains executable reference layers that make these distinctions concrete, including the exact-1B logical-weight engine and the geometric + bitwise VME.

## Decision

Jarvis-X adopts the evolutionary lineage documented in `docs/HYPERFOLDVME_EVOLUTIONARY_HISTORY.md` as the canonical historical interpretation of HyperFoldVME.

The canonical lineage is:

```text
symbolic field
  -> closed dynamical loop
  -> deterministic state machine / VM
  -> 3D spatial computer
  -> autoencoding runtime
  -> multimodal recurrent system
  -> verification-gated adaptation
  -> geometric bytecode
  -> bitwise VME
  -> portable machine-state ABI
```

The canonical operational invariant is:

```text
Encode -> Refine -> Decode -> Verify -> Correct -> Remember -> Recur
```

## 1. Intelligence boundary

The system SHALL treat intelligence as a functional description of observable software behavior, not as a separate substance contained in the machine.

Physical hosts execute ordinary instructions and state transitions. Higher-level behavior may be described as intelligent only insofar as those transitions implement context-sensitive perception, inference, prediction, planning, correction, adaptation or goal-directed control.

## 2. Logical / physical / measured separation

All future HyperFoldVME work SHALL preserve:

```text
logical abstraction != physical implementation != measured performance
```

A declared logical address space SHALL NOT be reported as resident memory unless it is actually allocated.

A modeled throughput or scaling expression SHALL NOT be reported as measured hardware bandwidth.

Measured bandwidth SHALL derive from actual transferred bytes and elapsed time:

\[
B_{measured}=\frac{bytes\ actually\ transferred}{\Delta t}.
\]

## 3. Recursion boundary

Astronomical or symbolic recursion counts MAY be retained as logical metadata, scheduling semantics or mathematical notation.

Physical execution MUST remain bounded by explicit criteria such as:

\[
\|Z^{(k+1)}-Z^{(k)}\|\le\varepsilon
\]

or

\[
k=k_{max}.
\]

Logical recursion depth SHALL NOT be represented as literal physical instruction throughput.

## 4. Fixed-point boundary

A fixed point

\[
Z^\star=F(Z^\star)
\]

establishes self-consistency with respect to the operator (F). It does not by itself establish truth, semantic correctness, physical correctness or external-world validity.

External validation remains a distinct concern.

## 5. Compression and residual boundary

Compression SHALL NOT be assumed lossless from latent geometry alone.

If an encoder discards information required for exact reconstruction, residual or side information MUST be represented explicitly.

The canonical relation is:

\[
R^{(\ell)}
=
H^{(\ell)}
-
D_\ell(E_\ell(H^{(\ell)})).
\]

Compression claims SHALL include residual and metadata costs.

## 6. Candidate-first authority

Adaptive state transitions SHALL preserve the candidate-first pattern established by ADR-016 and ADR-017:

\[
S_{t+1}^{cand}=\mathcal M(S_t,U_t),
\]

followed by validation and explicit promotion or rollback.

Unverified candidate state MUST NOT silently become authoritative state.

## 7. Multimodal boundary

A runtime MAY be described as multimodal when modality-specific source spaces are explicitly mapped into a shared representational space, for example:

\[
H_t
=
\Phi_{fusion}
(E_1(X_1),\ldots,E_M(X_M)).
\]

Independent visual/audio/text outputs without a shared computational representation are not sufficient by themselves to establish multimodal fusion.

## 8. Visualization boundary

The preferred visualization rule is:

```text
animation variable == computational variable
```

where practical.

Visualization SHOULD expose runtime state such as latent activation, residual magnitude, memory persistence, fixed-point residual, active support or scheduler policy.

The renderer remains an observation/projection layer and SHALL NOT be conflated with the high-dimensional state it visualizes.

## 9. Sparse virtual geometry

Large declared manifolds SHALL be treated as logical spaces unless dense allocation is explicitly measured and demonstrated.

Implementations SHOULD use sparse active sets, tiles, pages, octrees, Morton/Z-order addressing, memory mapping or equivalent techniques when logical scale exceeds the physical budget.

## 10. Bitwise lowering

Geometric and ANN state MAY be lowered into fixed-width integer forms.

The current reference profile includes:

```text
64-bit voxel/control word:
[x:12 | y:12 | z:12 | value:4 | modality:2 | residual:8 | flags:14]

64-bit ANN/VME instruction:
[opcode:8 | dst:8 | srcA:8 | srcB:8 | immediate:32]
```

A (4000^3) logical coordinate space fits in 12 bits per axis.

The existence of this bit-level encoding does not imply custom physical hardware; it is a software ISA/reference representation until separately implemented on hardware.

## 11. Portable state ABI direction

HyperFoldVME SHALL treat state serialization as a transactional lifecycle:

```text
Read -> Validate -> Decode candidate -> Verify -> Commit
```

A portable machine-state ABI SHOULD explicitly encode logical and resident geometry separately and SHOULD include integrity verification before live state mutation.

The target semantic property is:

\[
\mathcal V_t
\xrightarrow{serialize}
B_t
\xrightarrow{transport}
B_t'
\xrightarrow{verify/restore}
\mathcal V_t'
\]

with equality required for all authoritative serialized components when exact migration is claimed.

## 12. Historical convergence

The canonical interpretation of the project's development is progressive concretization:

\[
\boxed{
\text{metaphor}
\rightarrow
\text{mathematical object}
\rightarrow
\text{algorithm}
\rightarrow
\text{data structure}
\rightarrow
\text{instruction}
\rightarrow
\text{bit}
}
\]

The contemporary HyperFoldVME is the convergence of:

\[
\boxed{
\text{Dr Moagi mathematics}
+
\text{Jarvis-X verification/control}
+
\text{3D spatial computation}
+
\text{autoencoding/latent recurrence}
}
\]

## Consequences

### Positive

- Future architectures inherit explicit claim and measurement boundaries.
- New backends can change physical representation without changing logical semantics.
- Large-scale conceptual models can remain useful without implying impossible dense allocation or throughput.
- Verification, rollback and state migration remain first-class architectural properties.
- The project history becomes inspectable as an engineering progression rather than a collection of disconnected formulations.

### Costs

- Implementations must report more metadata: logical extent, resident extent, precision, iteration budgets, convergence status and measured performance.
- Some compact conceptual claims require more careful qualification.
- Cross-runtime serialization requires versioning and compatibility discipline.

## Canonical references

- `docs/HYPERFOLDVME_EVOLUTIONARY_HISTORY.md`
- ADR-016 — canonical typed state, transactions and geometry profiles
- ADR-017 — Dr Moagi operational autoencoding equation
- ADR-018 — sparse speculative 3D self-optimizing runtime
- ADR-019 — 1000 GB 3D Cloud ROM engine
- PR #284 — exact-1B multimodal 3D intelligence engine
- PR #288 — geometric + bitwise bytecode ANN / VME
