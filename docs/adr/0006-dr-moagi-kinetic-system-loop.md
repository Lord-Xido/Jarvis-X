# ADR-006: Adopt the Dr Moagi kinetic system loop

**Status:** Accepted  
**Date:** 2026-08-15  
**Extends:** ADR-002, ADR-003, ADR-004, ADR-005

## Context

Jarvis-X already separates the deterministic authority substrate from adaptive codec, sparse-field, multimodal and precision-verification research layers. The next requirement is to define how geometry, neural activation, physics, swarm propagation, learning and bounded topology changes evolve together without allowing worker threads or adaptive kernels to mutate authoritative state in place.

Earlier swarm prototypes expressed these concerns as mutable per-neuron functions. That model is insufficient for a production runtime because structural mutation can invalidate storage, index-addressed synapses become stale after compaction, graph propagation can be accidentally rerouted by global arbitration, recursive echo can create unbounded event storms, and rejected topology changes can leak hidden state such as identifier allocation.

## Decision

Jarvis-X adopts a three-level kinetic model:

```text
Xi(t)     state kinetics
Theta(t)  parameter / learning kinetics
G(t)      topology kinetics
```

with the master interpretation

```text
dXi/dt = F_external
       + F_neural
       + F_physical
       + F_swarm
       + F_decode
       - grad_Xi L

dTheta/dt = -eta * grad_Theta L + M_Theta
dG/dt = T_spawn - T_prune + T_rewire + T_merge
```

These equations are operational only through a bounded discrete transaction:

```text
committed state Xi_t
-> immutable snapshot
-> route operations
-> compute local deltas / forces
-> bounded graph propagation
-> integrate candidate state
-> stage topology mutations
-> Pi_Lambda verification
-> atomic COMMIT or complete ROLLBACK
-> Xi_(t+1)
```

No worker, operation executor, renderer, learner or topology operator may structurally mutate the committed node store during the compute phase.

## Stable identity and routing

Nodes use immutable stable IDs. Synapses reference IDs rather than vector indices or storage addresses.

Routing modes are explicit:

```text
DIRECT  -> exact stable node ID
GLOBAL  -> every node in the frozen snapshot
GRAPH   -> explicit synaptic target IDs during propagation
```

Graph propagation may not silently fall back to an unrelated global best-neuron selector.

## Kinetic integration

For node `i`, the mechanical state follows

```text
v_(i,t+1) = clamp_v(gamma_v * (v_i,t + dt * F_i,t))
x_(i,t+1) = x_i,t + clamp_dx(dt * v_(i,t+1))
```

where the force may contain bounded physical, decoded, swarm and corrective components.

Neural state follows the same transactional rule:

```text
p_(i,t+1) = decay_i * p_i,t + I_i,t
a_(i,t+1) = sigmoid(k * (p_(i,t+1) - threshold_i))
```

Learning changes parameters only in the candidate copy. Topology changes are staged and applied only after state verification.

## Echo stability and backpressure

Propagated events carry an explicit TTL and amplitude. A propagation terminates when either

```text
ttl == 0
```

or

```text
abs(amplitude) < echo_epsilon.
```

Each step also has a hard event budget. A candidate exceeding that budget fails closed. For a linearized echo operator the desired stability condition remains

```text
rho(gamma_echo * W) < 1,
```

but runtime event bounds remain mandatory even when a spectral estimate is unavailable.

## Codec relationship

ADR-006 does not replace the same-space codec contract of ADR-003. Production kinetic encode/decode operators must bind to the declared codec closure

```text
A_theta = D_theta o E_theta
R_theta(Psi) = Psi - A_theta(Psi)
```

and preserve the immutable anchor and rate/distortion telemetry defined by ADR-002/003.

The C++ kinetic reference uses a deliberately small local encode/decode surrogate to exercise state-transition semantics. It is a scheduler/kinetics reference, not a claim that this surrogate is the canonical 3D codec.

## Concurrency contract

Permitted parallelism is:

```text
snapshot read
-> shard-local delta computation in parallel
-> deterministic reduction / barrier
-> candidate verification
-> single logical commit
```

Prohibited parallelism includes structural `push_back`, erasure, compaction, ID allocation, graph rewiring or authoritative queue mutation while another worker holds references into the committed node store.

## Topology contract

Topology operators are proposals, not immediate mutations.

```text
SPAWN(parent, count)
PRUNE(node, threshold)
REWIRE(...)
MERGE(...)
```

A spawn count is represented explicitly and must create exactly that many children when admitted. ID allocation belongs to the transaction and rolls back when the candidate is rejected.

## Required invariants

1. One cycle reads one immutable committed snapshot.
2. Stable node IDs survive storage relocation and compaction.
3. Graph edges reference stable IDs, never transient vector positions.
4. DIRECT, GLOBAL and GRAPH routing semantics are explicit and testable.
5. Echo propagation is bounded by TTL, amplitude cutoff and per-step event budget.
6. Velocity, displacement, coordinate, node-count and event-count ceilings are explicit.
7. Structural mutations occur only at the commit barrier.
8. A failed candidate leaves state, topology and ID allocation unchanged.
9. Learning and optimization operate on measured residuals and candidate parameters.
10. Measured execution is reported separately from simulated or logical scale.
11. Rendering remains a projection unless separately admitted as authoritative input.
12. The deterministic VM/policy/transaction boundary remains authoritative.

## Reference implementation

The bounded C++17 reference is:

```text
cpp_runtime/include/jarvisx/kinetic_system.hpp
cpp_runtime/src/kinetic_system_main.cpp
cpp_runtime/tests/kinetic_system_tests.cpp
```

The normative mathematical and operational specification is:

```text
docs/DR_MOAGI_KINETIC_SYSTEM_LOOP.md
```

## Validation

Conformance requires tests demonstrating:

- frozen-snapshot mechanical integration;
- direct and global routing;
- bounded echo propagation;
- exact transactional spawn counts with unique stable IDs;
- synapse pruning without index corruption;
- failed projection rollback;
- finite residual/learning updates;
- warning-clean C++17 compilation;
- a bounded runtime smoke cycle.

## Consequences

### Positive

- the ANN/swarm/physics loop gains one coherent state-transition semantics;
- concurrency hazards are moved behind an explicit snapshot/commit boundary;
- topology evolution becomes reversible and auditable;
- graph propagation and global scheduling are no longer conflated;
- the kinetic abstraction can be lowered to CPU, SIMD, GPU or distributed backends without changing authority semantics.

### Negative

- topology mutation becomes at least one barrier behind proposal generation;
- stable-ID lookup adds indexing overhead;
- deterministic reference execution prioritizes correctness over peak throughput;
- production codec integration still requires explicit binding to the existing 3D codec runtime rather than the local surrogate.
