# Dr Moagi Kinetic System Loop

## Status

Canonical bounded kinetic extension for the Jarvis-X Layer 4/5 research runtime, adopted by ADR-006.

This specification turns the unified ANN/swarm/physics/codec loop into a transactional state-transition system. It preserves the authority, sparse-state, anchor, codec, precision and rollback boundaries established by ADR-002 through ADR-005.

---

## 1. Three coupled kinetics

Let the complete runtime be

\[
\mathcal K_t=(\Xi_t,\Theta_t,G_t,Q_t,\Omega_t),
\]

where:

- \(\Xi_t\): committed geometric, neural, physical and latent state;
- \(\Theta_t\): learnable/runtime parameters;
- \(G_t=(V_t,E_t)\): stable-ID graph topology;
- \(Q_t\): admitted operation/event stream;
- \(\Omega_t\): telemetry, anchors, journals and persistent optimization memory.

The conceptual continuous laws are

\[
\boxed{
\frac{d\Xi}{dt}
=
\mathcal F_{ext}
+\mathcal F_{neural}
+\mathcal F_{physical}
+\mathcal F_{swarm}
+\mathcal F_{decode}
-\nabla_{\Xi}\mathcal L
}
\]

\[
\boxed{
\frac{d\Theta}{dt}
=-\eta\nabla_{\Theta}\mathcal L+\mathcal M_{\Theta}
}
\]

and

\[
\boxed{
\frac{dG}{dt}
=\mathcal T_{spawn}-\mathcal T_{prune}
+\mathcal T_{rewire}+\mathcal T_{merge}.
}
\]

The runtime never integrates these expressions by mutating committed state in place. They define candidate deltas inside a transaction.

---

## 2. Per-node state

For stable node ID \(i\):

\[
\xi_i=
(x_i,v_i,n_i,c_i,s_i,a_i,p_i,\tau_i,e_i,f_i,\varepsilon_i,W_i),
\]

where:

```text
x_i           3D position
v_i           velocity
n_i           normalized local direction
c_i           curvature descriptor
s_i           spectral weight
a_i           activation
p_i           potential
tau_i         activation threshold
e_i           energy / resource state
f_i           fitness or utility descriptor
epsilon_i     reconstruction / prediction residual
W_i           outgoing stable-ID synapses
```

Storage position is not identity. `i` remains stable when the physical container is relocated or compacted.

---

## 3. Operational cycle

One cycle is:

```text
1  acquire admitted operations Q_t
2  freeze snapshot S_t = (Xi_t, Theta_t, G_t)
3  resolve DIRECT/GLOBAL targets against stable IDs
4  compute operation-local forces and parameter deltas
5  propagate GRAPH events with TTL/amplitude/event-budget bounds
6  integrate node candidate state from S_t + deltas
7  stage topology proposals
8  run Pi_Lambda over state + topology + resource budgets
9  COMMIT candidate atomically, including ID allocator state
10 emit telemetry / receipt / rejection reason
11 repeat from the new committed snapshot
```

The essential invariant is

\[
\boxed{
S_t\;\text{is read-only during steps 3--7.}
}
\]

---

## 4. Routing semantics

Operations carry an explicit scope.

### Direct

\[
R_{direct}(o)=\{\operatorname{id}(o)\}.
\]

Failure to resolve the stable ID rejects the operation/candidate according to policy.

### Global

\[
R_{global}(o)=V(S_t).
\]

A global `PHYSICS`, `AI`, `LEARN` or `RENDER` request is therefore not silently reduced to one best-scoring node.

### Graph

For a propagated event from node \(i\):

\[
R_{graph}(i)=\{j:(i,j)\in E_t\}.
\]

The synapse target is authoritative for that hop.

---

## 5. Mechanical kinetics

For unit mass in the reference model,

\[
F_i
=F_i^{physical}
+F_i^{spring}
+F_i^{swarm}
+F_i^{decode}
+F_i^{corrective}.
\]

Velocity and position are integrated as

\[
\tilde v_{i,t+1}
=\gamma_v\left(v_{i,t}+\Delta t F_{i,t}\right),
\]

\[
v_{i,t+1}=\Pi_{v_{max}}(\tilde v_{i,t+1}),
\]

\[
\Delta x_i
=\Pi_{\Delta x_{max}}(\Delta t\,v_{i,t+1}),
\]

\[
x_{i,t+1}=x_{i,t}+\Delta x_i.
\]

The projection prevents one unstable force from producing an unbounded frame displacement.

A spring edge may use

\[
F_{ij}^{spring}
=k_{ij}(\|x_j-x_i\|-L_{ij})\hat r_{ij}.
\]

A bounded swarm cohesion term is

\[
F_i^{cohesion}=k_c(\bar x_{\mathcal N(i)}-x_i).
\]

Production variants may add alignment and separation so long as each contribution remains bounded and transactionally verified.

---

## 6. Neural kinetics

Potential evolves as

\[
p_{i,t+1}
=d_i p_{i,t}+I_{i,t}
\]

and activation as

\[
a_{i,t+1}
=\sigma\left(k(p_{i,t+1}-\tau_i)\right).
\]

A graph inference predictor may be

\[
\hat a_i
=\frac{\sum_j w_{ij}a_j}
{\sum_j|w_{ij}|+\epsilon},
\]

with prediction residual

\[
r_i=a_i-\hat a_i.
\]

The residual may contribute a candidate potential correction and telemetry. It is not permitted to overwrite the immutable run anchor used by the codec/field runtime.

---

## 7. Encode/decode closure

The production volumetric codec relationship remains ADR-003:

\[
A_\theta=D_\theta\circ E_\theta,
\qquad
R_\theta(\Psi)=\Psi-A_\theta(\Psi).
\]

Kinetic correction may use the same-space residual as a force or parameter signal:

\[
F^{corrective}
=K_R\,\mathcal L_{lift}(R_\theta(\Psi)),
\]

where `L_lift` is an explicitly typed mapping from the field residual into the local kinetic state. No latent tensor may be subtracted directly from incompatible geometry.

The C++ reference `ENCODE` operator implements only a bounded local surrogate so that routing, residual, learning and rollback mechanics can be tested without introducing a second codec implementation. It must not be reported as the canonical 3D encoder.

---

## 8. Echo as bounded wave propagation

For hop \(h\):

\[
e_j^{h+1}=\gamma_e w_{ij}e_i^h.
\]

The linearized field is

\[
\mathbf e_{h+1}=\gamma_e W\mathbf e_h.
\]

A desirable stability condition is

\[
\rho(\gamma_eW)<1.
\]

Runtime safety does not rely on that condition alone. Every event also carries:

```text
stable target ID
amplitude
TTL
```

and propagation terminates on

```text
TTL == 0
or abs(amplitude) < echo_epsilon
or events_this_step >= max_events_per_step.
```

Exceeding the hard event budget rejects the candidate rather than allowing queue explosion.

---

## 9. Learning kinetics

A bounded Hebbian candidate update may be

\[
\Delta w_{ij}
=\eta a_i a_j-\eta\lambda_w w_{ij}.
\]

The updated weight is projected into its admissible interval before commit.

True STDP requires explicit spike times:

\[
\Delta w=
\begin{cases}
A_+e^{-\Delta t/\tau_+},&\Delta t>0,\\
-A_-e^{\Delta t/\tau_-},&\Delta t<0.
\end{cases}
\]

An implementation may not label a purely activation-correlated rule as STDP unless relative spike timing is actually represented.

---

## 10. Optimization kinetics

Let the measurable objective be

\[
\mathcal L_t
=w_R D_{recon}
+w_A D_{anchor}
+w_L L
+w_M M
+w_E E_{compute}
-w_U U_{verified}.
\]

Parameter updates are candidate transformations:

\[
\Theta'=
\Pi_{\Theta}
\left(\Theta-\eta\nabla_{\Theta}\mathcal L\right).
\]

A heuristic adaptation that does not evaluate a gradient must be reported as such rather than described as gradient descent.

Mechanics-level candidate optimization continues to inherit the shadow/canary/rollback contract of the inward self-optimizing runtime.

---

## 11. Topology kinetics

Topology changes are staged after node-state computation.

### Spawn

```text
SPAWN(parent_id, count)
```

means exactly `count` candidate children, subject to `max_nodes`.

For child \(k\):

\[
x_k=x_{parent}+\delta x_k,
\qquad
v_k=0.
\]

The stable-ID allocator belongs to candidate state. If the transaction fails, the allocator is restored and IDs are not consumed invisibly.

### Prune

A reference threshold operator may remove

\[
|w_{ij}|<\epsilon_w.
\]

A production system should prefer measured synaptic flux or utility when available:

\[
\Phi_{ij}=\operatorname{EMA}(|w_{ij}a_i|).
\]

### Merge and rewire

Merge/rewire operators must emit explicit topology proposals and stable-ID remapping receipts. They may not compact storage first and repair edges later.

---

## 12. Pi_Lambda kinetic projection

Before commit, the candidate must satisfy at least:

```text
all scalar/vector values finite
unique non-zero stable IDs
all synapse targets resolve
abs(position_axis) <= max_abs_position
speed <= max_speed
displacement_per_step <= max_displacement
node_count <= max_nodes
events_processed <= max_events_per_step
weights inside declared bounds
codec/model versions coherent where codec operations participate
policy/authority checks pass
```

The transaction is

\[
\Xi_t
\xrightarrow{propose}
\widetilde\Xi_{t+1}
\xrightarrow{\Pi_\Lambda}
\begin{cases}
\Xi_{t+1}=\widetilde\Xi_{t+1},&accept,\\
\Xi_{t+1}=\Xi_t,&reject.
\end{cases}
\]

Topology and allocator state are included in \(\widetilde\Xi\).

---

## 13. Concurrency lowering

The deterministic reference is serial. A parallel backend may lower it to:

```text
immutable snapshot
  -> shard 0 delta kernel --\
  -> shard 1 delta kernel ----> deterministic reduction/barrier
  -> ...                    --/
  -> topology proposal aggregation
  -> Pi_Lambda
  -> atomic commit
```

Worker-owned mutation is restricted to private deltas/candidate shards. Workers may not perform structural mutation on the committed `std::vector`, adjacency store, sparse map or authoritative queue while other workers retain references.

A GPU backend should therefore use the same broad split:

```text
CPU / authority plane:
  routing, budgets, versions, commit, journal

GPU / data plane:
  force kernels, ANN propagation, codec kernels, spatial reductions
```

---

## 14. Telemetry

Every kinetic step reports at least:

```text
cycle
nodes_before
nodes_after
operations_consumed
propagated_events
spawned_nodes
pruned_synapses
mean_activation
mean_residual
max_speed
max_displacement
committed
rejection_reason
```

Production integrations additionally preserve the codec/field telemetry:

```text
reconstruction_mse
anchor_mse
rate / coded bytes
model and mechanics versions
resource usage
measured latency / throughput
```

Logical node count, virtual depth and simulated cores are always reported separately from measured resident memory and physical execution throughput.

---

## 15. Reference C++17 implementation

Files:

```text
cpp_runtime/include/jarvisx/kinetic_system.hpp
cpp_runtime/src/kinetic_system_main.cpp
cpp_runtime/tests/kinetic_system_tests.cpp
```

The reference deliberately optimizes for deterministic semantics and auditability. Its operation set currently exercises:

```text
ENCODE
DECODE
PHYSICS
AI
ECHO
LEARN
SWARM
OPTIMIZE
SPAWN
PRUNE
```

`MERGE`, explicit STDP and production `Autoencoder3D` coupling remain later extensions and must preserve the transaction contract when added.

---

## 16. Conformance tests

A conforming implementation demonstrates:

1. physics/swarm forces use one frozen source snapshot;
2. graph edges use stable IDs;
3. direct routing reaches the requested stable ID;
4. global routing visits the full frozen node set;
5. echo stops under TTL/amplitude/event limits;
6. spawn count is exact and unique IDs are committed transactionally;
7. pruning does not invalidate unrelated edges;
8. an out-of-bounds candidate is rejected with authoritative state unchanged;
9. parameter learning remains finite and bounded;
10. the runtime compiles warning-clean under the repository C++17 profile;
11. a smoke workload completes multiple committed cycles.

---

## 17. Canonical operational interpretation

Jarvis-X is not defined by the statement "everything is a neuron." The stronger abstraction is:

\[
\boxed{
\text{everything admitted becomes a bounded operation on a common transactional state substrate.}
}
\]

Geometry is state. Neural activity is state. Physics is a state transition. Encoding is contraction into a representation. Decoding is a typed outward reconstruction. Error is a corrective signal. Learning changes the transition parameters. Evolution changes topology. Rendering is a projection unless separately promoted. The commit boundary decides what becomes authoritative.

The resulting loop is

```text
OBSERVE
-> ENCODE
-> ROUTE
-> PROPAGATE / COMPUTE
-> INTEGRATE
-> DECODE
-> MEASURE RESIDUAL
-> LEARN / OPTIMIZE
-> STAGE TOPOLOGY
-> VERIFY
-> COMMIT
-> inward re-entry
```

and every arrow is bounded by explicit state types, resource ceilings and rollback semantics.
