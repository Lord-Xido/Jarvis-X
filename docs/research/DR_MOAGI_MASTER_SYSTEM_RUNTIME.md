# DM–vOmegaXi+ Master System Runtime

**Status:** proposed executable integration contract  
**Reference:** `src/jarvisx/dr_moagi_master_runtime.py`  
**Decision:** `docs/adr/0012-dr-moagi-master-transaction-runtime.md`

## 1. Systems interpretation

The Dr Moagi stack is intentionally two coupled machines separated by a transaction membrane:

```text
ADAPTIVE / RESEARCH MACHINE
observation -> encode -> recurse -> predict -> project -> decode hypothesis

AUTHORITY MACHINE
evidence -> admission -> lowering -> capability projection -> VM -> audit -> commit
```

The research machine searches state space. The authority machine decides what may become committed system state.

The defining invariant is:

```text
candidate != truth != authority
```

A candidate can be numerically bounded and still false. A candidate can be epistemically supported and still lack execution authority. A deterministic program can execute correctly while representing a false upstream premise. The master runtime therefore requires both kinds of admission.

## 2. Typed system state

The system-wide logical state is decomposed as:

```text
O : externally sourced observations/evidence
H : generated hypotheses
Xi/Z : research/latent state
Theta : model/runtime parameters
Omega : bounded working/statistical memory
A : authoritative VM/task state
Phi : spatial permeation/output field
rho : audit-linked execution receipt
```

These are distinct state classes. No operator may combine them additively without an explicit type-preserving map.

In particular:

```text
grad_Z L      belongs to the Xi/Z update
grad_Theta L  belongs to the Theta update
```

and any parameter-to-state influence requires an explicit transport map.

## 3. Master transition

The outer transition is:

```text
H_(t+1) = G_DM(O_t)
V_epi = Verify(H_(t+1), O_t, A_0, E_1...E_n)
```

If `V_epi` rejects, the transaction ends in quarantine.

If admitted, an explicit lowering adapter produces bounded candidate programs:

```text
P_1:m = PlanBuilder(H_(t+1))
```

The authority runtime then applies:

```text
plans
 -> bounded deterministic selection
 -> capability projection
 -> isolated canonical VM execution
 -> VM ledger verification
 -> final-state digest
 -> system audit append/verify
 -> ExecutionReceipt rho_t
```

The outer commit predicate is:

```text
Commit_DM = V_epi.admitted AND rho_t.committed
```

Only under `Commit_DM` does the master runtime expose the admitted scene, candidate parameter update, source field and permeation field as authoritative outputs.

## 4. Compact DM–vOmegaXi+ law

A compact system equation is:

\[
H_{t+1}
=
D_\Theta\left(
\Pi_\Lambda\left[
S_{\Delta t}
\left(
R^{\circlearrowleft}(E(O_t))
+P_t
-\eta_Z\nabla_Z L_t
\right)
\right]
\right),
\]

followed by the non-additive trust operators

\[
V_t^{epi}
=
A_{epi}(H_{t+1};O_t,A_0,E_{1:n}),
\]

\[
\rho_t
=
A_{sys}(C_{H\rightarrow VM}(H_{t+1})),
\]

and

\[
S_{t+1}
=
\begin{cases}
Commit(H_{t+1},\Theta_{t+1}^{cand},\Phi_{t+1},\rho_t),
& V_t^{epi}=ADMIT\land\rho_t.committed,\\
S_t,&\text{otherwise.}
\end{cases}
\]

The parameter candidate is separately typed:

\[
\Theta_{t+1}^{cand}
=
\Theta_t-\eta_\Theta\nabla_\Theta L_t.
\]

## 5. Pi_Lambda hierarchy

`Pi_Lambda` should be interpreted as a family of projections rather than one scalar clamp:

```text
Pi_Lambda =
  Pi_numeric
  o Pi_type
  o Pi_topology
  o Pi_resource
  o Pi_epistemic
  o Pi_capability
  o Pi_security
```

Not every projection occurs in one function. The invariant is that every authority transition passes the applicable membranes before publication.

## 6. Memory separation

The overloaded symbol `Omega` should be treated as a namespace of distinct memory classes:

```text
Omega_working     : candidate-generation residual/history state
Omega_statistical : codec/model/runtime statistics
Omega_episodic    : bounded task-history representation
Omega_provenance  : append-only execution/audit history
```

Mutable learning memory must never be equivalent to immutable provenance:

```text
Omega_learning != Omega_ledger
```

The current canonical ledger remains the authority-side provenance mechanism.

## 7. Kinetic interpretation

The verified geometric runtimes support a discrete, dissipative interpretation:

```text
candidate motion -> contraction/refinement -> projection -> verification -> commit/rollback
```

This is a bounded computational kinetic system. The current Helmholtz/Green field is a spatial propagation operator; it is not a validated electromagnetic field or time-domain physical wave.

A future causal 4D field extension would require an explicit law such as:

\[
\partial_t^2\Phi
+2\alpha\partial_t\Phi
-c_\Phi^2\nabla^2\Phi
=Q[\Xi_t],
\]

with measured timestep stability, propagation velocity, dispersion, damping, boundary conditions and an explicit energy functional.

## 8. Failure topology

The outer runtime must fail closed on:

```text
unverified hypothesis
missing independent evidence
provenance-invalid evidence
empty or invalid lowering result
capability rejection
resource-budget rejection
bytecode failure
VM ledger failure
system audit failure
request-id collision
non-finite or malformed state
```

A failure after epistemic admission does not authorize the candidate parameter update or permeation output.

## 9. Provenance limitation

Software labels such as `kind=sensor` are contracts, not cryptographic proof of origin. Production provenance should bind evidence to authenticated adapters using mechanisms such as:

```text
signed source identity
mTLS adapter identity
content digest + source signature
trusted timestamp
attested runtime identity
append-only provenance ledger
```

This is the main remaining epistemic security boundary.

## 10. Durability limitation

`SystemRuntime` currently provides deterministic in-process transaction semantics and audit linkage. Production authority requires a durable state adapter with crash recovery, replay/idempotency persistence, transaction isolation and explicit external side-effect coordination.

## 11. Canonical operational loop

```text
OBSERVE EXTERNAL REALITY
 -> AUTHENTICATE / CLASSIFY EVIDENCE
 -> ENCODE
 -> RECURSE / PREDICT / REFINE
 -> Pi_Lambda RESEARCH PROJECTION
 -> DECODE HYPOTHESIS
 -> EPISTEMIC VERIFY
      reject -> quarantine
      admit  -> verified research candidate
 -> EXPLICIT LOWERING TO PlanCandidate
 -> CAPABILITY + RESOURCE PROJECTION
 -> ISOLATED VM EXECUTION
 -> LEDGER VERIFY
 -> SYSTEM AUDIT
      reject/fail -> authority rollback
      commit      -> authoritative receipt
 -> COMMIT THETA / Q / PHI / OUTPUT AUTHORITY
 -> RE-OBSERVE
```

The resulting closed loop is:

\[
\boxed{
Reality
\rightarrow
Description
\rightarrow
Hypothesis
\rightarrow
Verification
\rightarrow
Authorization
\rightarrow
Execution
\rightarrow
Audit
\rightarrow
Commit
\rightarrow
Reality'
}
\]

This document is the system-wide interpretation of DM–vOmegaXi+; it does not claim AGI, consciousness, physical electromagnetic computation, unbounded recursion throughput or production security from the existence of the notation alone.
