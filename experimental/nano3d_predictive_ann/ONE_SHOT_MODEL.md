# One-shot geometric state, cache and verification model

The purpose is to demonstrate a **closed-loop, bounded operational mechanism**
without conflating synthetic classification, Riemannian convergence, or cached
source reuse with general software correctness.

## 1. Explicit Riemannian iteration on S²

Let z,t in S², f(z)=1 - <z,t>, and grad f(z)=-t+<z,t>z.
At step k, with P_z(v)=v-z<z,v>:

- g_k=grad f(z_k)
- m_k=beta_1*m_{k-1}^{transported}+(1-beta_1)*g_k
- v_k=beta_2*v_{k-1}+(1-beta_2)*||g_k||² (a scalar, hence coordinate-free)
- d_k= -eta_k*[m_k/(1-beta_1^k)] / (sqrt(v_k/(1-beta_2^k)) + eps)
- z_{k+1}=Exp_{z_k}(P_{z_k}(d_k)),
  Exp_z(u)=cos(||u||)z+sin(||u||)u/||u||
- m_k^{transported to z_{k+1}}=m_k - ((m_k · z_{k+1})/(1+z_k · z_{k+1}))*(z_k+z_{k+1})

The implementation normalizes the numerically computed exponential map and
projects momentum tangent to the new state. The objective is synthetic.
Exactly K iterations occur, including when gradients are already small.
The code **rejects** failed norm or convergence checks. No general geodesic
convergence theorem follows from this demonstration.

## 2. Predictor

Existing Nano3D learns synthetic 3D octant classes, precomputes 32³ argmax
predictions into a 64 KiB Morton-indexed table, then emits one-line unsigned
C++ functions. Every output's expected opcode is reconstructed independently
in Python from deterministic task IDs using an independent xorshift32 and
unsigned-32 oracle. Compilation and four runtime tests per function are required.

## 3. Precompute/reuse decision

The expected net gain is a user-supplied scenario, **not a measured saving**:

G = n*[p_valid*(c_fresh-c_lookup)-(1-p_valid)*c_invalidate]-c_precompute.

Use cache only if G>0 and the context/compiler/source/test schema fingerprint
matches, and the stored file hashes match. All accepted files are recompiled
and re-tested on reuse. Source hashes are receipts, not cryptographic signatures.
The Riemannian optimizer's synthetic state is verified and recorded alongside
each release, but **does not select or prove correct C++ templates**.

## 4. State-admission protocol

candidate = (optimized geometry audit, source, contextual provenance)
V(candidate) = [geometric numerical checks] AND [independent template oracle]
   AND [all translation units syntax-accepted] AND [all runtime tests passed].

On V=1, stage->immutable release, then atomically replace latest.json;
on V=0, remove stage and preserve latest. Publication is not a distributed
transaction, power-loss durable journal, hostile-input sandbox, or a proof of
arbitrary C++ correctness.

## 5. Rates

Verified admission rate for run with N emitted source lines, T wall seconds:
R_accepted=N/T. Local benchmark figures apply **only to the synthetic workload**.
The one-shot reference deliberately includes compilation and revalidation, and
must not represent its table-throughput as end-to-end throughput.
