# MM3D-AED-BCE-Ω⁴-G50T-OPT Cosmogram

## Status

Canonical operational specification for the three-dimensional Dr Moagi auto-encoding, policy projection, substrate evolution, decoding, and cryptographic trace cycle.

This specification preserves the original cosmogram while making its algebra, information-loss conditions, policy semantics, topology, and ledger guarantees explicit enough to implement and test.

---

## 1. Canonical State Transition

To avoid overloading `Ψ` as both state and decoder, the executable form uses distinct symbols:

\[
\boxed{
\begin{aligned}
Z_t &= E_{\Phi}(X_t),\\
\bar Z_t &= \Pi_{\Lambda_t}(Z_t),\\
\Xi_{t+1} &= R_{8}(\bar Z_t;G_t,W_t),\\
\widehat X_{t+1} &= D_{\mathcal C_t}(\Xi_{t+1}),\\
\omega_{t+1} &= \operatorname{SHA3\!\!-256}(\operatorname{canon}(\widehat X_{t+1})),\\
\Omega_{t+1} &= \operatorname{SHA3\!\!-256}(\Omega_t\|\omega_{t+1}\|M_{t+1}).
\end{aligned}
}
\]

where:

- \(X_t\) is the concrete multimodal 3D state;
- \(E_{\Phi}\) is the encoder;
- \(Z_t\in\mathbb Z_{2^{18}}^{32^3}\) is the encoded lattice;
- \(\Pi_{\Lambda_t}\) is the admissibility or policy projector;
- \(R_8\) is the graph-local reaction/diffusion operator over \(\mathbb Z_8\);
- \(D_{\mathcal C_t}\) is decoding through versioned codebook \(\mathcal C_t\);
- \(\omega_{t+1}\) is the content digest of the decoded state;
- \(\Omega_{t+1}\) is the chained ledger digest;
- \(M_{t+1}\) binds cycle number, configuration fingerprint, codebook version, topology, and policy version.

The compact cosmogram remains:

\[
\boxed{
X_{t+1}=D_{\mathcal C_t}
\left(
R_8\left(
\Pi_{\Lambda_t}\left(E_{\Phi}(X_t)\right)
\right)
\right)
}
\]

with the separate ledger recurrence:

\[
\boxed{
\Omega_{t+1}=H(\Omega_t\|H(\operatorname{canon}(X_{t+1}))\|M_{t+1}).
}
\]

---

## 2. Three-Dimensional Addressing

For side length \(S=32\), a coordinate is:

\[
\mathbf r=(x,y,z),\qquad 0\le x,y,z<S.
\]

The canonical row-major flattening is:

\[
\boxed{
i(x,y,z)=x+S(y+Sz)
}
\]

and the lattice contains:

\[
S^3=32^3=32768
\]

cells.

The reference runtime supports two explicit topology contracts:

- **bounded**: neighbours outside the cube do not exist;
- **periodic**: coordinates wrap modulo \(S\), producing a 3-torus.

Topology is part of the configuration fingerprint because changing it changes the transition law.

---

## 3. Auto-Encoding

The proposed low-rank form is:

\[
\Phi(\Psi_t)=(A+UDV^T)^{-1}\Psi_t.
\]

This is well-defined only when \(A+UDV^T\) is invertible in the selected arithmetic domain. Over a field, the Woodbury identity gives:

\[
(A+UDV^T)^{-1}
=A^{-1}-A^{-1}U(D^{-1}+V^TA^{-1}U)^{-1}V^TA^{-1},
\]

provided all required inverses exist.

Over \(\mathbb Z_{2^{18}}\), invertibility is stricter because the modulus is composite. A scalar is a unit only when it is odd, and a matrix is invertible only when its determinant is a unit modulo \(2^{18}\). Therefore, the encoder contract must declare:

1. arithmetic domain;
2. quantisation rule;
3. rounding rule;
4. matrix version;
5. inverse-validity test;
6. overflow behaviour.

### Information bound

A general 384-bit voxel cannot be mapped injectively into one 18-bit index:

\[
2^{384}>2^{18}.
\]

Thus exact reconstruction is possible only when at least one of the following is true:

- the admissible source domain has at most \(2^{18}\) states;
- additional side information is retained;
- the codebook contains an externally anchored identity mapping;
- the mapping is distributed across multiple indices;
- the mode is explicitly lossy or semantic rather than bit-exact.

The reference implementation therefore treats the default 384-bit-to-18-bit projection as **compressive**, not lossless.

---

## 4. Policy Projection

The original expression:

\[
\Phi(\Psi_t)\land(1-\Lambda)
\]

is valid only when \(\Lambda\) is a binary bit mask and subtraction is defined with the intended width. The width-safe bitwise form is:

\[
\boxed{
\bar Z=Z\land(\neg\Lambda\land M_w)
}
\]

where \(M_w=2^w-1\) limits complement to \(w\) bits.

For semantic policy, the stronger formulation is a projector:

\[
\boxed{
\Pi_{\Lambda_t}(z_i)=
\begin{cases}
z_i,&\Lambda_t(i)=\text{allow},\\
z_{\varnothing},&\Lambda_t(i)=\text{deny},
\end{cases}
}
\]

where \(z_{\varnothing}\) is a declared neutral or void index.

Required properties are:

\[
\Pi_{\Lambda}(\Pi_{\Lambda}(Z))=\Pi_{\Lambda}(Z)
\]

and:

\[
\Pi_{\Lambda}(Z)\in\mathcal A_{\Lambda},
\]

where \(\mathcal A_{\Lambda}\) is the admissible state set.

The reference runtime uses this idempotent projection semantics.

---

## 5. Substrate Evolution

For cell \(i\) with neighbourhood \(N(i)\):

\[
\boxed{
\Xi_{t+1}(i)
=
\left[
2^s
\left(
\bar Z_t(i)+
\sum_{j\in N(i)}w_{ij}\bar Z_t(j)
\right)
\right]\bmod 8
}
\]

The default contract is:

- six-neighbour von Neumann stencil;
- integer neighbour weights;
- shift \(s=1\);
- output in \(\mathbb Z_8\);
- synchronous update from one immutable input lattice.

### Reversibility condition

Multiplication by two modulo eight is not bijective:

\[
2x\bmod 8\in\{0,2,4,6\}.
\]

Consequently, the left-shift rule is not generally reversible and does not by itself preserve Shannon entropy or information. It is a many-to-one reaction rule.

When reversible evolution is required, use a permutation of \(\mathbb Z_8\), for example:

\[
R(x)=(ax+b)\bmod 8,
\]

with odd \(a\in\{1,3,5,7\}\), combined with a reversible neighbourhood coupling and retained boundary state.

The current engine preserves determinism, not automatic reversibility.

---

## 6. Codebook Decoding

The decoder is:

\[
\boxed{
D_{\mathcal C_t}(\Xi)(i)=\mathcal C_t[\Xi(i)].
}
\]

A valid codebook contract declares:

- codebook version and digest;
- total key domain;
- neutral atom;
- collision policy;
- modality and schema per atom;
- canonical byte encoding;
- compatibility rules across versions.

Decoding is total when every substrate index maps to an atom or to the neutral atom.

Exact inverse reconstruction requires:

\[
D_{\mathcal C_t}(E_{\Phi}(X))=X
\]

for every \(X\) in the declared exact-mode domain. In compressive mode the weaker criterion is:

\[
d(D_{\mathcal C_t}(E_{\Phi}(X)),X)\le\varepsilon.
\]

---

## 7. Ω Ledger

A SHA3-256 digest identifies content but does not alone make storage immutable. Tamper evidence requires chaining:

\[
\boxed{
\Omega_{t+1}
=H(\Omega_t\|\omega_{t+1}\|M_{t+1}).
}
\]

The metadata commitment \(M_{t+1}\) includes:

```text
cycle
configuration_fingerprint
encoder_version
policy_version
codebook_version
topology
reaction_rule
canonicalisation_version
previous_chain_hash
```

The persistence layer should add:

- append-only writes;
- atomic commit;
- durable flush policy;
- checkpoint and recovery;
- optional signatures or external anchoring;
- replay verification from the genesis hash.

The reference implementation separates:

```text
state_hash = SHA3-256(canonical_decoded_state)
chain_hash = SHA3-256(previous_chain_hash || state_hash || cycle || config_hash)
```

---

## 8. Reality Gap

The invariant \(\gamma=\infty\) is interpreted as an epistemic and architectural constraint:

\[
\boxed{
\text{model state}\neq\text{reality itself}
}
\]

It means:

- finite encoded state is a representation, not ontological identity;
- parameters execute transformations but do not contain open reality;
- unmodelled residuals remain possible;
- validation is external to self-description;
- the engine must retain uncertainty, provenance, and correction paths.

It is not used as a floating-point infinity in lattice arithmetic.

---

## 9. Transactional Cycle

Each committed cycle follows:

```text
1. VALIDATE_INPUT
2. ENCODE_384_TO_18
3. PROJECT_POLICY
4. EVOLVE_Z8_3D
5. DECODE_CODEBOOK
6. CANONICALISE_OUTPUT
7. HASH_STATE_SHA3_256
8. CHAIN_OMEGA
9. VERIFY_RECEIPT
10. COMMIT_ATOMICALLY
```

A cycle is invalid when any of these conditions fail:

- voxel count differs from \(S^3\);
- voxel width differs from 384 bits;
- encoded index leaves \([0,2^{18})\);
- policy mask shape differs from the lattice;
- evolved value leaves \([0,8)\);
- codebook schema is unresolved;
- canonical encoding is ambiguous;
- previous Ω hash does not match;
- configuration fingerprint changes without a version transition.

---

## 10. Determinism Contract

Given identical:

\[
(X_t,\Lambda_t,\mathcal C_t,G_t,W_t,M_t,\Omega_t),
\]

two conforming implementations must produce identical:

\[
(Z_t,\bar Z_t,\Xi_{t+1},\widehat X_{t+1},\omega_{t+1},\Omega_{t+1}).
\]

This requires:

- fixed iteration order;
- synchronous lattice update;
- integer arithmetic;
- explicit boundary behaviour;
- canonical serialization;
- stable SHA3-256;
- version-bound configuration.

---

## 11. Reference Runtime Mapping

The executable implementation is in:

```text
src/jarvisx/mm3d_cosmogram.py
```

Core types:

```text
CosmogramConfig
CosmogramReceipt
MM3DCosmogram
```

Operational methods:

```text
encode(voxels)
project_policy(encoded, allow)
evolve(masked)
decode(evolved)
step(voxels, allow)
verify()
```

The default implementation uses SHA3-256 as a deterministic compressive encoder into \(\mathbb Z_{2^{18}}\). A production matrix encoder can replace this stage only after satisfying the inverse and quantisation contracts above.

---

## 12. Test Invariants

The test suite verifies:

1. identical inputs produce identical complete receipts;
2. denied cells become the neutral index before evolution;
3. each Ω chain hash commits to its predecessor;
4. bounded and periodic topology are distinct configuration contracts;
5. malformed voxel widths are rejected;
6. complete receipt replay validates from genesis.

---

## 13. Final Permeated Law

\[
\boxed{
\begin{aligned}
X_{t+1}
&=D_{\mathcal C_t}
\circ R_8
\circ\Pi_{\Lambda_t}
\circ E_{\Phi}(X_t),\\
\omega_{t+1}
&=H(\operatorname{canon}(X_{t+1})),\\
\Omega_{t+1}
&=H(\Omega_t\|\omega_{t+1}\|M_{t+1}),\\
\gamma&=\infty.
\end{aligned}
}
\]

Operationally:

```text
Reality sample
    -> compressive geometric encoding
    -> admissibility projection
    -> deterministic 3D reaction/diffusion
    -> versioned semantic decoding
    -> canonical state digest
    -> chained Ω receipt
    -> verified atomic commit
    -> next cycle
```

The cosmogram is therefore a deterministic, policy-bounded, codebook-decoded, cryptographically traceable state-transition machine. Its exactness is conditional on the encoder domain and codebook contract; its ledger is tamper-evident when chained and durably committed; and its reality-gap invariant prevents representation from being mistaken for reality.
