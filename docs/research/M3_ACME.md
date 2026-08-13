# Moagi 3D AutoCodec Manifold Equation (M³-ACME)

## Status

Executable research specification for the Jarvis-X M³-ACME adapter. Governed by ADR-006 and bounded by ADR-002.

## 1. Primary equation

The operational form is

```text
X_hat = D_Phi(A(E_Theta(X ⊙ C)))
```

with a runtime transaction:

```text
observed records X
-> validate explicit compliance/provenance assertions C
-> canonicalize admitted records X_c
-> encode/compress E_Theta
-> integrity-bound latent packets Z
-> source-grounded abstraction A(Z) = Z*
-> decode D_Phi(Z*) / latent payload
-> structured reconstruction X_hat
-> component loss + receipt
```

The current reference implementation uses deterministic canonical JSON + zlib for `E_Theta` / `D_Phi`. That choice provides an executable round-trip baseline; it is not a claim that zlib is a neural autoencoder.

## 2. Field coordinates

Each admitted record maps the conceptual axes as follows:

| Coordinate | Runtime representation |
|---|---|
| `x` structure | DOM/layout/structural metadata supplied by the caller |
| `y` semantics | entities, relations, topics, sentiment, confidence supplied by the caller |
| `z` provenance | source URL, trust score, permission basis |
| `t` time | timezone-qualified observation timestamp |

`x,y,z` are conceptual manifold coordinates rather than Euclidean pixel positions in this adapter. A future tensor backend may embed them into explicit numeric coordinates, but that embedding must be versioned and independently validated.

## 3. Compliance operator C

The reference gate requires explicit caller assertions for:

```text
authorized == true
robots_permitted == true
terms_permitted == true
restricted_data == false
permission_basis != empty
valid provenance URL
HTTPS provenance by default
0 <= trust <= 1
timezone-qualified observed_at
```

This is a processing gate, not a legal oracle. The runtime does not crawl external sources and does not infer whether an assertion is legally sufficient.

## 4. Encoding E_Theta

For an admitted record `r`, define canonical bytes

```text
b = CanonicalJSON(r)
```

and latent packet

```text
Z = {
  codec_version,
  SHA256(b),
  zlib(b),
  provenance metadata,
  source-provided semantic projection
}.
```

The deterministic invariant is

```text
same admitted record + same codec version -> same b, digest and compressed bytes.
```

## 5. Abstraction A

The reference abstraction operator is conservative:

```text
A(Z) -> {
  entities,
  relations,
  topics,
  sentiment,
  confidence,
  provenance,
  observed_at,
  trust
}
```

Only values already present in the admitted record are projected. Missing semantic data remains missing rather than being guessed.

## 6. Decoding D_Phi

Decode is permitted only when:

```text
packet.codec_version == runtime.codec_version
zlib payload decompresses successfully
uncompressed size <= configured ceiling
SHA256(uncompressed) == packet.digest
uncompressed bytes decode to a JSON object.
```

Then

```text
X_hat = JSONDecode(uncompressed bytes).
```

For the deterministic reference codec, an admitted canonical record has exact round-trip reconstruction.

## 7. Composite loss

The runtime reports

```text
L_Moagi = L_reconstruction
        + lambda_s L_semantic
        + lambda_p L_provenance
        + lambda_t L_temporal
        + lambda_c L_compliance
        + lambda_g L_graph.
```

Reference semantics:

- `L_reconstruction`: 0 for exact canonical-object equality, otherwise 1.
- `L_semantic`: 0 when semantic metadata round-trips exactly, otherwise 1.
- `L_provenance`: 0 when provenance metadata round-trips exactly, otherwise 1.
- `L_temporal`: clipped observation age / configured freshness horizon.
- `L_compliance`: 0 for admitted records; non-compliant records are rejected before encoding.
- `L_graph`: fraction of source-provided relations whose endpoints are missing from the source-provided entity set.

The reference loss is intentionally interpretable rather than differentiable. A learned backend may implement gradient updates for `Theta` and `Phi`, but must preserve the runtime receipt and compliance boundary.

## 8. CLI execution

Given a JSON list of records:

```bash
python -m jarvisx.m3_acme records.json
```

The process exits `0` when all records are admitted and `2` when any records are rejected. Output is a JSON receipt containing accepted/rejected counts, latent digests, abstractions, reconstructed structured records and component losses.

## 9. Promotion path

A learned 3D/tensor backend may replace the deterministic reference codec only after it adds:

```text
explicit numeric axis embedding
train/eval data contract
version-bound Theta/Phi artifacts
deterministic or reproducibility mode
reconstruction and anchor fixtures
semantic/provenance preservation tests
resource ceilings
model integrity verification
measured throughput and memory
rollback on failed Pi_Lambda validation
```

The deterministic reference implementation remains the conformance oracle for permission gating, provenance preservation, receipt structure and fail-closed decode semantics.
