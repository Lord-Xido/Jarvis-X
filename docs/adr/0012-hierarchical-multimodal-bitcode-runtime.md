# ADR-012: Hierarchical multimodal bitcode runtime

- **Status:** Accepted
- **Date:** 2026-09-01
- **Decision owners:** Jarvis-X maintainers

## Context

Jarvis-X has specialist paths for bytecode, volumetric state, graphics, animation,
firmware, model packaging, and multimodal orchestration. The ROM Forge material
adds more format-specific prototypes. Treating each file type as unrelated would
duplicate transport, integrity, resource, and provenance semantics. Treating all
bytes as self-describing would be equally incorrect: an arbitrary bit string has
no unique interpretation without a format or schema contract.

The repository therefore needs a common binary intermediate representation below
semantic frontends and above raw storage. It must preserve arbitrary bytes,
remain deterministic, reject hostile size declarations, and avoid granting
decoded content execution authority.

## Decision

Adopt `.jxbi` version 1 as the canonical Jarvis-X universal bitcode envelope for
typed digital artifacts.

The dependency direction is:

```text
format-specific frontend/backend (optional)
              |
              v
representation contract + opaque bytes
              |
              v
universal bitcode envelope
              |
              v
storage / transfer / evidence
```

The envelope consists of:

1. a fixed big-endian header;
2. a canonical JSON manifest;
3. an ordered encoded payload;
4. SHA-256 at artifact, manifest, payload, and chunk levels;
5. a domain-separated binary Merkle root over raw chunks.

Version 1 uses per-chunk `zlib` only when it is smaller and `identity` otherwise.
It carries an explicit `RepresentationContract` with media kind, media type,
format name, source name, optional schema, and bounded metadata.

The recursive closure is executable, not metaphorical:

```text
Q0 = encode(B, C)
(B_hat, C_hat) = decode(Q0)
Q1 = encode(B_hat, C_hat)
accept iff B_hat == B and Q1 == Q0
```

No `.jxbi` decode path executes the artifact, imports it, renders it, or writes to
a path derived from container metadata.

## Authority boundary

Universal bitcode standardizes representation and integrity. It does not replace
the canonical VM or `Lambda` policy boundary. A decoded program, model, document,
or media object remains data. Any execution or external side effect requires a
separate explicit lowering, capability projection, bounded execution, verification,
and commit path.

Format detection is advisory. Signatures and extensions provide representation
hints; they do not authenticate origin or establish semantic truth.

## Legacy-source disposition

The 16 ROM Forge artifacts are not copied into live package namespaces. Several
are explicit stubs or perform eager side effects, and one notebook is structurally
invalid. Their top-level hashes and audit dispositions are retained in
`reference/rom_forge_legacy/manifest.json`. Tested common primitives are promoted
through the new runtime rather than by importing prototype code.

## Consequences

Positive:

- one versioned transport and integrity contract for every digital modality;
- exact deterministic round trips and reproducible container identity;
- localized corruption evidence through chunk hashes and a Merkle root;
- explicit resource bounds before decompression;
- semantic adapters can evolve independently of storage framing;
- legacy prototypes have traceable provenance without becoming authoritative.

Costs and limits:

- canonical manifests add overhead, especially for many small chunks;
- `zlib` is a portability reference, not a state-of-the-art media codec;
- format hints do not parse or validate full media grammars;
- cross-modal conversion remains a separate model/backend concern;
- SHA-256 gives integrity, not confidentiality, authenticity, or truth.

## Alternatives rejected

### Commit the supplied ZIP archives directly

Rejected because opaque archives conceal placeholders, invalid files, duplicate
dependencies, import-time effects, and unreviewed deployment templates.

### Make the latent ANN representation the only interchange format

Rejected because learned representations are not generally lossless, stable
across model versions, or sufficient to recover original arbitrary bytes.

### Infer all semantics from raw bytes

Rejected because unlabelled bit strings admit multiple interpretations. The
representation contract is a first-class input.

### Reuse the canonical VM bytecode container

Rejected because executable instruction authority and passive multimedia/data
transport have different validation and threat boundaries.

## Verification

The focused suite covers:

- signature, extension, UTF-8, JSON, and fallback detection;
- empty, compressible, incompressible, and multi-chunk round trips;
- deterministic canonical re-encoding;
- header, manifest, payload, raw-chunk, layout, and Merkle corruption;
- truncated, trailing, noncanonical, and malformed containers;
- bounded decompression and configurable resource ceilings;
- explicit contract validation;
- atomic CLI encoding, inspection, verification, decoding, and overwrite refusal.
