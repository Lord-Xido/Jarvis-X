# Jarvis X / Dr Moagi — cryptographic IP baseline witness

**Scope:** Source-preservation and mathematical-evidence attestation of the exact public Jarvis X repository contents as they stood at the frozen revision. This document does **not** assert that the entire architecture is patentable, novel, privately owned, independently time-stamped or production-ready.

## 1. Historical source frozen without changing implementation

Repository: https://github.com/Lord-Xido/Jarvis-X

| Anchor | Value |
|---|---|
| Frozen historical commit | b511300d565cc13865e3c2026050538bf9e2b149 |
| Frozen Git tree | 302ae87d36bd00d57fddd350c7ecf44bc4f92821 |
| Archived branch | ip/frozen-20261029-b511300d565c |
| Commit date recorded by GitHub | 2026-10-29T16:44:36Z |
| GitHub commit signature verification | verified=true, reason=valid on the GitHub API for the squash-merge commit |
| Tracked Git blobs in historical snapshot | 969 |
| Sum of tracked blob contents | 8,909,527 bytes |
| Cryptographic content hash | SHA-256, NOT only Git SHA-1 |
| Fixed SHA-256 Merkle root | 14f701faa0cf515b8863d1cf09dfa36ecae8582cf1ee6ffe6f9f35ab606fbebb |
| Canonical full manifest SHA-256 | 3d30ff365b62636bbf85206a62a07cf84b91a7f048d82165f525411804db85e0 |

The archived branch is an additional reference, not an immutable protected reference. A Git branch can be forcibly moved or deleted by a sufficiently authorized actor. The **specific commit and tree objects**, together with the published SHA-256 content manifest and fixed SHA-256 Merkle root, are the verification anchors. Keep independent offline backups.

The GitHub verified GPG signature is by the GitHub squash-merge service; it does not independently authenticate an inventor's signature or prove title. The commit date is recorded by GitHub and is **not** an RFC3161 independent trusted timestamp.

The copyright license in the frozen repository is MIT. This witness does NOT revoke, modify or override the copyright license or any third-party rights.

## 2. Byte-exact source envelope and Merkle formula

Read the historical git tree **as committed**, not current working files. Every tracked regular file, executable blob or symlink blob is represented by immutable Git object ID plus exact Git path bytes and mode.

For byte path p, mode m, Git blob bytes b:

    B = SHA256(b)
    L = SHA256(
        "JARVISX-IPFREEZE/v1/leaf\0" ||
        u32be(len(p)) || p || u8(len(m)) || m ||
        u64be(len(b)) || B
    )

Sort all leaves by raw path bytes ascending. At each level, pair successive 32-byte hashes; if odd, duplicate the final hash. Define:

    N = SHA256("JARVISX-IPFREEZE/v1/node\0" || left32 || right32)

Repeat to a single 256-bit root. File index, mode, path bytes, size, per-file SHA-256 and SHA-256 Merkle root are recorded in a deterministic canonical JSON manifest. **An altered file byte, mode, path, omission or ordering change produces a different digest, barring a cryptographic hash collision.** SHA-256 offers computational collision resistance; it is not a mathematical guarantee that distinct inputs can never hash alike.

The frozen historical commit remains unchanged while proof code is added in subsequent commits. Only tracked bytes are covered: no private chat files, untracked work, external libraries, user-held keys, remote deployment, third-party patent rights or absent Git LFS payloads are implicitly frozen.

## 3. Bitwise, mathematical and geometric exactness in the 3D witness

The bounded shadow witness has four typed domains at fixed coordinates:

    ANN       = (1, 1, 1)
    Geometry  = (6, 6, 1)
    OS        = (6, 1, 6)
    VM        = (1, 6, 6)

For A, B, C, D in this order, u=B-A, v=C-A, w=D-A, the exact determinant det[u,v,w] is -250 and signed tetrahedral volume is:

    V = det[u,v,w]/6 = -125/3 != 0

This proves noncoplanarity of the **four status vertices**, not that hardware is fabricated in three physical dimensions.

The 8x8x8 U8 lattice has 512 voxels and canonical row-major index:

    L(x,y,z) = x + 8*y + 64*z.

The byte assigned to a vertex is 8*domain_index + status_index + 1, with domain_index 0..3 and status_index 0..4. There are 4x5=20 distinct vertex-state codes and 5^4=625 possible four-participant status combinations. The math checker independently encodes ALL 625, checks bit-identical reconstruction using the frozen geometric codec, verifies that all 512 Morton indices form a permutation and SHA-256-commits the complete finite status atlas. The 84-byte DM3D-ENV1 header and 512-byte payload yield **596 bytes**, with packet SHA-256 integrity.

The rational contraction starts with z(0) = (1, 1/2, 0, 0) and runs **1000 actual exact rational iterations**:

    z(k+1) = (z(k)+1)/2
    z(k)   = 1 - 2^-k(1-z(0))
    E(k)   = sum_i(1-z_i(k))^2 = (9/4) / 4^k
    E(1000) = 9 / 4^1001

Every rational step is verified with numerator and denominator arithmetic, not floating point. This proves an exact property of a **display-only contraction**, not 1000 full ANN inference or learning steps, hardware throughput, a 10^18-voxel materialized machine, general lossless multimodal compression or successful real distributed ACID commits. The production gate remains BLOCKED.

## 4. Reproduce and independently verify

A full Git checkout including the frozen commit is required. From the repository root:

    python -m reference.ip_provenance.freeze --out ip-frozen-baseline-manifest.json --require-anchor
    python -m reference.ip_provenance.freeze --verify ip-frozen-baseline-manifest.json --require-anchor
    python -m reference.ip_provenance.math_exact > ip-3d-exact-mathematics.json
    python -m unittest discover -s reference/ip_provenance -p 'test_*.py' -v

The two SHA-256 values in this document are also pinned as separate files at docs/ip/2026-10-29-root.sha256 and docs/ip/2026-10-29-manifest.sha256.

CI repeats the tests, reproduces the entire full-tree SHA-256 manifest, checks both fixed anchors and archives the source-inventory and mathematical proof JSONs.

## 5. Optional owner-held detached digital signature

No owner private key is created in the repo or available to the assistant. A personal Ed25519 signature needs a **separately generated and securely controlled** owner's private PEM file; verify the public key against an independently documented identity. Do not upload private keys or commit them.

Offline example on a trusted machine:

    openssl genpkey -algorithm Ed25519 -out /secure/owner-private.pem
    chmod 600 /secure/owner-private.pem
    openssl pkey -in /secure/owner-private.pem -pubout -out /secure/owner-public.pem

    python -m reference.ip_provenance.sign sign --manifest ip-frozen-baseline-manifest.json --private-key /secure/owner-private.pem --out owner-attestation.sig.json

    python -m reference.ip_provenance.sign verify --manifest ip-frozen-baseline-manifest.json --signature owner-attestation.sig.json --trusted-public-key /secure/owner-public.pem

The signed digest is domain-separated and binds the entire canonical manifest to the frozen tree, but signature validity alone identifies only the corresponding key, not a legally established inventor, assignee or patent owner. Obtain an independent qualified timestamp (e.g. RFC3161) if needed and keep it with the exact signed manifest and public-key identity record.

## 6. Legal/evidentiary qualification

- This is a strong **technical integrity** and source history witness.
- It is not a patent application, patent grant, proof of originality, inventorship, assignment, freedom to operate or copyright exclusivity.
- The original public MIT licensing is unaffected.
- Public disclosure before filing can affect patent novelty and filing options; consult a qualified patent practitioner about priority dates and jurisdiction-specific deadlines.
- Substantive production qualification remains separately controlled; the binary 3D proof is not a live service safety approval.
