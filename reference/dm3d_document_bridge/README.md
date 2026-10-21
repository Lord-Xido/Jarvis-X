# DM3D public document bridge — isolated interoperability reference

**Scope:** This is an intentionally generic, standalone demonstration of finite byte encoding, 3D spatial indexing, lossless PNG packing, OOXML Word embedding, exact decoding and SHA-256 verification. It is **not** the confidential DM-PCS runtime, private technical dossier, authenticated-encryption engine, trainable autoencoder, or an authoritative Jarvis-X codec. No private artefacts are included.

## Usage

    python reference/dm3d_document_bridge/codec.py selftest
    python reference/dm3d_document_bridge/codec.py encode source.py document.docx
    python reference/dm3d_document_bridge/codec.py decode document.docx restored.py
    sha256sum source.py restored.py

The reference uses Python 3.10+ standard library only. Independent Microsoft Word compatibility tests optionally use python-docx.

## Geometric contract

Bytes are represented as a finite 16³ or 32³ uint8 tensor with a version marker, uncompressed payload length, storage-mode marker, SHA-256 digest and optionally zlib-compressed bytes. Three rounds of independent x/y/z XOR differences precede deterministic Morton-ordered shell reindexing. The resulting tensor is mapped to a grayscale PNG atlas tiled by z-slices. The PNG is embedded through OOXML drawing and relationship parts inside a .docx. Decoding reverses each operation and validates exact binary identity.

N = (10^24)^((10^24)^(10^24)) can be stored as a finite symbolic expression without materializing its decimal expansion. Likewise, the script can encode a snapshot of its own source bytes; this does not execute recovered code or enable self-modification.

## Boundaries and security

- The input is bounded to 32,768 bytes, but incompressible data near that limit may not fit within the 32³ volume after metadata.
- DOCX and embedded PNG sizes are capped; malformed PNG structures, bad CRCs and altered data are rejected.
- **This example is not encryption:** SHA-256 is an integrity check, not authorship authentication or confidentiality.
- The reference is not compatible with the private DM-PCS archive or other Jarvis-X packet formats.
- This is an educational/research reference, not a hardened hostile-input parser or deployment component. Do not automatically execute decoded payloads.
- Untrusted document ingestion requires separately reviewed ZIP resource limits, signature validation, and security controls.
- Compression is best-effort; PNG/DOCX container overhead can exceed the source size. No self-optimisation, neural training or autonomous optimisation is claimed.
- Keep proprietary source, encryption secrets, confidential IP disclosures and private provenance outside this public repository.

## Regression suite

    cd reference/dm3d_document_bridge
    python -m unittest -q test_codec.py

The path-scoped GitHub Actions workflow runs the same isolated tests without changing production entrypoints or authoritative VM formats.
