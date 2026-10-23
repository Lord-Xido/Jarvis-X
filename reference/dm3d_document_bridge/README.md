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


## DM3D v2 DOCX integration adapter (public interface, private engine)

The sibling `v2_adapter.py` offers a **read-only** check for the locally held v2 Word package and an explicit local subprocess bridge to a **separately supplied, trusted Python runtime**. The public repository does not contain the private DOCX, runtime source, training weights, output media or IP dossier.

    python reference/dm3d_document_bridge/v2_adapter.py inspect /private/path/engine.docx
    python reference/dm3d_document_bridge/v2_adapter.py run /private/path/engine.docx factorial5 \
      --runtime /trusted/local/dm3d_runtime.py --out /tmp/dm3d-results --approve-external-runtime

The `run` subcommand is disabled without the explicit approval flag. The verifier checks ZIP path/size limits, required `dm3d/` members, the SHA-256 runtime digest, program names, ANN architecture metadata, ROM/RAM map metadata and Morton key shape. It does **not** execute embedded Python, prove trustworthiness, authenticate the author or exhaustively validate all media blocks. The adapter never downloads code, mutates the Word file, or performs hidden code updates. Review the separately supplied external runtime before approving execution.

    cd reference/dm3d_document_bridge
    python -m unittest -q test_v2_adapter.py

All test DOCX packages are constructed in temporary directories from synthetic content. The public bridge is intended to remain distinct from the separately controlled proprietary package.


## DM3D 1,000-instruction DOCX bridge (public interface only)

The standalone `v1000_adapter.py` recognizes the separate DM3D 1000-op document format without publishing or importing its private VM implementation. It validates bounded ZIP/XML structure, the 1,000 eight-byte instruction records, contiguous program counters, per-record CRC16, stored SHA-256 digests and payload size limits. The adapter **does not decode the private payload**, verify the declared source digest against decoded bytes, authenticate the document, or execute anything during inspection.

    python reference/dm3d_document_bridge/v1000_adapter.py inspect /private/path/DM3D_1000_Bytecode_VM.docx

A locally provided, independently reviewed private runtime is invoked only with explicit approval; use optional runtime SHA-256 pinning to guard against unintended file replacement:

    python reference/dm3d_document_bridge/v1000_adapter.py run /private/path/DM3D_1000_Bytecode_VM.docx \
      --runtime /trusted/local/dm3d_1000_bytecode_vm.py \
      --output /tmp/restored.bin \
      --runtime-sha256 <approved-runtime-sha256> \
      --approve-external-runtime

The public adapter is **not** the bytecode interpreter. Its synthetic test suite has no private VM program, document, input, output, trade secret, credential, or model weight:

    cd reference/dm3d_document_bridge
    python -m unittest -q test_v1000_adapter.py

All confidential source and complete 1,000-instruction documents remain external to this public repository. Review local documents and runtime before approving execution; SHA-256 and CRC detect corruption but do not establish authenticity or trustworthiness.
