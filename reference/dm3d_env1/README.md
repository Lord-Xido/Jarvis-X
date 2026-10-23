# DM3D-ENV1: public cross-language geometry and state-envelope reference

**Status:** bounded integration laboratory, not the confidential DM-PCS protocol, a production VM, authenticated encryption, an autoexecuting DOCX, or evidence of system-wide adoption. This independent reference implements only a strict portion of [ADR-016](../../docs/adr/0016-canonical-typed-state-transaction-and-geometry-profiles.md) and [ADR-017](../../docs/adr/0017-dr-moagi-operational-autoencoding-equation.md).

## Binary wire contract

Unsigned **big-endian** integer fields; byte offsets start at zero. No optional sections, flags, extra bytes or implicit layout conversions. Total fixed header: **84 bytes**.

| Offset | Bytes | Field | Canonical meaning |
|---:|---:|---|---|
| 0 | 8 | magic | ASCII DM3DENV1 |
| 8 | 2 | version | 1 |
| 10 | 2 | flags | 0 only |
| 12 | 16 | profile | ASCII DM3D-CUBE-U8-V1 plus one zero padding byte |
| 28 | 1 | layout | 1 = X-fastest row-major |
| 29 | 1 | boundary | 1 = periodic torus (neighbour geometry) |
| 30 | 2 | reserved | 0 |
| 32 | 4 | dimension X | cube side |
| 36 | 4 | dimension Y | equals X |
| 40 | 4 | dimension Z | equals X |
| 44 | 8 | payload byte count | side cubed |
| 52 | 32 | SHA-256 | digest of payload bytes only |
| 84 | side³ | payload | 8-bit voxel values |

The supported cube sides are 2, 4, 8, 16, 32 and 64. Maximum data payload is **262,144 bytes** and maximum complete packet is **262,228 bytes**. The serializer rejects malformed, noncanonical, oversized or truncated envelopes and wrong SHA-256 payload digests.

The payload is **not stored in Morton order**. It uses X-fastest layout:

- Row-major: L(x,y,z) = x + side*y + side²*z.
- Optional Morton reinterpretation: M(x,y,z) = Σ[k] ((x_k << 3k) + (y_k << (3k+1)) + (z_k << (3k+2))).
- Known coordinate: side=16, (x,y,z)=(1,2,3) gives L=801 and M=53.

To use a Morton-based runtime, convert after decoding and invert the permutation before re-encoding. An actual learned or floating-point payload requires a **different, explicitly versioned profile** with typed precision semantics.

## Reproducing the reference

Python 3.10+ and C++17 standard library, no third-party dependency:

    cd reference/dm3d_env1
    python -m unittest -v test_envelope.py
    python envelope.py pack input.bin 16 document.env
    python envelope.py inspect document.env
    python envelope.py unpack document.env output.bin
    g++ -std=c++17 -O2 -Wall -Wextra -Werror envelope.cpp -o envelope_cpp
    ./envelope_cpp pack input.bin 16 cpp.env
    ./envelope_cpp inspect cpp.env
    ./envelope_cpp unpack cpp.env cpp-output.bin
    ./envelope_cpp index 16 1 2 3

The conformance tests check byte-identical envelopes from Python and C++, independent reverse-direction decoding, exact Morton geometry, malformed-input rejection, and candidate-first transaction behaviour. GitHub CI runs Python versions 3.10, 3.12 and 3.13, and both GCC and Clang compile checks.

## Provisional-versus-authoritative transaction demonstration

ReferenceTransaction is a **one-domain, in-memory** model of the ADR-016 promotion rule. Stage a candidate without changing the authoritative state; verify its shape, hash and roundtrip; require explicit local approval before committing; reject stale epochs or tampering.

    from envelope import Geometry, ReferenceTransaction
    tx = ReferenceTransaction(bytes(512), Geometry(8))
    proposal = tx.stage(bytes([0xA5]) * 512)
    assert tx.epoch == 0
    receipt = tx.commit(proposal, approve=True)
    assert receipt["decision"] == "COMMITTED"

Receipts have before/candidate hashes, an epoch, declared verification gates, a commit decision and a deterministic hash-chain head. All this is **illustrative**, not cryptographic authorization, tamper-proof provenance, network auth, multi-domain ACID commit, durable checkpointing or crash-safe journal persistence. The example has no external-world claims: external evidence is marked NOT_APPLICABLE rather than PASSED. No neural, fixed-point, learned, encrypted or compressive semantics are implied by a raw byte transport envelope.

Existing Dr Moagi OS, DMOS2, DM3D Word interfaces, MP4 codecs and Moagi3D floating-point evolution remain separate. A future adapter should explicitly map data and state domains, preserve necessary residuals for lossless reconstruction, provide typed stage receipts and prove complete commit/rollback semantics across all namespaces it touches.

## Security and intellectual-property boundary

An ordinary SHA-256 header is an integrity check **without authenticity**: a malicious author can alter both bytes and digest. Never run code from a DOCX or a parsed envelope. Do not use this bounded reference as a hostile-document parser, secret store or deployment security control. No private DM-PCS implementation, documents, model weights, private protocol identifiers, credentials, encrypted archives or confidential benchmark/IP dossiers belong in this public directory.
