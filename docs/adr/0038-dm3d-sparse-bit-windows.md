# ADR 038 Isolated DM3D sparse-bit Windows runtime

- Status: Proposed integration candidate
- Date: 2026-10-26
- Tracking: https://github.com/Lord-Xido/Jarvis-X/issues/409

## Context

The local sparse-bit computer has a working Windows implementation and tests, but
the repository's canonical Python VM and C++ DM3D runtimes use different state and
instruction formats. Sharing a DM3D name does not make these formats compatible.
The requested decimal 1 GB executable size is a distribution constraint.

## Decision

Add an isolated app under apps/windows/dm3d-sparse-bit. Keep the canonical entrypoints
and existing bytecode formats unchanged. Document D3VM as the initial bit-register
format and DM3D version 1 as its canonical snapshot. Any future translation must
explicitly validate semantics and bounds before promoting state to another backend.

Use the existing Windows .NET Framework x64 compiler with no runtime dependency
downloads. Build compact by default; make exact 1,000,000,000-byte packaging an
explicit switch. Record executable-image bytes and zero-filled reserve separately.
Generate binaries and acceptance reports in CI artifacts rather than Git history.

Require built-in dense-reference, codec, Morton, interpreter and quota tests plus
independent process tests against both compact and final padded executables. Tests
operate in a temporary copy, never the user's data directory.

## State and transaction boundary

The authoritative state is a bounded active-bit set plus iteration counter. A STEP
computes a fresh GF(2) candidate and commits only after validating the active quota.
A bytecode run is not all-or-nothing: earlier successful instructions remain after
a later fault. Snapshot decoding validates the complete input before replacing
memory. Snapshot save flushes a temporary file then replaces the destination.

The command router is local. It does not intercept another application's traffic.
Audit records expose BEGIN, OK and FAILED outcomes, but do not provide automatic
crash replay, distributed exactly-once effects or canonical promotion receipts.

## Consequences

- Review and build are isolated from the canonical control plane.
- The large reserved file region has no functional or performance significance.
- Python 3.10+ is needed for acceptance tests, not execution.
- Windows compatibility, signing, hostile concurrent filesystem access and
  production hardening require separate work and evidence.
- A future adapter can connect an explicitly selected caller, with its own format
  translation, authorization and verification contract.
