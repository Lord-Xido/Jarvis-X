# ADR-0037: Isolated DM3D DOCX iterative bridge with explicitly trusted local runtime

- **Date:** 2026-10-23
- **Status:** Proposed — pending maintainer review
- **Scope:** Public reference adapter only; private DM-PCS runtime remains external.

## Context

Jarvis-X already has a standalone public reference/dm3d_document_bridge/
demonstration for DOCX/3D indexing and separate read-only adapters. A separately
maintained DM-PCS prototype supports bounded source-code refinement,
reversible nested encoding, byte-exact decoding and regression checks.
Publishing the proprietary runtime or executable DOCX payloads would violate
the confidentiality boundary and imply unsupported production readiness.

GitHub integration must distinguish source integrity, tested observable
behavior, execution authorization, and software security.

## Alternatives considered

1. **Copy the private SDK and DOCX into public Jarvis-X.**
   Rejected: exposes proprietary materials and enlarges the attack surface.
2. **Automatically execute embedded DOCX code in GitHub Actions.**
   Rejected: violates explicit approval and hostile-input isolation boundaries.
3. **Leave local proof unintegrated.**
   Rejected: prevents independent compatibility tests and CI review.
4. **Isolated public adapter and explicitly pinned opt-in local runtime.**
   Selected: enables testable interoperability without exposing private code.

## Decision

Maintain reference/dm3d_document_bridge/v3_iterative_adapter.py as an isolated
public CLI, with synthetic unit tests, workflow CI, and usage documentation.

- Inspect bounded OOXML ZIP/XML documents and reconstruct marked source.
- Verify a separately supplied runtime script SHA-256 before explicit invocation.
- Invoke a trusted local SDK only after an explicit operator approval flag,
  using bounded round and nested-layer limits.
- Verify original, best and reconstructed byte streams against SHA-256.
- Independently invoke the preapproved SDK a second time for nested decode.
- Validate outer Morton-address/XYZ invariants and emit a local JSON receipt.
- Never upload DOCX documents, private source or keys to GitHub.
- Keep Python subset behavioral assertions scoped to tested inputs.

The private SDK and DOCX remain external; CI uses synthetic fixtures, not
confidential files. The adjacent pre-existing Auto Bytecode 3D Pipeline YAML
file is corrected by restoring real line breaks and workflow execution.

## Consequences and risks

Benefits: explicit trust boundary, path-scoped CI, reversible integrity tests,
repeatable local receipts, a useful compatibility contract, and no automatic
execution during DOCX inspection.

Limitations: the bridge is not a sandbox or full Python semantic validator.
Entrypoint hash pinning does not authenticate a source publisher or pin
imported dependencies. Morton coordinates are logical addresses, not proof of
3D GPU computation, neural training, or unlimited self-improvement. External
runtime failure may leave partial outputs; the adapter requires a fresh folder
and verifies artifacts before recording success.

## Validation

- Synthetic DOCX parsing, malformed ZIP/XML, marker and Morton XYZ tests.
- Fail-closed execution without approval or a matching runtime digest.
- A synthetic approved-runtime subprocess test covering the full bridge
  command protocol, independent decode, source hash and local JSON receipt.
- GitHub Actions Python 3.10, 3.12 and 3.13, with public codec tests.
- Main quality gate, repository system gate, and CodeQL before promotion.
- Local externally held DM-PCS end-to-end proof may inform the maintainer,
  but is not uploaded, published or run automatically in public CI.

## Migration, compatibility and promotion

- No changes to canonical Jarvis-X bytecode, installed package entrypoints,
  database schemas, existing private archive formats, or state migration.
- This is a reference CLI invoked by file path. Installing jarvisx neither
  executes nor installs the private runtime.
- The feature remains experimental; a future installed API requires a
  separate approved ADR, security isolation and versioning.
- Canonical main-branch promotion requires human maintainer review and the
  checks prescribed by GOVERNANCE.md. This proposed ADR is not itself an
  approval to publish proprietary material or deploy production services.
