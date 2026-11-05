# DM3D vNext integration boundary (research only)

This directory is an isolated Python numerical reference, not a replacement
for Jarvis-X's canonical VM, MM3D wire formats, Dr Moagi OS, or authority ledger.

## Contract map to accepted ADRs

- **ADR-001:** independent opt-in research module; no modifications to canonical source, codec interfaces, deployment targets or production entry points.
- **ADR-016:** Config provides finite resource limits; model parameters and omega represent adaptive model and memory state; train stages trial weights/memory and rolls back rejected candidates based on training and validation MSE. These transactions apply only to local in-process model state and are not the canonical cross-domain transaction protocol.
- **ADR-017:** the 3D Haar hierarchy records explicit detail coefficients. The nine-agent latent branch performs candidate encoding, inward ridge correction refines latent vectors, and an independent packet decoder reconstructs from serialized state. Dropping sparse coefficients is explicitly lossy.
- **External/world verification:** not implemented; the held-out synthetic reconstruction test is codec evidence on the test fixture, not world correspondence.
- **Wire compatibility:** the experimental .dm3d.npz format is a local NumPy ZIP container, not compatible with existing Jarvis-X deployed DM3D, MP4, or VM archive formats.

## Claim limitations

- Numeric near-lossless mode with all residuals preserved is not byte-for-byte losslessness for original media.
- The symbolic 10^18000000 address space and symbolic 10^(24*10^24) recursive budget are not executed.
- No GPU/distributed agent autonomy, self-modifying code, cryptographic authentication, adversarial ZIP security or general intelligence claim follows.
- SHA-256 is used only for accidental array-integrity checking, not authentication. Parser hardening is required before public/untrusted exposure.

## Promotion gates

A separate proposal would need canonical typed state, atomic cross-domain commit/rollback, policy authorization, appropriate external evidence, hardened serialization, and real-media generalization tests, independently reviewed and benchmarked.
