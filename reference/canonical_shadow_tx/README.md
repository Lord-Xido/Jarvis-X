# CanonicalShadowTx/1 — isolated multi-surface integration reference

**Status:** Bounded, executable reference laboratory. **Not** an authoritative replacement for CodexVM, Dr Moagi OS, the trained neural model or the canonical byte envelope. Addresses *part* of ADR-016, specifically a shared candidate/receipt representation across four real isolated engines.

## Implemented path

1. Run a **fresh, in-memory CodexVM**: SET A=19, SET B=23, ADD Ψ=A+B, HALT. Its own journal is checked and an independent arithmetic oracle enforces Ψ=42.
2. Run a **fresh DrMoagiOSKernel** with an 8³ sparse field, 512-cell resource limit, and **state_dir=None**. Its internal cycle commits only to its isolated, throwaway instance. The bridge rechecks an independently decoded DMOS2 packet against the observed coordinate/value snapshot and captures a verified OS journal head.
3. Run the **trained synthetic-vocabulary Conv3D ANN** from apps/recursive-3d-ann/runtime.py against its published 64-token example. Compare predicted symbols to the original reference, require finite latent state, reconstruct exact original bytes with the XOR residual, and record a canonical token-proof snapshot. This does **not** infer general language understanding or compression superiority.
4. Reconstruct a strict 4³ raw byte cube from the **DM3D-ENV1** reference and verify its checksum and reversible Morton indexing.
5. Make one **immutable typed DomainFrame** per surface: domain, declared profile, and canonical bytes. Stage the sorted set of **exactly four domains**, independently check their structural and reconstruction invariants, enforce per-frame and aggregate byte budgets, hash the candidate, and leave the bridge's owned authoritative state unchanged.
6. Require explicit *local in-process* approval, re-check every frame and candidate seal, and atomically promote only the coordinator's **owned in-memory snapshots**. Persist no files and perform no network operations. Record deterministic SHA-256 lineage and verify replay.

The system equation is:

    B_cand = stage(VM_shadow, OS_shadow, ANN_shadow, ENV1_shadow)
    B_next = B_cand if all_internal_gates_pass AND explicit_local_approval else B_before

External evidence is NOT_APPLICABLE, not elevated to PASS. Per-source checks establish internal consistency, not independent correspondence to reality. A SHA-256 seal is an integrity checksum, not cryptographic authentication of the caller.

## Run

From repository root (PyTorch is required for the ANN adapter):

    python -m pip install -e ".[test]" -r apps/recursive-3d-ann/requirements.txt
    python -m unittest discover -s reference/canonical_shadow_tx -p test_bridge.py -v
    python -m reference.canonical_shadow_tx.bridge

A dedicated GitHub workflow validates real imported backend behavior (not mocks or API shims), all-or-none promotion of four bridge-owned snapshots, stale/tampered proposals, invalid authority/profile, OS packet corruption, ANN residual corruption and VM independent arithmetic mismatch. It uploads a machine-readable verification receipt.

## Contract limits

- **No multi-process ACID:** The VM and OS already own independent internal transaction boundaries. The bridge runs them against throwaway isolated instances; it cannot undo a mutation to a live OS, VM, database or other application.
- **No durable crash recovery:** Journal and state live only in this Python process. No WAL, fsync or recovery protocol exists.
- **No new credentials:** The approve=True argument is a local execution guard, *not* identity authentication or RBAC. Remote endpoints are not exposed.
- **No universal binary schema:** ENV1 remains a U8 4³ data envelope; DMOS2 remains separate compressed float64 sparse geometry; the ANN uses synthetic tokens. A bridge receipt records their declared identities, not a universal lossless conversion between incompatible formats.
- **No independent ANN generalisation or fixed-point theorem**, no accelerator benchmark, no physical 10^18-cell allocation, and no self-modifying deployment.
- **Only one bounded demonstration:** Arithmetic oracle uses a known 19+23 program; OS 8³ demo; ANN stored 64-token reference; ENV1 4³ cube. A production adapter must add general program/geometry coverage, authenticated operation, version negotiation and multi-domain durable commit/rollback protocols before authority promotion.

The src/jarvisx imports, live control-plane routes, persistence, and model weights are unchanged. This is a forward-compatible research bridge, not a production migration.
