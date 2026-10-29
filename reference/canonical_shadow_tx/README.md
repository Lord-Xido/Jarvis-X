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


## Phase 4: Durable SQLite shadow-store laboratory (ADR-016 issue #425)

The companion implementation in durable.py is a **single SQLite database** with WAL journaling and synchronous=FULL. It persists four typed backend snapshots, prepared proposals, atomic commit receipts and a verifiable hash chain.

**Semantic boundary:** SQLite commits the four coordinator-owned logical namespaces together because all writes occur inside ONE SQLite transaction. It does **not** commit live VM, OS, ANN or other independently owned process state. The observed backend instances remain isolated and ephemeral. This is a bounded *single-store* transaction laboratory, not cross-process ACID, two-phase commit, distributed consensus or an automatic migration.

### State machine and sequence

    NEW -> PREPARED (durably persisted candidate, current state untouched)
    PREPARED -> COMMITTED (explicit approval, epoch/hash check, one SQL transaction)
    PREPARED -> ABORTED (explicit abort, no current-state changes)
    COMMITTED -> COMMITTED (retry returns the same receipt)
    PREPARED + concurrent epoch advance -> STALE (manual abort/reprepare needed)

- prepare(txid, frames) verifies the same four independent backend outputs as CanonicalShadowTx/1, records candidate bytes, source epoch/hash and versioned geometry profiles. At most 32 transactions may be outstanding.
- commit(txid, approve=True) uses BEGIN IMMEDIATE, validates and rechecks candidate evidence, writes every domain inside one SQL transaction, appends a receipt, moves the epoch and commits the WAL in one SQLite transaction. An exception or process termination before COMMIT rolls the whole database transaction back on reopen.
- Crashes **after** SQLite COMMIT may prevent return to caller but not the durable decision; a retry using the same transaction ID returns the stored receipt.
- abort(txid) rejects a pending proposal and persists the decision. Prepared records are deliberately **not** replayed/committed automatically on restart, because approval authority may have changed.
- Every open executes SQLite quick_check and verifies the current snapshots, before/after hash chain, commit sequence and receipt metadata. Corrupt state or journal causes a fail-closed exception. SHA-256 is an integrity checksum, not proof against a malicious local administrator who can rewrite both data and hashes.
- Concurrent SQLite writers are serialized by BEGIN IMMEDIATE. Changes between prepare and commit are rejected as stale (epoch and state hash compare-and-swap).

The dedicated CI tests process death using os._exit at each of four domain writes, before SQL COMMIT and after SQL COMMIT; reopens the same database and proves either **all four** coordinator-owned snapshots committed or **none** did. Fault injection also tests a simulated I/O exception, stale writers, bad manifests, manual abort and tampered journal/state.

**Reproduction:**

    python -m unittest discover -s reference/canonical_shadow_tx -p test_durable.py -v

The workflow uploads canonical-durable-receipt.json after generating a real VM/OS/ANN/geometry batch and reopening the database to verify its persisted transaction. Performance, durability and forensic guarantees are limited by the SQLite filesystem, operating system and storage flush semantics; the CI process-kill test does **not** emulate sudden physical power loss.

**Next production blockers:** Adapters must implement independent prepare/apply/recovery ownership and either a distributed commit protocol with explicit read visibility guarantees or a saga with compensating actions; the single-DB reference cannot provide cross-service atomic visibility. Further work needs authenticated initiators, hostile-input fuzzing, long-running concurrency tests, a durable WAL recovery strategy for all backends, independent external evidence and a human security review before changes to live services.


## Phase 5: independent participant files and durable coordinator decisions

The next bounded reference is implemented in multiprocess.py. It stores **four independent shadow domain copies in four SQLite files**, and a fifth SQLite file contains an authoritative *coordinator decision*.

The protocol implements **eventually recoverable committed intent**:

    coordinator PREPARING
        -> each local participant PREPARED (fenced by token and epoch)
        -> coordinator PREPARED
        -> explicitly authorized coordinator COMMIT_DECIDED [durably logged]
        -> participant A applies
        -> participant B applies
        -> participant C applies
        -> participant D applies
        -> coordinator COMPLETE

If a process fails **before** COMMIT_DECIDED, restart recovery aborts the unfinished prepared candidates; no participant has applied new state. If a process fails **after** the durable decision, recovery replays remaining idempotent participant applies and only then marks the operation COMPLETE. Coordinator-issued monotonic fencing tokens and per-participant epoch checks reject stale candidates. One unresolved transaction is allowed at a time.

**Critical isolation limitation:** Between COMMIT_DECIDED and COMPLETE, direct reads of each participant can see different state versions. This is **not atomic cross-process visibility**, and the code does not offer distributed ACID, consensus, an authenticated RPC transport, compensating transactions, leases, leader election, or a proof of recovery under arbitrary malicious failure. The coordinator's explicit coordinated_read(txid) gate refuses intermediate states, but cannot prevent raw participant reads. A participant can be left permanently unapplied if its database disappears: recovery fails closed rather than forging completion. All participants are *shadow SQLite files*, not live VM/OS/ANN services. SQLite WAL=FULL is configured but physical power-loss guarantees are outside CI.

### Reproduce the reference

    python -m unittest discover -s reference/canonical_shadow_tx -p test_multiprocess.py -v

The GitHub workflow runs separate-participant regression tests (real training-checkpoint observations plus SQLite persistence), process-kill after the coordinator decision and after successive participant applications, interrupted prepare, stale-fence rejection, corrupt participant evidence, missing participant and proof that raw uncoordinated readers can see partial state. It produces a JSON receipt after restarting recovery following a deliberately interrupted first participant application.

**Operational safety:** No code under src/jarvisx or any live API endpoint is changed. The coordinator's approve=True is a local guard, not an authenticated identity check. No real-world financial, vehicle, medical, hardware or host-control operations are connected.

To achieve an actual production-grade distributed transaction, separately owned services must implement durable prepare/commit/recover, authenticated RPC, arbitrated leadership/fencing, transactionally isolated reads or explicitly documented sagas, durable failure evidence, recovery under network partitions, and external independent safety review. Issue #425 remains open for that work.
