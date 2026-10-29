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


## Phase 6: authenticated local participant processes (ADR-016, issue #425)

The isolated reference now has an optional process transport in ipc.py. It launches one **real independent Python process per participant** with one SQLite database and one Unix-domain socket each. The coordinator opens its own SQLite database in a separate process and uses IPCCoordinator, which overrides the participant boundary with authenticated socket calls rather than opening another participant's SQLite state. The four participants carry the existing *shadow observations* of VM, OS, ANN and geometry, not real live service state.

### Local-only RPC protection

- AF_UNIX sockets are bound under a trusted user-owned directory, chmod 0600; no TCP listener or public network port is created.
- The process-only environment variable JARVISX_SHADOW_IPC_SECRET must contain a randomly generated, 32–256-character ASCII secret. **Do not commit or embed this secret into browser JavaScript, URLs, logs or source code.**
- Every request binds protocol version, participant domain, method, canonical JSON arguments, timestamp and 128-bit random nonce into HMAC-SHA256; the service returns a corresponding authenticated response.
- Requests beyond a 45-second clock-skew window are rejected. Accepted nonces are persisted **in the owning participant's SQLite file**, preventing replay after process restart (within the validity window). One request per nonce; retries require a new signed nonce, while existing prepare/commit operations remain idempotent as defined in Phase 5.
- Input/response size is capped at 1 MiB, RPC method and argument shapes are explicitly allowlisted, and errors do not expose raw exception strings.
- No shared secret is generated or deployed by the repository. For local use, start four participant service processes and a separate coordinator with the **same securely supplied secret**, and restrict the containing directory's permissions.

### CI validation

    python -m unittest discover -s reference/canonical_shadow_tx -p test_ipc.py -v

CI runs each participant as a real Python child process and checks signed requests, invalid/tampered MACs, nonce replays (including after service restart), timestamp expiry, absence of operator approval, participant downtime, coordinated read gating and coordinator crash recovery after one local participant commit. An additional end-to-end process smoke check emits canonical-authenticated-process-receipt.json after recovery.

### Limitations and mandatory follow-up

This is authenticated **same-host local-process IPC**, not independently administered remote systems. Anyone able to read the process secret or edit the shadow SQLite files can bypass its logical protections. Unix socket permissions and HMAC do not replace OS account isolation, key rotation, TLS, operator RBAC, malicious-client resistance, durable leadership, network partition handling or safe authorization of arbitrary participant commands. A compromised authorized coordinator can still issue incorrect commands. Raw participant state reads can observe partially applied commits and must not be interpreted as an atomic distributed snapshot.

**Never connect this transport to production Dr Moagi OS, CodexVM, ANN weights, finance, vehicles, patient care or device kernels without additional independent security review, authenticated endpoint identity, durable per-service protocol changes and isolation verification.** Issue #425 remains open.


## Phase 7: fail-closed decision authority and production gate

A security review of the local IPC prototype identified a release-blocking flaw: an authenticated client with the transport shared secret could call a participant's commit directly before the coordinator had recorded a durable decision. Similarly, a signed abort could interfere with a transaction whose commit decision had already been durably recorded.

The local shadow service now loads the coordinator's **already durable SQLite transaction record in read-only mode** to verify each participant mutation:

- **prepare** requires a coordinator PREPARING/PREPARED intent and exact match of txid, epoch, fencing token, domain, declared profile, canonical bytes and verified whole-manifest hash;
- **commit** requires a COMMIT_DECIDED/COMPLETE coordinator decision, matching transaction and fencing token, and the same independently checked manifest;
- **abort** requires a durable ABORTED coordinator decision. The coordinator writes that decision **before** contacting participants, and recovery replays any unfinished participant aborts after restart.

The coordinator and participants still share a **local filesystem trust boundary**. A hostile process with permission to replace or rewrite the coordinator database (and its WAL files) can subvert the proof. Coordinator integrity is checked for the candidate manifest, but this is **not an independently signed distributed consensus certificate**. The existing in-process approve=True remains insufficient as an identity-bound operator permission. No new production networking, operator RBAC, live adapters, TLS, or independent storage authority is present.

The participant service now also rejects a group/world-accessible Unix socket directory to limit accidental local exposure. Accepted signed requests still require valid HMAC authentication and durable anti-replay nonces.

### Strict release gate

    python -m reference.canonical_shadow_tx.release_gate
    python -m reference.canonical_shadow_tx.release_gate --require-production

The first command emits a machine-readable JSON readiness inventory; the second **exits with code 2** because the required real-world production evidence is absent. CI checks this fail-closed behavior and uploads canonical-production-readiness.json alongside the shadow and recovery receipts. The build may be green while production deployment remains **BLOCKED**.

A production release still requires independently owned live VM/OS/ANN adapters, genuine atomic read isolation or explicit compensating sagas, operator identity and authorization, leader fencing across host/network failures, storage power-loss qualification, signed release/operations/incident runbooks and external security assessment. See issue #425. **Do not close #425 or label these references production-grade solely because the existing CI passes.**


## Phase 8: actual 3D geometric observability and mathematical verification

The executable spatial witness in geometry_witness.py binds the *existing isolated* four-domain transaction coordinator to a **non-coplanar tetrahedral 3D state representation**. This is an observability/verification transform, not a new backend or a distributed atomicity guarantee.

### Exact voxel geometry and bit-level reconstruction

Each participant is pinned to one immutable lattice coordinate in an 8 × 8 × 8 U8 volume:

| Domain | (x, y, z) |
| --- | --- |
| ANN | (1,1,1) |
| ENV1 geometry | (6,6,1) |
| OS | (6,1,6) |
| CodexVM | (1,6,6) |

The signed tetrahedron volume has magnitude 250/6 lattice-cubic units, nonzero, so these vertices span three independent spatial axes. For each domain index i in canonical ordered sequence (ANN, geometry, OS, VM), encode the state with byte 8*i + state_code + 1, where state codes are ABSENT=0, PREPARED=1, COMMITTED=2, ABORTED=3, UNREACHABLE=4. The other 508 voxels remain zero; invalid nonzero voxels are rejected at reconstruction.

Exact X-fastest addressing is L(x,y,z) = x + 8*y + 64*z. The status lattice occupies 512 bytes. Encapsulation via the existing 84-byte SHA-256 checked DM3D-ENV1 header yields a **596-byte** binary output. A Morton-to-row-major-to-Morton roundtrip and full per-vertex inverse reconstruction are validated on every encoding. SHA-256 protects against accidental byte corruption, not against a malicious party that can replace both data and hashes.

### Bounded 1000 inward steps without false computational claims

Represent each domain's observed commit progress as z(0) = 1 for COMMITTED, 1/2 for PREPARED, and 0 otherwise. Define visual contraction toward the all-committed reference point:

    z(k+1) = (z(k) + 1) / 2
    z(k)   = 1 - 2^(-k) [1 - z(0)]
    E(k)   = sum_i (1 - z_i(k))^2 = E(0) / 4^k
    0 <= k <= 1000

This is a closed-form **display-only** contraction with a mathematical fixed point; it does NOT execute 1000 ANN updates, automatically commit participants, converge a transactional system, accelerate hardware, or establish a learned encoder/decoder. The boolean logical_commit_fixed_point is true only when the **observed participant states** actually report COMMITTED across all four domains. For incomplete commits, geometrically approaching the all-committed visual target never elevates the transaction's completion status.

### Real end-to-end stage evidence

    python -m unittest discover -s reference/canonical_shadow_tx -p test_geometry_witness.py -v
    python -m reference.canonical_shadow_tx.geometry_smoke --out canonical-geometry3d-evidence

The smoke path runs the real isolated VM, sparse OS, trained synthetic ANN and 3D geometry observers. It then prepares four SQLite participants, interrupts after ANN applies, reads a **partially committed** 3D witness, and restarts coordinator replay to complete. It generates a JSON + binary ENV1 + isometric SVG triplet for PREPARED, COMMIT_DECIDED, and COMPLETE, along with summary.json. A separate signed Unix-socket subprocess test exercises exactly the same geometric witness through the authenticated participant-process boundary.

The witness checks actual durable transaction status, participant-domain state hashes/profiles, epochs, monotonic fencing tokens, envelope inversion and canonical candidate manifest. Intermediate stages can have valid structural evidence, but complete=false remains mandatory. The raw-state reads are not transactionally isolated across independent processes; only coordinator-gated COMPLETE is eligible for a completed commit receipt.

**Safety:** Geometry snapshots are read-only. No production services, financial controls, vehicles, clinical patient records, live credentials or hardware devices are connected. The existing fail-closed production readiness gate remains BLOCKED and issue #425 must remain open until the independent production evidence is available.
