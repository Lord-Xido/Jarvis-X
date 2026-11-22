# JARVIS-X Ω³ v2.3 — Trust-anchored publication and consumer gateway

**Status:** A reference implementation, not a production deployment or a proof of unbypassable security. The original v1.0 mathematical baseline and the C++17 fractal engine are unchanged.

## Why v2.3 exists

Earlier evidence forgery testing demonstrated that the v2.1 verifier could accept a modified voxel when an adversary recomputed the unkeyed final-state digest and made checkpoint, trace and report mutually consistent. The v2.2 independent replay corrected that arithmetic failure, but used an HMAC shared key and depended on downstream consumers choosing to verify published artifacts.

v2.3 moves to three logical roles:

1. **Untrusted producer:** Writes checkpoint.bin, trace.csv and run.json; receives no signing credential or write access to trusted releases.
2. **Trusted publisher:** Uses no-follow descriptor reads to snapshot precisely the submitted evidence, independently replays every fixed-genesis Q16 six-neighbour transition, and refuses mismatches. The publisher signs SHA-256 evidence digests and workload scope using Ed25519, and publishes a uniquely named release by same-filesystem rename and fsync.
3. **Consumer gateway:** Loads a public-key allowlist and minimum release sequence from a *separately administered* policy, validates the Ed25519 certificate, hashes the actual bytes read from open file descriptors, checks checkpoint metadata, and returns verified checkpoint bytes in memory or stdout. Raw producer output never qualifies as a trusted release.

The consumer requires **no private signing key** and does not return a filesystem path as a substitute for verified data. An expired or revoked signer, a release below the protected minimum sequence, or a modified evidence file fails closed.

## Mathematical contract

For fixed initial Q16 field and the documented recurrence, the publisher accepts only if independently replayed states match all trace steps, the final checkpoint state and the final hash. Confidence scores are not part of the authorisation function.

**Authenticated release** requires simultaneously: successful independent deterministic replay, an approved signing identity and policy, a valid Ed25519 signature on the canonical manifest, and matching SHA-256 digests of the exact consumed files.

This is an implementation-specific contract. It is not a proof that the operating system, key service or all external applications enforce the contract.

## Reproducing the tests

Install the pinned Python package **cryptography==46.0.4** in an isolated environment. Compile the existing C++17 kernel with CMake, then run:

~~~sh
python3 experimental/fractal_cloud_v2/tests/test_containment_v23.py \
  /tmp/jx3d-v23-build/jx3d_cloud \
  experimental/fractal_cloud_v2/tests/golden_v1_depth2_steps24.csv
~~~

The 21-case adversarial suite covers legitimate publishing, arithmetic evidence forgery, checkpoint and certificate tampering, forged signer substitution, public-key revocation, minimum-sequence enforcement, unsigned output rejection, symlink/duplicate-JSON rejection, protected key permissions, consumer-memory consistency and group-read-only publication. Test keys are randomly generated and discarded; no production signing key is stored in GitHub.

The dedicated CI additionally replays **160,000 cells for 24 steps**, signs the accepted result and independently verifies the release with a pinned public key. This is correctness evidence on a hosted CPU runner, not a bandwidth or availability measurement.

## Separate OS identities and access control — deployment requirements

| Resource | Example ownership / mode | Requirement |
| --- | --- | --- |
| Producer output | producer:evidence-reader, 0750 | Governor may read; producer may not alter release namespace |
| Signing private key | governor:governor, 0600 (parent 0700) | Accessible only to governor signer principal |
| Releases root | governor:consumer, setgid 2750 | Publisher writes; consumer group may traverse/read, not write |
| Release directory | governor:consumer, 0750 | Private to publisher and approved consumers |
| Evidence and certificate | governor:consumer, 0640 | Consumer group read-only |
| Trusted public-key policy | controlled independently of producer | Separate administrator sets key allowlist and minimum sequence |

The trusted release root must be pre-provisioned with an appropriate group and ancestor-directory traversal permissions. The reference implementation checks release-root ownership/mode, signing-key ownership/mode, allowed evidence filenames and symlinked final file components. It **does not create** Linux users, mount namespaces, access-control policy, Kubernetes workloads, a KMS or mandatory application routing.

Enforce separate OS UIDs, service-account privileges and storage mounts before using the system in a real deployment. In particular, the producer must have no path to the signer secret or trusted release tree; consumers must receive signed output via the gateway, not simply read raw producer files.

## Limits and residual hazards

- A policy file that the producer can replace destroys the public-key trust boundary. A signature is no stronger than the *external* policy that pins its signer.
- The publisher accepts a caller-provided sequence number; this does not create a trusted distributed monotonic counter. Replay prevention requires separately administered, persistent minimum-sequence state and key revocation.
- The Ed25519 certificate binds a verified *result*, not an attestation of the C++ binary, GPU or host. Compromised governor keys and compromised verification code remain out of scope.
- Copying source evidence into private staging and consuming the checked bytes narrows TOCTOU races, but does not prove power-loss durability, secure kernel behaviour, distributed transactional integrity, or immunity to arbitrary privilege escalation.
- The replay verifier currently supports only the documented fixed genesis, no arbitrary resumed/input workloads. Exceeding its proof budget produces UNKNOWN, never a claimed success.
- This repository cannot guarantee that unrelated consumers bypassing the gateway are blocked; application routing and operating-system ACLs must enforce that.
- There is no proof here of sustained 1000 GB/s, 1000x acceleration, indefinitely unattended operation, or universal freedom from self-derailment.

## Next verification milestones

Perform *real* cross-UID access-denial tests, KMS/HSM-backed key custody, external sequence allocation, mandatory API ingestion, immutable/versioned storage and fault-injected crash/restart model checking. Treat each as UNKNOWN until supported by deployment-specific evidence.

**Verdict:** v2.3 demonstrates asymmetric evidence-bound releases and fail-closed, public-key-pinned consumer verification. The known coherent-evidence forgery is rejected. Global self-derailment containment is still conditional on independently protected governance.
