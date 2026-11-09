# JARVIS-X Ω³ — v2.2 self-derailment containment (reference implementation)

**Status:** independent deterministic replay and signed artifact publication are implemented and adversarially tested. **Not** a proof of production isolation, binary attestation, long-term availability, or all-consumer access control. The canonical C++ engine, mathematical recurrence and v1 golden trace remain untouched.

## Why this patch exists

The v2.1 epistemic supervisor can accept a counterfeit result when an actor alters one voxel, recomputes the unkeyed FNV digest, and edits `trace.csv` and `run.json` to agree. The verified counterexample changed Q16 from 4946 to 4947 and supplied a consistent forged final hash. Merely checking mutually dependent artifacts cannot establish their authenticity.

## What v2.2 enforces in this reference harness

1. Reconstruct the fixed 3D Menger-like coordinates and initial Q16 field independently in Python, rather than importing C++ state or trusting `run.json` values.
2. Recompute **all** synchronous six-neighbour Q16 transitions from genesis, and check the ordered per-step hash against the supplied trace.
3. Compare the resulting final state and chain parent digest with the checkpoint. Reject a forged internally consistent final state even when a golden trace is omitted.
4. Fail closed if required evidence is missing, nonregular, malformed, oversized, or outside the independent replay's permitted resource budget. `UNKNOWN` permits at most the configured bounded retries, after which the supervisor records `SAFE_HALT_UNKNOWN`.
5. Stage an allowlisted release of `checkpoint.bin`, `trace.csv`, and `run.json`, then sign a certificate binding the artifact SHA-256 values, workload, independent replay method, and random release identifier with HMAC-SHA-256.
6. Require the consumer to rerun `release_verify()` before using the output. An incorrect key or altered release artifact fails. Certificate generation is performed only after verified state replay and copy-hash consistency.
7. Require the HMAC signing key file to be outside the untrusted output tree and have mode `0600`. A public test key is **not** suitable for real deployment.

## Reference launch

```bash
cmake -S experimental/fractal_cloud_v2 -B /tmp/jx3d-build -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/jx3d-build --parallel 2

# In production a separately controlled authority creates and owns this secret,
# not the untrusted C++ runtime or its operating-system user.
python3 - <<'PY'
from pathlib import Path
import secrets
p = Path('/tmp/jx3d-demo-signing.key')
p.write_bytes(secrets.token_bytes(32))
p.chmod(0o600)
PY

python3 experimental/fractal_cloud_v2/tools/containment_v22.py \
  --binary /tmp/jx3d-build/jx3d_cloud \
  --output /tmp/jx3d-contained-demo \
  --key-file /tmp/jx3d-demo-signing.key \
  --golden experimental/fractal_cloud_v2/tests/golden_v1_depth2_steps24.csv \
  --depth 2 --steps 24 --workers 2 --attempts 2 \
  --timeout-seconds 30 --replay-timeout-seconds 60
```

The application must treat `attempt_*/` raw outputs and `run.json.status=COMMITTED` as *untrusted*. The canonical trusted-release namespace is `verified/`, **but this GitHub patch cannot force existing application consumers to obey it**. Consumer-side verification is mandatory before use.

## Verified threat model and limits

| Threat | v2.2 reference behaviour | Residual assumptions |
|---|---|---|
| Altered Q16 final voxel + coherent FNV/trace/report forgery | Rejects on full replay | Fixed genesis and recurrence are trusted and unchanged |
| Counterfeit binary returning forged bundle | Rejects forged numerical result | Does not constrain arbitrary side effects of executable |
| Missing/corrupt evidence | FALSE or UNKNOWN, never publish | Supervisor and its FS APIs execute correctly |
| Hung child process | Bounded termination | Operating-system scheduler/watchdog runs |
| Modified published checkpoint/certificate | Consumer rejects on renewed check | Consumer must call verifier immediately before use; TOCTOU not eliminated |
| Producer obtains signing secret | **Not contained** | Separate OS identity / KMS / privileged signing service required |
| Producer changes verifier or publication routing | **Not contained** | Protected deployment and mandatory gateway required |
| Arbitrary data inputs or resumed checkpoints | **Not supported by this verifier** | Different initial condition requires independently authenticated input/replay model |
| Large recursive workloads | UNKNOWN if `cells×max(1,steps) > 6,000,000` | Proof budget limits reference deployment |
| Crash during publication, disk loss, network partition | **Not proven** | `os.replace` does not imply fsync or cross-machine durability |

The Python verifier is independent implementation logic but is **not** an independent fault domain on a shared host. HMAC authenticates the governor's release certificate only while the key remains secret; it does not authenticate the original C++ executable. Use an isolated verification principal with protected policy, credential storage, and immutable output storage in any production deployment.

## Reproducibility

```bash
python3 experimental/fractal_cloud_v2/tests/test_containment_v22.py \
  /tmp/jx3d-build/jx3d_cloud \
  experimental/fractal_cloud_v2/tests/golden_v1_depth2_steps24.csv
```

16 adversarial tests are included. A local depth-four, 24-iteration independent replay (160,000 cells) and post-release HMAC verification also completed successfully, but the roughly 24.5-second total is a development-environment measurement and not a certified cloud performance figure.

## Next production assurance gates

1. Run the untrusted engine and verifier under separate OS users or isolated nodes; ensure neither untrusted process nor build job can read/modify policy or signing credentials.
2. Require a separately administered ingress/gateway: raw engine `COMMITTED` is never accepted as proof of authorized release.
3. Use an identity-protected key service or asymmetric signing with trust-anchored public keys; rotate credentials and support revocation.
4. Publish to immutable/versioned object storage and verify artifacts at read time. Harden against symlink/hardlink races and TOCTOU issues.
5. Extend independently replayed and authenticated genesis models to resumed checkpoints and new data. Require explicit, authenticated workload manifests.
6. Model governor, publisher, crash, restart and privilege assumptions with temporal logic; model-check finite state spaces and prove the permitted transitions. CI passing does not prove all reachable operating-system states.

**Security claim is scoped:** v2.2 blocks the known v2.1 coherent-evidence forgery for this deterministic fixed-genesis workload and makes release modification detectable under a secret-key assumption. No unconditional proof of global autonomous safety is claimed.
