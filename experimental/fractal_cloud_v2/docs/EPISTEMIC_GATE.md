# JARVIS-X Ω³ — Epistemic Gate v2.1

An **external Python safety supervisor** for the existing deterministic C++17 3D fractal engine. It enforces explicit `TRUE`, `FALSE`, and `UNKNOWN` states without changing the locked v1.0 mathematics or the existing v2.0 binary runtime.

| Verdict | When | Behavior |
|---|---|---|
| `TRUE` | Complete, internally consistent evidence and independently recomputed final checkpoint hash | Publish verified output and permit downstream commit |
| `FALSE` | Runtime rollback, arithmetic rejection, trace inconsistency or tampered state | Immediately reject, quarantine outputs and stop; exit 2 |
| `UNKNOWN` | Runtime timeout, abnormal exit or missing evidence | Retry at most N times; then safe halt; exit 3 |

The supervisor constrains attempts to 1–5, deadline to ≤600 seconds each; on timeout it kills the spawned process group. Unverified outputs are never promoted into `verified/`. Attempts and their evidence remain available for forensic analysis. An atomic `epistemic_decision.json` records every decision.

```bash
python3 tools/epistemic_gate.py \
  --binary /path/to/jx3d_cloud \
  --golden /path/to/golden_v1_depth2_steps24.csv \
  --output /tmp/jx3d-epistemic \
  --depth 2 --steps 24 --workers 4 \
  --attempts 2 --timeout-seconds 30

python3 tests/test_epistemic_gate.py \
  /path/to/jx3d_cloud /path/to/golden_v1_depth2_steps24.csv
```

The **independent Python checksum reconstruction** uses the binary checkpoint’s coordinates and Q16 field, recomputing the FNV-1a chain terminal digest. A golden trace comparison additionally establishes byte-for-byte reproducibility on the locked depth-2 workload. It does **not** establish cryptographic authenticity, verified end-to-end environmental correctness, or failure independence from the same host. FNV-1a is not a security signature; use SHA-256 authenticated manifests or signatures for adversarial settings.

The watchdog is a separate OS process but shares host, power, storage and OS dependencies. It cannot make indefinite service availability guarantees or recover from total host failures. Production deployment needs independent watchdog placement, durable transactional storage, permission isolation and audit of all externally observable side effects.