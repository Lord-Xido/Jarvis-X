# JARVIS-X Computational PDF Runtime v1.0 — provenance and verification

2026-11-28. Isolated reference implementation, not a replacement for the main engine.

The six runtime/source/receipt files are byte-identical Git blobs from the locally tested PDF companion archive. The standalone HTML Evidence Lab, PDF and ZIP themselves are not included in this commit. Their local provenance is retained below:

| Artifact | Digest |
|---|---|
| Computational Runtime PDF SHA-256 | 5be314021ce90ef74c95856205749caf13d93a1590c30052e76d0d53c463fbdc |
| Runtime Source ZIP SHA-256 | 8ffd1dd085d09f7cf5140adc92be664e1047520c70c8a4d7f0e11aee153f6052 |
| HTML workspace Git blob SHA-1 | cd85a3bc6c34958f58ec00348f92b848c92696d7 |

## Reproduce

From this directory, run `python3 selftest.py` with Python 3.9+ and g++ (C++17). The test emits a new receipt; the checked-in receipt records the original local execution. Timings and zlib versions are environment dependent. Local results: 11 test groups, 1000 address and codec round trips each, 32 native cross-checks, 6 injected fault rejections. A repeat local run also passed.

## Architectural boundaries

- Logical 90-bit Morton addressing; actual processing is streamed in independent 512-byte tiles, no dense virtual allocation.
- Causal integer neural predictor with reversible XOR residual; CTR accepts exact candidates no larger than the baseline compressed payload.
- Python host implements file streaming, compression and integrity; C++17 implements only a bounded tile codec and canonical bytecode interpreter.
- No semantic multimedia generation, cross-tile ANN communication, GPU/distributed execution, encryption/authentication, cloud synchronization, autonomous code evolution, or UI integration.
- The PDF is a carrier of source attachments, not an execution environment.
- Do not interpret prior local receipts as GitHub Actions results.
