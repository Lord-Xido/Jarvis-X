# DM-IMP Commercialization Orchestration Contract

## Purpose

This document is the canonical execution contract for converting the Dr Moagi Intelligence Media Processor (DM-IMP) from an operational reference architecture into defensible, reproducible, licensable processor IP.

The contract does **not** treat "beyond SOTA" as a blanket architectural label. Every claim must be attached to a named metric, workload, baseline, hardware/software configuration and reproducible result.

## Control law

Execution order is:

```text
Working -> Robust -> Portable -> Elegant -> Advanced
```

Commercial promotion follows the same candidate-first law as the runtime:

```text
claim/proposal -> shadow evidence -> validation gate -> commit or rollback
```

No claim, benchmark, hardware result or commercial representation advances without its corresponding evidence gate.

## Canonical system decomposition

DM-IMP commercialization is treated as one closed system with six coupled workstreams:

```text
IP / provenance -----------+
                           |
Benchmark evidence --------+----> verified technical position
                           |                |
Formal verification -------+                v
                                        Hardware proof
                                             |
                                             v
                                       Codec integration
                                             |
                                             v
                                     Licensing / design win
                                             |
                                             v
                                           Royalties
```

GitHub control issues:

- #176 — orchestration epic
- #177 — IP/provenance, prior art and claim families
- #178 — reproducible SOTA benchmark harness
- #179 — FPGA/RTL proof of VCL-BVM-8
- #180 — codec-aware media adapters
- #181 — formal semantics, fuzzing and crash consistency
- #182 — licensing package, royalty model and design-win pipeline

## Evidence states

Every technical or commercial assertion MUST occupy one of these states:

1. `proposed` — architectural hypothesis or design target.
2. `implemented` — code or hardware artifact exists.
3. `tested` — deterministic tests or conformance checks pass.
4. `benchmarked` — reproducible measurements exist against named baselines.
5. `externally_reproduced` — a third party has reproduced the result.
6. `commercially_qualified` — the result has entered an evaluation/design-in process.
7. `licensed` — executed agreement and royalty/reporting obligations exist.

Promotion may only move forward one evidence class when the required artifacts exist. Failed gates roll back the claim state rather than being reworded as success.

## Claim-to-evidence matrix

| Claim family | Required evidence before external assertion |
| --- | --- |
| Sparse global 3D substrate + local VCL tiles | address-space correctness, memory behavior, throughput and memory-traffic comparison |
| VCL-BVM-8 ISA | deterministic semantics, conformance tests, malformed-bytecode safety, execution-cost measurements |
| Inward `512 -> 8 -> 512` transform | reconstruction metrics, compression/state-retention measurements, matched baseline comparison |
| Transactional Theta/Omega adaptation | candidate/commit/rollback tests, adaptation-overhead measurements, non-regression evidence |
| Entropy/mask pruning | actual information metric definition, ablation against no-prune baseline, quality/compute trade-off |
| Sub-nanosecond signaling | post-place-and-route timing on named hardware; simulation alone is insufficient |
| Beyond-SOTA performance | reproducible superiority on a named metric/workload/baseline; never a global label |
| Patentability / novelty | counsel-reviewed prior-art and jurisdiction-specific claim analysis |

## Technical invariants

The following invariants permeate implementation, benchmarking and licensing diligence:

1. **Authoritative state is not mutated before validation.**
2. **Theta and Omega are rolled back together when Lambda rejects a candidate.**
3. **Benchmark inputs, quality targets and baseline conditions are matched.**
4. **Codec loss and DM-IMP transform loss are measured separately.**
5. **Sub-nano claims require hardware timing evidence.**
6. **No deterministic reference operator is represented as a trained frontier model without training/evaluation evidence.**
7. **The 10^36 coordinate domain is addressability, not physical allocation.**
8. **The literal 10 GB executable overlay is not represented as active runtime state unless the runtime actually maps and uses it.**
9. **Security, formal verification and fuzzing results state bounded assumptions explicitly.**
10. **Commercial terms do not expand technical claims beyond the evidence package.**

## Benchmark promotion gate

Any performance claim must ship with:

```text
metric
workload
input corpus
baseline implementation
hardware
OS/runtime/compiler versions
quality target
cold/warm methodology
sample count
raw measurements
summary statistics
reproduction command
commit SHA
```

Required metrics include p50/p95/p99 latency, throughput, resident/peak memory, bytes moved, reconstruction/semantic quality, adaptation overhead, rollback rate and energy or a defensible compute/power proxy.

## Hardware promotion gate

VCL-BVM-8 hardware claims require:

```text
RTL commit
reference-model conformance corpus
synthesis tool/version
part/device
clock constraints
post-route Fmax
worst negative slack
critical path
pipeline depth
LUT/ALM + FF + BRAM/SRAM + DSP usage
power estimate/measurement
throughput per tile
latency per tile
```

A sub-nanosecond claim is valid only for the exact local operation whose verified timing is below 1 ns.

## IP promotion gate

The IP workstream must maintain a provenance table covering at least:

- first conception/date evidence;
- repository commits and public disclosures;
- DM-IMP architecture;
- VCL-BVM-8;
- JX3DVM1 integration;
- transactional Theta/Omega adaptation;
- inward spatial media transform;
- persistent-state/reservoir concepts.

Known primitives must be separated from potentially differentiating combinations. Repository originality alone is not evidence of legal novelty.

## Licensing promotion gate

A technology may enter external evaluation only when the data room can provide the subset appropriate to the agreement:

- architecture brief;
- provenance/claim map suitable for counsel review;
- benchmark report and raw-data reproduction path;
- verification/security report;
- evaluation executable/SDK or hardware image;
- supported integration boundary;
- known limitations;
- proposed field of use;
- royalty/economic model;
- audit/reporting requirements.

Preferred commercial structure is ownership retention with field-of-use licensing, measurable milestones, minimum commitments where appropriate and auditable royalty reporting. Blanket exclusivity requires explicit economic justification.

## Royalty operating model

Commercial progress is tracked as:

```text
target
-> qualified
-> NDA/evaluation
-> benchmark reproduced
-> integration/design-in
-> commercial proposal
-> license executed
-> production volume
-> royalty report
-> audit/reconciliation
```

Royalty models may include:

```text
upfront fee + NRE + per-unit royalty
upfront fee + percentage of licensed revenue
per-workload / cloud-usage royalty
minimum annual commitment + running royalty
```

No royalty rate is treated as canonical until technical advantage, claim strength, substitutability, addressable volume and comparable licensing economics have been evaluated.

## Fixed-point commercialization criterion

The commercialization system reaches its operational fixed point only when evidence, implementation and commercial representation agree:

```text
technical claim == reproducible evidence == licensed representation
```

If any one of these diverges, the system is not at a verified fixed point and must cycle inward through measurement, correction and re-validation.

## Current anchor

Initial DM-IMP mainline integration anchor:

```text
62d0cd8e0cfb28e452645a86fa43de709004a416
```

This document is the system-wide orchestration specification. The child issues hold executable work; this document defines the invariants they must all obey.
