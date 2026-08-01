# Jarvis-X Empirical Validation

This document consolidates the executable evidence for canonical Jarvis-X capabilities. The evidence gate is deliberately narrower than the project vision: every row names a falsifiable software claim, an observable protocol, a pass criterion and a boundary that the result does **not** establish.

## Run the evidence gate

```bash
python -m pip install -e ".[test]"
python -m jarvisx.empirical_validation \
  --repetitions 64 \
  --octree-max-depth 6 \
  --output artifacts/empirical-validation.json
```

The command exits with status `0` only when every required check passes. The JSON artifact records the commit, interpreter, platform, protocol, observations and boundaries. GitHub Actions uploads the report from each validation run.

## Consolidated evidence matrix

| Capability claim | Observable protocol | Required result | Canonical implementation | Boundary |
|---|---|---|---|---|
| Deterministic VM replay | Assemble one fixed five-word program and run it in 64 fresh VMs | Complete final states and traces are identical; all ledgers verify | `src/jarvisx/core.py`, parser, assembler, decoder and executor | Exercises the implemented ISA path; it is not a general performance or intelligence claim |
| Ω journal integrity | Build a deterministic three-entry hash chain, then mutate the middle state | Valid chain passes before mutation and fails afterward | `src/jarvisx/ledger.py` | Integrity is not confidentiality, trusted time or remote replication |
| Sparse transactional field | Reverse observation insertion order, replay a checkpoint and force an invalid candidate | State/journal digests remain equal; checkpoint restores; rejected persistent state rolls back | `src/jarvisx/dr_moagi_billion_field.py` | `1000³` is virtual address-space extent; active coordinates alone are materialized |
| Recursive octree geometry | Materialize depths 0–6 and compare measured metrics with closed forms | Nodes `(4^(D+1)-1)/3`, leaves `4^D`, volume `2^-D` at every depth | `src/jarvisx/fractal_octree.py` | Geometric self-similarity does not by itself prove long-memory quality |
| Fractional numerical invariants | Smooth a `4×4×4` impulse and compare split versus combined spectral steps | Mass drift `<1e-9`; variance and gradient energy decrease; semigroup error `<1e-10`; trace is canonical | `src/jarvisx/fractional_smoothing_3d.py` | Small-grid direct-DFT correctness is not production FFT performance or calibrated physics |

## Machine-readable report

The report schema identifier is:

```text
jarvisx.empirical-validation.v1
```

Every check contains:

- `claim`: the exact proposition being tested;
- `protocol`: the executed experiment;
- `passed`: the boolean gate result;
- `metrics`: raw observations and deterministic digests;
- `boundary`: the inference limit.

Timing fields are reported as observations only. No fixed throughput threshold is used because shared CI runners are not controlled benchmark hardware.

## Interpretation

A green empirical-validation workflow establishes that the checked implementation satisfies these bounded invariants on the recorded environment. It does not establish:

- artificial general intelligence, consciousness or subjective agency;
- unlimited or lossless memory;
- superiority over transformer, state-space or retrieval architectures;
- production security certification;
- physical realization of a virtual lattice;
- cross-platform floating-point bit identity;
- statistically significant model-quality improvement.

Claims outside the evidence matrix remain specifications, hypotheses or future experiments until a protocol, baseline, dataset and reproducible result are added.

## Extending the evidence gate

A new claim should be promoted only with:

1. a falsifiable statement;
2. an executable protocol;
3. a documented baseline or closed-form reference;
4. explicit pass/fail tolerances;
5. deterministic inputs or a declared stochastic seed protocol;
6. a machine-readable artifact;
7. an inference boundary preventing overstatement.

Performance comparisons should additionally include warm-up policy, sample count, hardware and software environment, uncertainty estimates and a named baseline implementation.
