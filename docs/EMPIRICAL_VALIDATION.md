# Jarvis-X Empirical Validation

This document defines the executable evidence gate for canonical Jarvis-X capabilities. The gate is deliberately narrower than the project vision: every check names a falsifiable software claim, an observable protocol, a pass criterion and a boundary that the result does **not** establish.

## Run the evidence gate

```bash
python -m pip install -e ".[test]"
python -m jarvisx.empirical_validation_v2 \
  --repetitions 64 \
  --octree-max-depth 6 \
  --output artifacts/empirical-validation.json
```

The command exits with status `0` only when every required check passes. The JSON artifact records the commit, interpreter, platform, protocol, observations and inference boundaries. GitHub Actions validates and uploads the report from each evidence run.

## Evidence architecture

Version 2 preserves every v1 core check and adds system-wide adaptive checks. The governing rule is:

```text
parent state
  -> bounded candidate
  -> hard constraints / resource projection
  -> objective improvement gate
  -> COMMIT or ROLLBACK
  -> deterministic receipt / evidence
```

Hard constraints are evaluated separately from the scalar objective. An optimizer cannot compensate for a failed resource, policy or integrity gate by obtaining a better score.

## Consolidated evidence matrix

| # | Capability claim | Observable protocol | Required result | Canonical implementation | Boundary |
|---:|---|---|---|---|---|
| 1 | Deterministic VM replay | Assemble one fixed five-word program and run it in fresh VMs | Complete final states and traces are identical; all ledgers verify | `core.py`, parser, assembler, decoder, executor | Exercises the implemented ISA path; not a general performance or intelligence claim |
| 2 | Ω journal integrity | Build a deterministic three-entry hash chain, then mutate historical state | Valid chain passes before mutation and fails afterward | `ledger.py` | Integrity is not confidentiality, trusted time or remote replication |
| 3 | Sparse transactional field | Reverse insertion order, replay a checkpoint and force an invalid persistent candidate | State/journal digests remain equal; checkpoint restores; rejected state rolls back | `dr_moagi_billion_field.py` | `1000³` is virtual address extent; active coordinates alone are materialized |
| 4 | Recursive octree geometry | Materialize bounded depths and compare with exact recursive closed forms | Node, leaf and retained-volume formulas match at every tested depth | `fractal_octree.py` | Geometric self-similarity does not prove long-memory quality |
| 5 | Fractional numerical invariants | Smooth a small 3D impulse and compare split versus combined spectral steps | Mass, dissipation, semigroup and trace invariants pass declared tolerances | `fractional_smoothing_3d.py` | Small-grid direct-DFT correctness is not production FFT performance or calibrated physics |
| 6 | Shared candidate admission | Evaluate an admissible improving candidate and a better-scoring candidate that violates a hard constraint | Admissible candidate commits; hard-constraint candidate rolls back; receipts verify and replay identically | `candidate_contract.py` | Establishes the common admission primitive, not migration of every research runtime |
| 7 | Field Runtime transaction | Run identical sparse field steps from opposite insertion orders, then force validator rejection | Identical state/metrics, bounded support and exact rollback | `dr_moagi_field_runtime.py` | Identity-codec fixture does not prove arbitrary learned-codec stability |
| 8 | DM-DD atomic adaptation | Execute identical residual-learning steps, then reject an adaptive proposal | State, Ω memory and Θ parameters evolve deterministically and roll back as one tuple on rejection | `dr_moagi_deep_distiller.py` | Tests the bounded scalar-gain reference learner, not large-model quality |
| 9 | Virtual 3D optimizer admission | Run bounded α/β search twice and adapt the result into the common candidate receipt | Same selected result and receipt; non-regressive score; shared decision agreement; zero terminal reality gap | `dr_moagi_virtual_3d_ae.py`, `candidate_adapters.py` | Bounded scalar search is not unrestricted self-modification or gradient-trained representation learning |
| 10 | Orthogonal quantization precision | Verify an orthonormal DCT-II basis and nearest-neighbour quantization reconstruction | Orthogonality passes and spatial residual remains within the deterministic L2 envelope | `orthogonal_quantization.py` | Transform correctness is not a production codec or perceptual benchmark |

## Machine-readable report

The report schema identifier is:

```text
jarvisx.empirical-validation.v2
```

Every check contains:

- `claim`: exact proposition being tested;
- `protocol`: executed experiment;
- `passed`: boolean gate result;
- `metrics`: observations and deterministic digests;
- `boundary`: explicit inference limit.

Candidate-admission receipts additionally contain parent/candidate state hashes, operator identity, objective before/after, component metrics, hard-constraint outcomes, resource envelope/usage, decision, rejection reasons and a deterministic SHA-256 receipt hash.

Timing fields remain observations only. Shared CI runners are not controlled benchmark hardware, so the evidence gate does not impose portable throughput claims.

## Interpretation

A green empirical-validation workflow establishes that the checked implementation satisfies these bounded invariants on the recorded environment. It does not establish:

- artificial general intelligence, consciousness or subjective agency;
- unlimited or lossless memory;
- superiority over transformer, state-space or retrieval architectures;
- production security certification;
- physical realization of a virtual lattice;
- cross-platform floating-point bit identity;
- statistically significant model-quality superiority;
- safety merely because a policy or coherence gate exists.

Claims outside the evidence matrix remain specifications, hypotheses or future experiments until a protocol, baseline, dataset and reproducible result are added.

## Extending the evidence gate

A new claim should be promoted only with:

1. a falsifiable statement;
2. an executable protocol;
3. a documented baseline, invariant or closed-form reference;
4. explicit pass/fail tolerances;
5. deterministic inputs or a declared stochastic seed protocol;
6. a machine-readable artifact;
7. an inference boundary preventing overstatement;
8. a shared candidate receipt when the subsystem can modify authoritative adaptive state.

Performance comparisons should additionally include warm-up policy, sample count, hardware/software environment, uncertainty estimates and a named baseline implementation.
