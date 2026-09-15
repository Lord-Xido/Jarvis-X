# Dr Moagi Recursive 3D Geometry Optimizer

`jarvisx.dr_moagi_recursive_geometry_optimizer` turns the sparse `1000^3` runtime inward onto its **bounded configuration geometry**. It does not rewrite Python source and it does not claim infinite performance.

## Meta-geometry

The optimizer represents the runtime by

```text
g = (g_spatial, g_representation, g_dynamics) in [-1, 1]^3
```

and decodes the three axes into the existing `DrMoagi3D1000xEngine` controls:

- **spatial** -> logarithmic active-tile budget;
- **representation** -> latent dimension plus deterministic encoder-basis seed;
- **dynamics** -> Omega memory decay and residual-correction gain.

The operational recursion is

```text
g_t
 -> decode(g_t)
 -> multimodal ingest
 -> E
 -> Z
 -> Omega
 -> D
 -> residual
 -> correction
 -> scheduling
 -> benchmark
 -> admissibility gate
 -> commit OR rollback
 -> g_(t+1)
 -> recur
```

## Search geometry

Each generation evaluates a bounded three-dimensional shell around the incumbent. The shell contains the 26 normalized neighbors of a cube plus 12 helical/non-Cartesian directions. The latter reduce axis locking during repeated contraction.

Successful generations expand the radius slightly. Rejected generations retain the incumbent and contract the radius.

```text
improvement -> commit -> radius *= 1.08
no improvement -> rollback -> radius *= 0.55
```

`--forever` means repeat the search/measure/commit-or-rollback protocol until interrupted. It does **not** mean unbounded or infinite performance.

## Benchmark suite

The reference evaluator constructs deterministic byte workloads representing:

- text;
- Python-like source code;
- image-like spatial gradients;
- audio-like waveforms;
- video-like temporally shifted frames;
- normalized 3D point-cloud data;
- high-entropy binary data.

Every candidate runs the same `DrMoagi3D1000xEngine` encode/Omega/decode/residual/correction/scheduling cycle.

## Internal objective

The internal rank is intentionally stable under CI timing noise:

```text
J = 0.72 * normalized_residual + 0.28 * normalized_active_work
```

Latency and p95 latency are measured and reported, but wall-clock timing is not used to promote candidates inside the deterministic reference search. Production performance promotion should use controlled hardware-specific benchmark infrastructure.

Candidates that materially regress reconstruction error are inadmissible even when they reduce active work.

## External SOTA gate

The optimizer never infers that an internal improvement is state of the art. A `beyond_external_baseline` result is possible only when a caller supplies a matched external baseline with thresholds from the **same workload, hardware, numerical precision, warm-up, repetitions and output-quality contract**.

Example:

```json
{
  "name": "matched-production-baseline",
  "max_residual_mse": 0.01,
  "max_median_latency_ms": 2.5,
  "max_active_voxels": 1000000,
  "minimum_work_reduction": 1000.0
}
```

Every enabled threshold must pass. Without this file, the report remains:

```text
external_sota_baseline_supplied = false
external_sota_gate_passed = false
```

This is deliberate. A local recursive search is evidence of internal optimization, not evidence of superiority to vLLM, SGLang, TensorRT-LLM, sparse-convolution runtimes, or any other external system.

## Run

Install the acceleration extra:

```bash
python -m pip install -e ".[accel]"
```

Run a finite search:

```bash
jarvisx-3d-optimize --generations 4 --out recursive-report.json
```

Run indefinite reiteration until interrupted:

```bash
jarvisx-3d-optimize --forever --out recursive-report.json
```

Run with an external matched baseline:

```bash
jarvisx-3d-optimize \
  --baseline examples/sota_baseline_template.json \
  --generations 4 \
  --out recursive-report.json
```

The template contains deliberately impossible zero thresholds and must be replaced with real measured values before it can pass.

## Relationship to the canonical meta-optimizer on `main`

The current PR branch predates the newer canonical `dr_moagi_meta_optimizer.py` on `main`. This module is therefore scoped specifically to the sparse `DrMoagi3D1000xEngine` and uses a distinct module/API name. When the PR branch is rebased, its recursive geometry protocol should compose with the canonical meta-optimization layer rather than replace it.

## Engineering boundary

The implementation optimizes configuration/manifold state only. It does not:

- modify its own Python source;
- execute generated host commands;
- bypass tests or benchmark gates;
- promote a candidate merely because it is newer;
- treat a virtual-space or sparse-work ratio as measured wall-clock speedup;
- assert SOTA status without matched external evidence.

The intended invariant is:

```text
Generate -> Evaluate -> Contrast -> Gate -> Commit/Rollback -> Recur
```
