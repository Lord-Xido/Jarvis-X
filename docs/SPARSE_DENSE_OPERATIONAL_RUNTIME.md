# Jarvis-X Sparse–Dense Operational Runtime

Status: **reference integration candidate**

This integration closes the previously separate execution paths between the
sparse `1000 x 1000` Dr Moagi multiparallel scheduler and the merged dense
volumetric Torch backend.

## Authority law

The dense backend is never authoritative. The operational path is:

```text
sparse authoritative surface
  -> deterministic tile grouping
  -> bounded dense tile gather
  -> dense backend candidate transform
  -> finite/shape verification
  -> scatter onto already-authoritative sparse coordinates
  -> numerical distortion gate
  -> optional epistemic evidence gate
  -> optional authority validator
  -> COMMIT or ROLLBACK
  -> deterministic candidate receipt
  -> re-enter / converge or stop at a hard round ceiling
```

A rejected candidate cannot mutate the sparse surface.

## Two distinct geometries

The integration deliberately distinguishes:

- `L = [1000] x [1000] x [K]`: sparse computational geometry, where X/Y
  identify logical processing loops and K is recursive AE/AD depth;
- `T = [C] x [D] x [H] x [W]`: a bounded dense numerical tile presented to a
  backend such as the Torch volumetric engine.

The default bridge places each sparse loop vector on the centre depth plane of
a dense tile. This is a computational embedding only. Sparse recursive depth is
not equated with physical world Z.

## Shared receipt

Every bridge transaction emits a deterministic receipt containing:

```text
parent_state_hash
candidate_state_hash
after_state_hash
objective_before
candidate_objective
objective_after
hard_constraints
verification = (integrity, numerical, epistemic, authority)
COMMIT | ROLLBACK
receipt_hash
```

The receipt enforces:

```text
COMMIT   => after_hash == candidate_hash
ROLLBACK => after_hash == parent_hash
```

Integrity is not external truth. Numerical convergence is not epistemic
verification. Epistemic verification is explicit and remains caller supplied
until a concrete evidence/CTR service is bound to this integration surface.

## Fixed-point telemetry

Each external candidate records:

```text
state_delta_rms
candidate_mse
global_core_delta_rms
```

The bounded recursive bridge declares local convergence only when both:

```text
state_delta_rms <= epsilon
global_core_delta_rms <= epsilon
```

Otherwise it continues only to the configured finite `max_rounds` ceiling.

## Torch adapter

`apps/dr-moagi-volumetric-torch/sparse_bridge.py` maps a bounded bridge tile to
`VolumetricMultimodalInterpreter`:

```text
DenseTile[C,D,H,W]
  -> torch.Tensor[1,C,D,H,W]
  -> recursive 3D encode/decode refinement
  -> candidate Tensor
  -> DenseTileResult
  -> sparse scatter
  -> verification
```

The current merged Torch outer shell has `C=16`, and the tile side/depth must
match the model `base_res`.

Run the integration reference:

```bash
cd apps/dr-moagi-volumetric-torch
python sparse_bridge.py
```

## Capability boundary

This closes a software integration path. It does not claim that one million
neural networks are simultaneously resident, that sparse recursion depth is a
physical coordinate, that the untrained Torch weights provide semantic truth,
or that convergence proves correctness in the external world. GPU performance
and rate-distortion superiority remain empirical benchmark questions.
