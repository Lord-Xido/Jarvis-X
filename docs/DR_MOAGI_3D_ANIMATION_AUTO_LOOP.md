# Dr Moagi 3D Animation Auto-Execution Loop

This layer turns the verified DM3D auto-encoding/decoding program into a finite cyclic
3D state machine suitable for animation generation and downstream rendering.

## Execution model

The outer bytecode contains a verified inner DM3D program and bounded loop metadata:

```text
DM3D-LOOP
  cycles
  feedback register
  convergence scalar register
  frame-register mask
  inner DM3D bytecode
  SHA-256 digest
```

The inner canonical program remains:

```text
X_t
  -> ENCODE_3D       -> Z_t^0
  -> REFINE_3D^K     -> Z_t^*
  -> DECODE_3D       -> X_hat_t
  -> REENCODE_3D     -> Z_tilde_t
  -> ERROR_3D
  -> CYCLE_ERROR
  -> ERROR_MEMORY
  -> CHECK_FIXED
  -> HALT
```

The outer executor then applies the explicit feedback edge:

```text
X_(t+1) <- X_hat_t
```

and repeats only up to the encoded cycle bound.

Mathematically:

```text
Z_t^0       = E(X_t)
Z_t^*       = R^K(Z_t^0)
X_hat_t     = D(Z_t^*)
Z_tilde_t   = E(X_hat_t)
Omega_(t+1) = rho Omega_t + (1-rho)(lambda_x e_x + lambda_z e_z)
X_(t+1)     = X_hat_t
```

where the current deterministic block-average / replication codec satisfies the latent
cycle identity to floating-point precision:

```text
E(D(Z_t^*)) ~= Z_t^*
```

## Animation frames

The canonical loop captures registers `r0..r4` on each cycle:

| Register | Stage |
| --- | --- |
| `r0` | input field |
| `r1` | encoded latent |
| `r2` | inward-refined latent |
| `r3` | decoded reconstruction |
| `r4` | re-encoded latent |

Each frame carries the complete `Volume3D` state for a downstream renderer and reports
shape, minimum, maximum, mean, and a SHA-256 state digest. Rendering is deliberately
kept outside the VM so the bytecode remains deterministic and graphics-backend neutral.

## Safety and bounded execution

The auto-loop format does not provide general jumps, shell access, dynamic imports, or
host-code execution. It can only repeat a previously verified finite DM3D program.
Execution is constrained by three independent bounds:

1. per-cycle `ProgramLimits`, including instruction, voxel, refinement and physical-step
   caps;
2. `AutoLoopLimits.max_cycles`;
3. `AutoLoopLimits.max_total_physical_steps` and `max_frames` across all cycles.

A PDF remains a transport container only. Opening the PDF does not trigger this loop.
The Jarvis-X runtime must explicitly extract and verify the bytecode before execution.

## Reference use

```python
from jarvisx.dr_moagi_animation_loop import (
    canonical_animation_loop_program,
    execute_auto_loop,
)
from jarvisx.dr_moagi_pdf_bytecode import make_seed_volume

program = canonical_animation_loop_program(cycles=4, refinement_passes=6)
result = execute_auto_loop(program, make_seed_volume(16))
print(result.report())
```

For interactive or video output, a renderer can consume `result.frames` in order and
map each volumetric state to points, voxels, meshes, SDF surfaces, or GPU textures.
Measured physical VM work and rendered-frame throughput should be reported separately.
