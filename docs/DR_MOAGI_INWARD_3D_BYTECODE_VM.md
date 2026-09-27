# Dr Moagi Inward 3D Spatial Bytecode VM

## Status

This is the executable spatial-bytecode bridge for the bounded Jarvis-X inward runtime. It maps the 8x8 input bus (64 scalar channels) into a 64-bit, explicitly 3D-addressed instruction stream and couples that stream to the existing GeometricBytecodeANN fixed-point engine.

It is an internal research runtime. Internal score improvements are not evidence of state-of-the-art performance; external SOTA claims require matched benchmark data and independent comparison.

## 1. 64-bit instruction format

Each instruction uses OP8 | FLAGS8 | X12 | Y12 | Z12 | ARG12.

The spatial pointer is P_t = (x_t, y_t, z_t), with each coordinate in [0, 4095].

The v1 z-axis regions are:

- input/snapshot: z=0..10
- diffusion/encoder: z=120..240
- latent fixed-point core: z=500..580
- decoder: z=720
- residual/error/correction: z=820..880
- geometry/commit: z=900..930
- shadow meta-optimization: z=940..970
- output: z=980
- halt: z=999

After a meta decision, the route points back to (0,0,0), closing the operational loop.

## 2. Inner state loop

The frame program is:

    STATE_SNAPSHOT
      -> INGEST
      -> DIFFUSE6
      -> ENCODE3D
      -> PHI3D_ITER
      -> FP_CHECK
      -> OMEGA_MEMORY
      -> DECODE3D
      -> RESIDUAL
      -> ERROR_FIELD
      -> CORRECT
      -> ADAPT
      -> GEOM_MAP
      -> RECUR
      -> OUTPUT_BUS
      -> HALT

The latent recurrence remains bounded by the existing normalized latent row sum. With row sum 0.72 and relaxation r, the control-plane certificate is:

q = (1-r) + 0.72r = 1 - 0.28r < 1 for r > 0.

The normalized convergence radius is:

R_t = min(1, fixed_point_residual / fixed_point_tolerance).

Thus R_t approaching zero corresponds to stronger fixed-point convergence.

## 3. Residual field

After decode, the residual e = x - x_hat is ranked by magnitude and projected back into the 4x4x4 latent cube. The runtime records the highest-error voxel coordinates as an explicit 3D error field.

## 4. Shadow self-optimization

The middle loop searches a bounded 3D policy neighbourhood with coordinates (refinement, memory, correction), each in {-1, 0, +1}.

The axes modify:

- refinement: latent relaxation and maximum fixed-point iterations
- memory: residual-memory coefficient
- correction: residual correction gain and diffusion strength

Every candidate runs inside a separate GeometricBytecodeANN loaded with the same authoritative source bus. Candidate evaluation never executes on the production ANN.

The internal score is:

S = 8*MSE + 2*fixed_point_residual + 0.002*refine_steps + 0.0005*active_voxels.

Promotion requires finite metrics, contraction bound below one, unchanged authoritative production state during shadow replay, the configured minimum score improvement, and MSE inside the allowed regression envelope. Otherwise the meta instruction is ROLLBACK.

## 5. Nested runtime clocks

The repository now exposes three related levels:

1. Spatial bytecode/state loop: dr_moagi_inward_3d_vm.py
2. Runtime configuration loop: this module plus dr_moagi_meta_optimizer.py
3. Architecture/orchestration loop: dr_moagi_system_evolution.py

The spatial VM does not replace the outer architecture controller. It supplies the missing bytecode-addressed execution plane underneath it.

## 6. CLI

Examples:

    jarvisx-inward3d-vm --cycles 4 --meta-interval 2
    jarvisx-inward3d-vm --cycles 2 --meta-interval 1 --disassemble --json

JSON output includes the active ANN configuration, per-frame reconstruction and fixed-point telemetry, contraction certificate, normalized convergence radius, high-error 3D voxels, shadow candidate evaluation and promotion decision, and explicit external_sota_unverified claim status.

## 7. Verification

Targeted tests cover instruction round-trip, coordinate bounds, traversal across input/latent/verification/output regions, finite fixed-point telemetry, contraction bound below one, production-state isolation during shadow search, return routing to the 3D origin, and repeated autonomic state plus meta cycles.

The dedicated GitHub Actions workflow runs the targeted tests and a JSON CLI smoke test on Python 3.12.
