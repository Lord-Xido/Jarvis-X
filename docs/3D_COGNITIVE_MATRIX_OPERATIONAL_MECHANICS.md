# 3D Cognitive Matrix Operational Mechanics — WebGL Demo

## Status

This repository example is the browser visualization supplied as
`examples/3D_Cognitive_Matrix_Operational_Mechanics.html`.

It is a deterministic-structure / stochastic-telemetry visualization companion to the executable
Jarvis-X inward 3D bytecode runtime. The page visualizes the intended mechanics; it is not itself
the authoritative numerical runtime, benchmark harness, cloud client, or proof of hardware
performance.

## Visualized pipeline

The demo renders five explicit stages:

1. 8×8 input bus injection at z = -25.
2. Encoder conduit transport through cubic Bézier particle paths.
3. 8×8×8 cognitive matrix / sparse activation visualization.
4. Decoder conduit transport toward the reconstruction plane.
5. 8×8 output-bus reconstruction at z = +25.

The page also exposes excitation-mode selection, latent sparsity threshold, conduit-speed control,
manual stage navigation, auto-cycle sequencing, and a Three.js orbit camera.

## Current simulation boundary

The present browser artifact uses animation-time equations and pseudorandom values for parts of its
telemetry:

- input modes are generated locally in JavaScript;
- latent voxel activation is a sinusoidal spatial wave thresholded by the sparsity slider;
- output reconstruction adds small random perturbations to the generated input signal;
- displayed MSE is a synthetic random value;
- displayed conduit latency is a presentation formula derived from the speed slider;
- the cloud beam and ONLINE indicator are visual elements and do not establish a live cloud
  connection.

Accordingly, the browser HUD must be interpreted as simulated visualization telemetry unless it is
later bound to measured runtime reports.

## Executable-runtime correspondence

The browser stages correspond conceptually to the executable spatial VM in
`src/jarvisx/dr_moagi_inward_3d_vm.py`:

| Browser visualization | Spatial VM/runtime analogue |
| --- | --- |
| 8×8 input grid | 64-channel input bus |
| encoder particle convergence | ENCODE3D / spatial routing |
| cognitive matrix | PHI3D_ITER / fixed-point latent core |
| decoder particle divergence | DECODE3D / output routing |
| reconstructed output plane | residual, error field, correction and output |
| repeated stage sequence | bounded inward recurrence |
| displayed sparsity control | active working-set / sparse-policy analogue |

The spatial VM remains authoritative for fixed-point telemetry, contraction bounds, shadow
candidate evaluation, promotion/rollback and traceable 64-bit spatial bytecode.

## Running the demo

Open the HTML file in a browser with network access to its CDN dependencies:

- Tailwind CSS
- Three.js r128
- OrbitControls
- Font Awesome

No build step is required.

## Claim boundary

The visual labels `1GB³`, displayed latency, displayed MSE and cloud-status indicator are not
hardware measurements. Hardware throughput, power, latency, photonic performance and external
state-of-the-art comparisons require separately instrumented benchmarks and provenance.
