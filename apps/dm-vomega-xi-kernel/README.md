# DM-vΩΞ+ Operational Core Kernel

This app turns the submitted GLSL operator sketch into a valid WebGL2 pipeline.

## What changed

The original snippet mixed vertex-stage semantics (`a_spatialVertex`, `a_quaternionToken`) with a fragment output (`fragColor`) and had no `gl_Position`. The operational form therefore splits the kernel into:

- `shaders/core.vert`: spatial/quaternion operator stack and recursive inward contraction;
- `shaders/core.frag`: cognitive-state spectrum mapping and point-sprite shading;
- `index.html`: self-contained WebGL2 host, controls, telemetry and the same embedded shader pair.

## Geometric operator

For a normalized token `q = (v,w)`, the descriptive map is

```text
Phi_q(p) = w p + v x p
```

The inward map is

```text
Lambda_Delta(p) = c + exp(-Delta) (Phi_q(p) - c)
```

where `c` is a bounded semantic attractor. For unit `q`, `Phi_q` is non-expansive, so for `Delta > 0`:

```text
||D Lambda_Delta|| <= exp(-Delta) < 1
```

The GPU applies twelve recursive iterations per resident vertex, making the inward fixed-point behavior an actual geometric recurrence rather than a label.

## Run

Serve the repository root with any static server, for example:

```bash
python -m http.server 8080
```

Then open:

```text
http://localhost:8080/apps/dm-vomega-xi-kernel/
```

The standalone `index.html` has no package or CDN dependency.

## Controls

- **Delta context / inward strength** controls the contraction coefficient `exp(-Delta)`.
- **hbar semantic stability threshold** gates the visual stability field.
- **point scale** controls raster point size.
- **orbital speed** controls the camera orbit only; it does not alter the contraction proof.

## Boundary

The visualization is a bounded GPU field simulator. It does not imply AGI, measured intelligence, physical energy, or hardware throughput. The operator names are architectural labels; operational claims are limited to the actual WebGL2 transformations executed by the shaders.
