# Electromagnetic-Photonic Pixel-Field Runtime

## Status

**Numerical reference / integration candidate.** The implementation in `src/jarvisx/photonic_rendering.py` defines a deterministic, bounded rendering contract in which a pixel is a finite-area spectral detector. It is intended for small correctness fixtures and architectural integration, not production rendering or full-wave electromagnetic simulation.

## 1. System interpretation

The subsystem separates the physical phenomenon from its digital representation:

```text
electromagnetic emission
→ wavelength-dependent propagation
→ material interaction
→ camera projection
→ finite-area detector integration
→ exposure and transfer function
→ integer RGB quantization
→ verification
→ commit or rollback
```

A pixel is not treated as an isolated coloured square. It is the result of a bounded measurement operator applied to a continuous spatial light field:

```text
p[i,j,c] = Q_b[
    integral over exposure time
    integral over wavelength
    integral over pixel aperture
    S_c(lambda) L(x, y, lambda, t)
]
```

The reference implementation evaluates this integral by deterministic finite sampling and a small ordered wavelength set.

## 2. Physical approximation boundary

The implementation includes:

- geometric camera rays;
- ray/sphere intersection;
- inverse-square emitter attenuation;
- wavelength-dependent RGB detector response;
- Lambertian diffuse reflection;
- one bounded roughness-controlled specular term;
- hard visibility tests;
- exposure, tone mapping, gamma transfer and 8-bit quantization.

It does **not** include:

- time-domain Maxwell integration;
- diffraction, polarisation or near-field coupling;
- calibrated BRDF or sensor spectral data;
- recursive global illumination;
- participating media;
- production acceleration structures;
- stochastic Monte Carlo claims;
- hardware GPU acceleration on the reference path.

The phrase electromagnetic-photonic therefore describes the physical interpretation and spectral measurement model, not a claim that the code solves Maxwell's equations.

## 3. Authoritative state transition

The runtime cycle is:

```text
observe scene and camera
→ validate dimensional and resource contracts
→ partition the detector into deterministic tiles
→ transport spectral energy to each surface sample
→ integrate each pixel aperture
→ quantize the detector response
→ compute the canonical frame digest
→ verify every output bound
→ commit frame and Omega digest, or rollback the entire cycle
```

The transition is represented as:

```text
(frame, state, receipt) = renderer.cycle(scene, camera, config)
```

A failed cycle returns `frame=None`, preserves the previous authoritative runtime state and emits a rollback receipt with a reason.

## 4. Deterministic orchestration contract

The detector is partitioned into row-major `WorkTile` records. Tile order, pixel order, wavelength order and digest serialization are fixed. A future CPU thread pool, GPU kernel or distributed worker fabric may replace physical execution only if it preserves:

- pixel semantics;
- tile and pixel coordinate identity;
- configured resource bounds;
- quantized RGB results for the declared numerical profile;
- canonical frame digest;
- whole-cycle commit and rollback behavior.

This creates an implementation boundary between **what a pixel means** and **where the computation runs**.

## 5. Pixel-to-3D lattice projection

A rendered detector sample can be projected into the canonical sparse `1000^3` spatial domain:

```text
X = floor((pixel_x + 0.5) * side / image_width)
Y = floor((pixel_y + 0.5) * side / image_height)
Z = floor((depth / (1 + depth)) * (side - 1))
```

A miss uses the far plane (`Z = side - 1`). This projection is an address mapping, not a claim that a 2D image uniquely reconstructs physical 3D reality.

## 6. Minimal use

```python
from jarvisx.photonic_rendering import (
    Camera,
    Material,
    PhotonicRenderer,
    PhotonicScene,
    PointEmitter,
    RenderConfig,
    Sphere,
    Spectrum,
)

scene = PhotonicScene(
    spheres=(
        Sphere(
            center=(0.0, 0.0, -3.5),
            radius=1.2,
            material=Material(reflectance=(0.8, 0.32, 0.14), roughness=0.35),
        ),
    ),
    emitters=(
        PointEmitter(
            position=(-2.5, 3.0, 0.0),
            spectrum=Spectrum.white_reference(),
            intensity=120.0,
        ),
    ),
)

renderer = PhotonicRenderer()
frame, state, receipt = renderer.cycle(
    scene,
    Camera(width=32, height=18),
    RenderConfig(samples_per_axis=2, tile_edge=8),
)

assert receipt.committed
assert frame is not None
assert state.frame_digest == frame.digest
```

## 7. Promotion requirements

Before this layer is represented as an accelerated or physically calibrated renderer, it requires:

1. independent spectral and radiometric reference comparisons;
2. a declared floating-point reproducibility profile;
3. CPU baseline benchmarks;
4. GPU tile-kernel equivalence tests;
5. image fixtures with retained machine-readable metrics;
6. explicit energy-accounting tolerances;
7. a threat model for untrusted scene and model inputs.
