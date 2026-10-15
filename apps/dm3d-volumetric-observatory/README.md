# DM3D Volumetric Matrix Observatory — browser demonstration

A self-contained bundled React 18 / Three.js 3D visualisation derived from the user-supplied **React Artifact(6).html**. This is an isolated proposed Jarvis-X interface, not a replacement for the verified DM3D v5 codec or a functioning 1 GiB³ ANN.

## What runs

Open `index.html` in a browser with WebGL enabled, or serve it using `python3 -m http.server` from this folder. The bundled JavaScript contains React, Three.js and OrbitControls. It renders a 3D wireframe cube, an interior cube, **512 instanced voxel glyphs** and a **1024-point toroidal track**. You can orbit the scene, toggle 8 bit-plane overlays, switch visual modes, inspect glyphs by double-clicking and export a generated canvas image.

The screenshot/bitmap export is a **2D canvas-generated latent texture**, not an export of the WebGL scene, a real encoded image, or an exact 3D volume.

## Evidence and numerical boundaries

- The displayed `1,073,741,824³` is a **logical** coordinate notation (2^90 addresses) and is **not allocated** in RAM/VRAM. The 3D renderer contains 512 instantiated boxes, not 2^90 physical voxels.
- Resident chunk and compression fields are initialized with fixed numbers and updated by state changes; no measured codec payload or reconstruction identity check is performed in this UI.
- Counters labelled as throughput, phase, entropy, reconstruction quality and evolution are **synthetic visual telemetry**: the app uses `Math.random()` and periodic timer updates rather than querying an executable kernel.
- The `AUTO-EVOLVE` action changes UI state; it is not verified training, kernel code-generation, or automatic improvement.
- The visual bit-plane toggles change overlay visibility; they do not establish bit-exact encoding/decoding.
- No GitHub API, model training service, live telemetry server, or DM3D v5 archive is connected by this page.

The primary demonstration page includes a visible provenance banner and replaces `LIVE` / `AUTO-EVOLVE MODE` labels with `SIMULATED` / `AUTO-EVOLVE DEMO`.

## Provenance and isolation

Upstream input: the attached `React Artifact(6).html` file. Original file SHA-256: `c2b48ae9ad0b111283097ed6bd1e144e1717d739cfa21e38d51c8a8ceff4bdb4`.

The integration preserves the bundled React/Three.js module, existing controls and CSS. It removes host-specific injected link/iframe scripts, nonportable root-level font URLs, and adds provenance labels. The visual bundle is not source-level editable React JSX; this is a demo integration, not a maintained npm build pipeline.

## Reproducible checks

From repository root:

```bash
python3 -m unittest discover -s apps/dm3d-volumetric-observatory -p 'test_*.py' -v
```

The tests check that the artefact remains self-contained, contains the expected rendering structure and provenance disclosures, and has a JavaScript module parseable by Node.js. An actual browser/WebGL smoke test is **not** part of CI and should be performed manually; static tests cannot establish shader/rendering compatibility.

## Next integration boundary

A separate, independently verified receipt adapter is required before reporting **actual** DM3D execution metrics. It should accept signed or hash-checked run records (`input_sha256`, `output_sha256`, `steps_executed`, `reconstruction_mse`, `elapsed_ms`, `allocated_bytes`), gate promotion through CTR, and distinguish user-loaded receipts from simulated HUD values. Until then the display remains explicitly demonstrative.
