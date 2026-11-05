# Validated execution evidence — 2026-11-05

- Python codec unit tests: 7 of 7 passed.
- C++20 core: CMake Release compilation succeeded with zlib and OpenSSL.
- Python packet decoded and SHA-256 verified by native C++ CLI.
- Native C++ packet decoded by Python with byte-for-byte identical 32³ voxels.
- HTTP `/api/health`: returned status `ok`.
- HTTP `/api/simulate`: generated Mandelbulb 32³, four cycles, per-cycle MSE=0.
- Final run: 32,768 raw bytes vs 7,282 packet bytes, compression factor 4.50×.
- HTTP `/api/download/{id}` packet decoded via POST `/api/decode` to matching SHA-256 state.
- Browser JS loaded and rendered 1,704 3D surface samples from actual decoded voxel bytes, with interactive controls and cycle telemetry. Browser-rendering test used injected *actual server responses* because the container's Chromium policy blocked direct localhost navigation. Thus browser code and server API were each tested; full network-backed browser integration was not exercised.
- Native JavaScript syntax verification via `node --check static/app.js` passed.

**Limitations:** No external public deployment, GPU rendering benchmark, load test, or security penetration test was performed. The browser screenshot was captured with the Canvas 2D projection fallback in headless Chromium. WebGL2 is implemented but not independently verified on GPU hardware in this run. The `.jx3d` integrity digest is not an authenticated signature. Public deployments need HTTPS, authentication, rate limiting, and persistent storage.
