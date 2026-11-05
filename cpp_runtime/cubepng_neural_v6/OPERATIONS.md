# CubePNG Neural v6 — production-staging operations

**Release classification:** Candidate for production hardening, not production-certified. Batch-only offline worker: no network listener. This PR does not provision cloud infrastructure or switch traffic.

## Reproduce local tests
```bash
sudo apt-get install -y cmake g++ libpng-dev zlib1g-dev libssl-dev python3
cd cpp_runtime/cubepng_neural_v6
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 2
python3 tests/make_fixture.py build/fixture.png
ctest --test-dir build --output-on-failure
```

## Container
```bash
docker build -t cubepng-neural-v6:local .
docker run --rm --read-only --network none --cap-drop ALL --security-opt no-new-privileges --memory 512m --cpus 2 cubepng-neural-v6:local self-test
# Host folder must be writable by UID 65532.
docker run --rm --read-only --network none --cap-drop ALL --security-opt no-new-privileges --memory 512m --cpus 2 -v "$PWD/data:/data" cubepng-neural-v6:local encode /data/input.png /data/output.jx6 --mode file --strategy adaptive --max-mib 256
```

## Controlled deployment procedure
1. Review and merge the PR only after passing GCC/Clang, sanitizer and container jobs.
2. Configure a GitHub protected environment named `cubepng-production` with independent approval, restricted to main. GitHub connector does not configure this automatically.
3. Manually dispatch the GHCR publish workflow on main. Record resulting image digest, provenance, SBOM and successful smoke test.
4. Replace placeholders in `k8s-job.example.yaml` with an immutable image digest and an existing PVC. Apply via approved cluster tooling.
5. Audit untrusted image parsing, run fuzzing and resource-exhaustion tests before any network API or multi-tenant use.

## File safety and security
The CLI creates randomized exclusive mode-0600 staging files in the output directory and publishes by no-overwrite hard-link creation. Existing target files are refused. Hard-link support is required; SIGKILL can leave orphaned staging files. Archive CRC32/SHA-256 checks are **not authentication or encryption**.

## Limitations
- PNG file mode validates signature/IHDR presence, not the entire PNG chunk structure.
- Pixel mode normalizes to RGBA8 and drops metadata; 16-bit/interlaced PNGs use file mode.
- Per-chunk 3→8→1 quantized MLP is not VQ-VAE or a general 3D ANN.
- The custom format has not beaten mature lossless PNG/WebP in representative tests.
- No authentication, authenticated encryption, public API, distributed scheduling, durable object-store workflow or audited fuzzing baseline.
- CI and GHCR publication remain unverified until corresponding GitHub Actions runs finish.
