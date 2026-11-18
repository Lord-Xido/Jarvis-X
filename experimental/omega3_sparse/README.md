# JARVIS-X Omega3 Integrated Reference Runtime

## Scope
C++17 sparse 3D virtual-address engine with 2^30 positions per axis, 90-bit Morton codes, 72-bit full-period tile scheduler, 64^3 virtual tiles, deterministic seeded byte values, a sparse persisted override map, a lossless 8^3 spatial XOR predictor codec, and a bounded 6-neighbour numerical refinement loop with objective-based accept/reject and trace. No dense volume allocation.

**Not implemented:** a learned ANN, trainable parameters, compression ratio guarantee, 64^3 tile codec, distributed runtime, autonomous code generation, authenticated or atomic storage. The `--quine PATH` command copies the file provided; it is not a pure quine. The sparse store is a simple plaintext prototype format and is not safe against malicious corruption or concurrent writers.

## Build
`g++ -O3 -std=c++17 -Wall -Wextra -pedantic jarvis_x_omega3_complete.cpp -o jarvis_x_omega3`

`clang++ -O2 -std=c++17 -Wall -Wextra jarvis_x_omega3_complete.cpp -o jarvis_x_omega3`

GCC/Clang `unsigned __int128` is a non-ISO extension, so `-pedantic` produces a warning.

## CLI
- `./jarvis_x_omega3 --selftest`
- `./jarvis_x_omega3 --voxel 7553 4034 2051`
- `./jarvis_x_omega3 --tile 0`
- `./jarvis_x_omega3 --tile-voxel 0 1 2 3`
- `./jarvis_x_omega3 --refine 0 8`
- `./jarvis_x_omega3 --write state.txt 7 8 9 201`
- `./jarvis_x_omega3 --read state.txt 7 8 9`
- `./jarvis_x_omega3 --bench 100000`
- `./jarvis_x_omega3 --quine jarvis_x_omega3_complete.cpp > reproduced.cpp`

## Algebra
Let p(t)=(505050+104729*(t mod 2^72)) mod 2^72. q=Morton72^-1(p). Global coordinate r=64q+l, l in [0,63]^3. Morton90(r)=2^18*p+Morton18(l). As gcd(104729,2^72)=1, p has period 2^72. The 8-bit voxel value is high8(mix64(low64(m) xor mix64(high26(m) xor seed))). This is deterministic but not injective.

## Refinement
The prototype operates on the first 8^3 voxels of a scheduled 64^3 tile, with a 6-neighbor smoothing candidate and reference fidelity. Objective J=4*sum(v-reference)^2 + sum_over_positive_axis_edges(v_i-v_j)^2. It accepts only candidates satisfying J(next)<=J(current). Integer quantization and bounded iteration do not guarantee strict decrease or convergence to a unique optimum; monotonic nonincrease is enforced by acceptance. Every accepted block is encoded then decoded and checked exactly.

## Local validation
GCC and Clang compiled and passed built-in self-tests. GCC ASan/UBSan self-tests and a short refinement run passed. Source-file reproduction was byte-compared. Example tile 0 local (1,2,3): coordinate (7553,4034,2051), Morton 132395827253, value 133. Example 10,000-query single-process run: about 6.16 million queries/s in this container; this is a workload-specific observation, not an architecture-wide throughput guarantee.
