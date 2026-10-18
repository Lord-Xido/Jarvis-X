# Dr Moagi: bounded 8K UHD 3D inward-loop framebuffer reference

Status: CPU software reference. No hardware/panel integration or 60 Hz claim.

## Exact contract and storage

- 8K UHD frame: 7680 x 4320 = 33,177,600 RGB pixels.
- Raw source/output (optional): RGB24 row-major, 99,532,800 bytes/frame.
- Streaming: strips of 64 rows, subdivided into 64x64 tiles. Partial border tiles supported.
- Each tile maps local row-major pixel address \`p\` to voxel \`p mod 8^3\` on an 8x8x8 periodic lattice. Three RGB channels per voxel.
- Persistent Omega: three quantized bytes per voxel and tile. No full-frame buffer is retained.

## Verification and fixed-point mechanics

For one tile, aggregate each voxel's sums, sums of squares, and counts. Let \`m=E(X)\` be the per-voxel RGB mean, \`z\` its currently committed latent vector, and \`Delta_T3\` the six-neighbor periodic Laplacian. Trial refinement is

\`\`\`
F(z) = clip_[0,1] [ z + gamma*(m-z) + diffusion*Delta_T3(z) ]
\`\`\`

The implementation defaults are \`gamma=.60\`, \`diffusion=.01\`, five steps and \`Omega\` EMA beta \`.88\`. A conservative sufficient sup-norm Lipschitz bound is \`q = 1-gamma+12*diffusion = .52 < 1\` for a fixed tile and constant parameters. This proves convergence of repeated \`F\` to its own fixed point, not optimal reconstruction or a self-training neural network.

The per-tile RGB channel MSE is evaluated without materializing the reconstructed image:

\`\`\`
SSE(z) = sum_(voxels,channels) (sum(x^2) - 2*z*sum(x) + count*z^2).
\`\`\`

CTR adopts the trial only when its SSE does not exceed the authoritative SSE by more than \`1e-10\` (floating point slack); otherwise it rejects that step. The final RGB24 output is quantized from the accepted latent. Metrics describe pre-quantized decoded values; output quantization can change pixel error slightly.

Persistent memory updates after every tile as \`Omega_next = round_8bit(beta*Omega_old+(1-beta)*z_authoritative)\`. The persistent state is recycled at the same tile coordinates in following demo frames.

## Build and exercise

\`\`\`sh
cmake -S cpp_runtime -B build/dm3d8k -DJARVISX_BUILD_GL_VISUALIZER=OFF
cmake --build build/dm3d8k --target jarvisx-8k-inward-framebuffer jarvisx-8k-inward-framebuffer-tests
ctest --test-dir build/dm3d8k -R 8k-inward-framebuffer --output-on-failure

# Full 8K UHD procedural reference, measured CPU time printed by runtime:
./build/dm3d8k/DrMoagi-8K-Inward-Framebuffer --width 7680 --height 4320 --frames 1

# Input must contain exactly one full row-major RGB24 frame (trailing bytes ignored):
./build/dm3d8k/DrMoagi-8K-Inward-Framebuffer --width 128 --height 96 \
  --input frame.rgb --output reconstructed.rgb
\`\`\`

## Boundaries and future work

This is a deterministic RGB codec/control proof, **not** a learned multimodal neural network, hardware GPU implementation, or panel driver. Computational tile compression is lossy and can reduce quality. The baseline uses the quantized Omega prediction; recursive improvement is verified against that prediction, not against a state-of-the-art image codec. At 8K, lossless RGB24 occupies about 94.9 MiB; this implementation's latent Omega is approximately 12.5 MB, plus two streaming scan strips and small temporary arrays. Reported \`bounded_bytes\` is a conservative model-side bound, not a profiler measurement of total process RSS. Wall times are hardware-specific, and 60 Hz cannot be assumed.

Integration point: existing \`cpp_runtime/src/dr_moagi_8m_aead_main.cpp\` and \`cpp_runtime/src/inward_graphics_ann_main.cpp\` retain their previous contracts. This separate streaming module keeps their behavior stable while exposing a CPU reference for subsequent 8K graphics/GPU adapters.
