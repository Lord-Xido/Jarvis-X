# Million-proposal geometric residual optimizer

Bounded CPU numerical reference extending the RRAET research slice under ADR-016 and ADR-017. It uses a stationary single-field convolution, not the lagged-history model in engine.py. Do not compare their losses as benchmark equivalents.

On the periodic 16^3 lattice, let H have center 10 and six neighbors of weight 1. Synthetic observation y=H truth. For an aligned cubic block B with sides 1,2,4, set w=H 1_B, r=y-Hx. Propose x'=x+d 1_B. The integer step d minimizes ||r-dw||² along this direction, subject to all bytes remaining in [0,255]. The exact change is d²||w||²-2d<w,r>. Commit only if negative and not vetoed. The scale sampling schedule is fixed coarse-to-fine; the integer step adapts to the current residual. The reference uses no trained ANN, learned vocabulary, actual distributed transport, production transaction coordinator or physical GPU.

Each proposal's delta is cross-checked by a direct difference of squares. Every accepted update and every 1000th iteration triggers a dense residual recomputation (same stencil, not an independently implemented convolution). Every 997th proposal is deliberately vetoed, including naturally non-improving ones. The FNV-1a trace fingerprints are deterministic diagnostics, not cryptographic attestations. The CSV records checkpoints at step 1 and every 1000 steps, not individual proposal rows. Residual exactness does not imply exact inverse recovery, compression or global optimality.

Local strict C++17 build and UBSan run, both including full replay, returned:
- 1,000,000 proposals per run; 4,096 voxels
- initial SSE 17,923,751,184; final SSE 416
- 28,267 commits, 971,733 rejected, 1,003 scheduled vetoes included in rejected
- FNV state 846f7b0989a747a7; trace 5b84f2983ad2f0cb
- identical checkpoint CSV across optimized and UBSan builds

Reproduce:
```sh
g++ -std=c++17 -O3 -Wall -Wextra -Wpedantic -Werror cloud_runtime/rraet/geometric.cpp -o /tmp/rraet
/tmp/rraet /tmp/trace.csv
```

This gate addresses bounded numerical refinement and the existing general CI typing issue is separately repaired in the same proposed change. Full ADR-016 runtime adapter, distributed transport, independent oracle, learned multimodal model and system-wide migration remain open. Do not label this million full-volume sweeps or a million repository-wide mutations.
