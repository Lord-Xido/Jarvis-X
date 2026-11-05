# JX3ANN Connected Cloud Verification Gate (2026-11-05)

This isolated integration connects two **existing, repository-resident** computational components to GitHub-hosted execution:

1. `cloud_runtime/tib/jx3d_1tib`: physically iterates 1024 one-MiB synthetic 3D byte tiles (1 GiB total), verifies exact recovery, and rejects an intentionally corrupted stream.
2. `experiments/jx3d_v11`: builds and executes the native C++20 ANN reference and its HTTP/runtime tests.

The two jobs run on GitHub-hosted Ubuntu **CPU** virtual machines and preserve logs as short-lived GitHub Actions artifacts. This is live cloud execution after the PR workflow runs, not a deployment of a persistent inference service.

## Critical boundaries

- This workflow **does not** run on ChatGPT's internal model accelerators, which are not offered as an arbitrary CUDA runtime.
- This workflow **does not** compile or execute the conversation's JX3-ANN **v7** CUDA kernel: that source is still only in the separately produced v7 cloud pilot package and is not present in this Git branch.
- GPU inventory is captured, but a GPU is neither provisioned nor assumed.
- The 1 GiB dataset is synthetic and generated during the check. It is **not** a 1 GiB user PNG, and it is **not** a `10^27`-byte physical computation.
- No paid third-party GPU hardware is provisioned by this pull request, and no credentials are stored.

## Cloud gate

Workflow: `.github/workflows/jx3ann-connected-cloud.yml`.

The gate runs on pull requests touching its own workflow file, so opening a draft PR starts a real cloud validation. Cloud CI pass/fail is authoritative only **after** GitHub Actions reports the result.

## Next step: actual v7 CUDA parity

Publish a reviewed, non-sensitive v7 source subtree separately and configure an explicit NVIDIA GPU-equipped runner. Require device execution and byte-for-byte CPU/GPU prediction comparisons, repeated benchmarks, and immutable artifact hashes before reporting any GPU speedup.
