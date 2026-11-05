# DM3D v5 — isolated 3D inward-loop reference

This directory is a **bounded numerical experiment** integrating the verified local v5 runtime into Jarvis-X without changing existing production codecs, models, or CI.

## Components

- `dm3d_septillion_inward_v5.py`: source/PDF-conditioned 3D search and CTR-like verification loop; 3D multiscale field; candidate timing, quality threshold, Pareto comparison, hash-chain PDF archives.
- `dm3d_septillion_inward_v4.py`: **scoped compatibility subset** of the local v4 reference. It contains the same v4 encoder/decoder, voxel generator, field and isometric plotting functions needed by v5, but intentionally omits v4's separate CLI, report generator, and v4 evolutionary runner.
- `test_dm3d_v5.py`: tests exact recovery, XYZ spatial transforms, fixed-target contraction and archive tamper detection.

## Run

Python 3.10+:

```bash
cd reference/dm3d_septillion_v5
python -m pip install -r requirements.txt
python dm3d_septillion_inward_v5.py selftest
python -m unittest -v test_dm3d_v5.py
```

For your own image and existing PDF:

```bash
python dm3d_septillion_inward_v5.py evolve \
  --seed-pdf input.pdf --input-image input.png \
  --outdir run --generations 4 --candidates 36 \
  --trials 2 --size 24 --psnr 36.5
python dm3d_septillion_inward_v5.py verify \
  --seed-pdf input.pdf --outdir run
```

The input PDF is an existing file. If the image path is absent the program produces a deterministic synthetic XYZ input volume. Generated `run/` archives and PDFs are outputs, not checked into Git.

## Mathematics and claims

The 3D recurrence is `F' = .55F + .10*N3(F) + .07*U(D(F)) + .28*Y`, where `N3` averages six periodic XYZ neighbors, `D` averages 2x2x2 spatial blocks, and `U` repeats the coarse field. With fixed `Y`, the mapping has an infinity-norm Lipschitz bound of 0.72.

`(10^24)^(10^24)` is a **symbolic** cardinality, not a number of performed iterations. The actual finite genotype lattice is `Z8 × Z8 × Z12` (768 combinations). Exact recovery is guaranteed and checked for the baseline codec only. Other configurations may be lossy and are constrained by a PSNR gate. SHA-256 hashes attest byte integrity, not authenticity. Benchmark times depend on the host and are not end-to-end PDF-generation speedups. No autonomous model training, source rewriting or AVX-512 acceleration is claimed.

## Local test evidence

The integration subset was exercised locally with a synthetic 12³ XYZ grid: v5 selftest passed, bounded two-generation feedback and 6-page reports completed, two-generation independent archive verification succeeded, and corruption of a PDF caused verification failure. **GitHub Actions is the independent repository-side check**. Original v5 benchmark outcomes obtained with the complete v4 file may differ because the source hash conditions proposal order and local timing influences kernel promotion.

This module is a reviewable reference, not a deployment-ready or trained intelligence engine.
