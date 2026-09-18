# AE-3D Infinity IDE / SDK / ANN Engine

A working reference implementation of the recursive byte-level 3D auto-encoding/decoding architecture.

## What is implemented

- Real `Uint8Array` byte ingestion.
- 256×32 learnable byte embedding/codebook.
- 6×6×6 = 216 voxel field, 32 latent dimensions per voxel.
- Toroidal boundary-aware 6-neighbour field folding.
- INT8 per-voxel quantization/dequantization.
- 256-way fused byte decoder (dot-product + argmax, no softmax required for inference).
- Recursive normalized-byte feedback `x_{t+1}=(1-β)x_t+β x_hat_t`.
- SGD training of byte embedding and decoder codebook.
- Exact byte verification, BER, MSE, PSNR, quantization MSE and latent-stability telemetry.
- Browser IDE with script editor, SDK console, 3D state visualization and model export.
- Node smoke test.

## Run

From this directory:

```bash
npm run smoke
npm run serve
```

Then open `http://localhost:8080`.

The browser IDE uses Three.js from a CDN, so the 3D panel requires network access. The ANN engine and Node smoke test do not.

## SDK

```js
const sdk = new AE3D.AE3DSDK({
  recursionDepth: 4,
  beta: 0.85,
  foldGain: 0.06,
  learningRate: 0.02
});

const result = sdk.run('AUTOENCODER_3D_BYTES');
console.log(sdk.decodeText(result.output));
console.log(result.metrics);

sdk.train('AUTOENCODER_3D_BYTES', { epochs: 10 });
```

## Engine equations

The executable recurrence is approximately:

```text
B_t -> Embed -> H_t(6×6×6×32) -> Fold3D -> INT8 Q/DQ -> ByteArgMax -> B_hat_t
  ^                                                                      |
  |---------------- x_(t+1)=(1-beta)x_t + beta*x_hat_t -----------------|
```

The torus in the IDE is a visualization of the 216-voxel state. It is not presented as separate physics; the actual computation is typed-array numerical software.

## Important benchmark note

The `1000-pass test` executes 1000 actual recursive passes. It is deliberately not labelled a 1000× wall-clock acceleration. A true macro-step/spectral accelerator would be a separate optimization layer and must be benchmarked against this reference engine.
