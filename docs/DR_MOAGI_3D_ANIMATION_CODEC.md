# Dr Moagi 3D Graphics Animation Autoencoder / Decoder

This runtime extends the Jarvis-X 3D auto-encoding work from static latent mappings into a temporal mesh-animation codec.

## Operational pipeline

```text
mesh frames [T,V,3]
  -> clip-wide spatial normalization
  -> position + temporal-velocity features
  -> feature standardization
  -> dense NumPy encoder
  -> latent timeline z[t]
  -> keyframe-aware delta coding
  -> symmetric integer quantization
  -> packet (.npz)
  -> latent dequantization / temporal reconstruction
  -> neural decoder
  -> denormalized mesh frames
  -> OBJ + PGM previews + reconstruction telemetry
```

The reference implementation is exactly **1,000 physical Python source lines**. To keep that invariant auditable, Git stores it as five ordered 200-line fragments under:

`reference/dr-moagi-3d-animation-autoencoder/fragments/`

The Jarvis-X bridge in `src/jarvisx/dr_moagi_3d_animation_codec.py` reconstructs the fragments byte-for-byte before execution and rejects the runtime if its line count, syntax, or SHA-256 digest changes unexpectedly.

Canonical source integrity:

- physical lines: `1000`
- SHA-256: `c3d05a11e3fbb91591f538c75bddc15a72dd398e62ae4624a7b1a0828efa620e`
- language/runtime: Python 3.10+
- numerical backend: NumPy (`jarvisx[graphics]`)

## Install

```bash
python -m pip install -e '.[graphics]'
```

## Verify the canonical source

```bash
jarvisx-dr-moagi-3d-animation --verify-reference
```

## Materialize the single 1,000-line file

```bash
jarvisx-dr-moagi-3d-animation --materialize ./dr_moagi_3d_animation_autoencoder_1000_lines.py
```

## Run an end-to-end cube animation encode/decode

```bash
jarvisx-dr-moagi-3d-animation \
  --demo cube \
  --frames 90 \
  --latent 12 \
  --hidden 48 \
  --epochs 400 \
  --quant-bits 12 \
  --keyframe 12 \
  --output ./dr_moagi_codec_output
```

For a deforming surface instead of a rigid/deforming cube, use `--demo wave`.

## Decode a saved packet

```bash
jarvisx-dr-moagi-3d-animation \
  --mode decode \
  --packet ./dr_moagi_codec_output/animation_packet.npz \
  --output ./decoded_output
```

## Packet state

The packet carries the constant mesh topology, spatial normalization state, quantization scales, quantized latent codes, keyframe mask, and decoder parameters needed to reconstruct the animation. The encoder parameters are intentionally not required for playback.

## Current scope

This is a deterministic research/reference implementation for explicit inspection of the full transformation path. It is CPU/NumPy based and is not presented as a production GPU animation codec. Natural next layers are batched GPU tensors, meshlet partitioning, entropy coding, learned motion prediction, and direct integration with the existing QSOL graphics codec and C++ 3D runtime.
