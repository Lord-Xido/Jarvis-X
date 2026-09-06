# Jarvis-X 3D MP4 Autoencoder

Executable Python reference path for video auto-encoding and decoding inside the QSOL graphics codec subsystem.

## Invariant

```text
MP4
  -> RGB frames [T,H,W,3]
  -> non-overlapping clips [N,3,Tc,H,W]
  -> Conv3D encoder E_phi
  -> latent Z [N,Cz,Tc/4,H/4,W/4]
  -> float16 latent archive (.npz)
  -> Conv3D decoder D_theta
  -> reconstructed RGB frames
  -> MP4
  -> optional original-audio remux via ffmpeg
```

The reconstruction objective is

```text
L = MSE(X_hat, X) + lambda_t * MSE(Delta_t X_hat, Delta_t X)
```

so the reference model penalizes both pixel error and temporal-motion error.

## Install

```bash
python -m pip install -r apps/qsol-graphics-codec/VideoPython/requirements.txt
```

Optional: install `ffmpeg` on `PATH` to preserve/remux the source audio stream.

## Deterministic synthetic smoke test

```bash
mkdir -p artifacts/mp4-ae
python apps/qsol-graphics-codec/VideoPython/jarvisx_mp4_3d_autoencoder.py \
  --synthetic \
  --input artifacts/mp4-ae/synthetic.mp4 \
  --output artifacts/mp4-ae/reconstructed.mp4 \
  --latent artifacts/mp4-ae/latent.npz \
  --width 32 \
  --height 32 \
  --clip-len 4 \
  --latent-ch 4 \
  --synthetic-frames 8 \
  --epochs 1 \
  --batch-size 2 \
  --device cpu
```

## Real MP4

```bash
mkdir -p artifacts/mp4-ae
python apps/qsol-graphics-codec/VideoPython/jarvisx_mp4_3d_autoencoder.py \
  --input input.mp4 \
  --output artifacts/mp4-ae/reconstructed.mp4 \
  --latent artifacts/mp4-ae/latent.npz \
  --width 128 \
  --height 128 \
  --clip-len 8 \
  --latent-ch 8 \
  --epochs 20
```

## Measurements

The CLI reports:

- source tensor shape;
- latent tensor shape;
- reconstructed frame shape;
- MSE;
- PSNR;
- raw RGB bytes;
- float32 and float16 latent bytes;
- raw/latent compression ratios;
- self-contained NPZ size.

The raw/latent ratio is the relevant deployment metric only when decoder weights are shared. The self-contained reference archive deliberately stores model weights as well as the latent so that encode and decode are separated by a real serialization boundary.

## Scope

This is a learned video-codec reference engine, not an H.264/H.265 replacement. It is intended to operationalize the Jarvis-X inward latent-state model on an actual MP4 stream and provide measurable reconstruction behavior that can later be connected to hierarchical fixed-point, residual-ledger, or coordinate-conditioned virtual-state decoders.
