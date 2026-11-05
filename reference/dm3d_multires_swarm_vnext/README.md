# Dr Moagi 3D Intelligence Engine — vNext 1.0

A **runnable, bounded Python/NumPy multiresolution 3D auto-encoding/decoding reference implementation**. It replaces irreversible average-pooling as the sole spatial representation with an invertible 3D Haar pyramid, retains a real 9-agent trainable ANN latent branch, and adds verified recursive operation and autonomous-but-bounded architecture selection.

## Installation

```bash
python -m pip install -r requirements.txt
python -m unittest -v test_vnext test_evolver
```

## Fully working commands

```bash
# Auto-evolve geometry depth, latent width, trained weights, and detail retention.
python evolve_architecture.py --edge 8 --tiles 8 --steps 80 --target-mse 0.015 --out evolved_output

# Train/encode/decode in one command. One packet is an independently decodable artifact.
python dr_moagi_vnext.py --edge 8 --tiles 8 --levels 2 --hidden 6 --steps 80 --out output

# Decode only: no original input, ANN encoder, or training process is used.
python dr_moagi_vnext.py --decode evolved_output/encoded.dm3d.npz --out decoded_only

# Resume the training state, including learned weights, swarm logits and recurrent memory.
python dr_moagi_vnext.py --resume evolved_output/model_checkpoint.npz --out resumed_output

# Ingest one or more real modality sources through the included legacy adapter.
python dr_moagi_vnext.py --text 'Hello' --out with_text
python dr_moagi_vnext.py --image image.png --audio speech.wav --out with_media

# Preassembled numerical multimodal array under key `input`, shape (B,N,N,N,4).
python dr_moagi_vnext.py --encode-npz numerical_input.npz --out encoded_input
```

Optional media dependencies: `Pillow` for image input, `opencv-python` for video input. Audio supports uncompressed PCM WAV in the included compatibility adapter. The adapter converts media to sampled numerical volumes; **it does not encode complete original media files losslessly**.

## Main artifacts

- `dr_moagi_vnext.py`: self-contained 3D Haar transform, trainable swarm, CTR optimizer, residual-preserving codec, packet I/O, runtime.
- `evolve_architecture.py`: finite architecture evolution over pyramid levels and latent widths.
- `dr_moagi_inward_intelligence.py`: previously developed real-media sampling adapters / earlier architecture compatibility.
- `test_vnext.py`, `test_evolver.py`: 24 automated regression tests.
- `evolved_output/encoded.dm3d.npz`: portable independently decodable packet.
- `evolved_output/model_checkpoint.npz`: stateful optimizer, memory and learned weights.
- `evolved_output/summary.json`: metrics, verification, resource and integrity data.
- `evolved_output/evolution_search.json`: all architecture trials.
- `evolved_output/candidate_search.json`: measured quality and packet sizes for each detail retention policy.
- `evolved_output/independent_decode/decoded_only.npz`: actual separate decoder process output.

## Reproducibility and measured prototype results

A fixed-seed local run on 8 tiles of 8×8×8 voxels with 4 synthetic modality channels (4,096 spatial voxels, 16,384 scalar values) evaluated four finite architectures. Selection target on development validation MSE: <= 0.015. Selected **two Haar levels and six latent channels**, retaining **25% of detail coefficients**.

- Full array MSE: **0.00536514**.
- Independent held-out test tile MSE: **0.00613050** (target met).
- Original float64 array: **131,072 bytes**; portable `.npz` packet: **24,867 bytes** (ratio **5.2709x** on these synthetic fields, including decoder matrices and indices).
- Baseline low-resolution 2× spatial average-pooling irreducible MSE on the same input: **0.12896038**. The new detail-retaining codec yields a 95.84% smaller MSE than this floor, on the same input.
- 9/80 optimization trial steps accepted; 71 rejected by finite train + development validation CTR gating. Rejections are **not errors**: the last verified parameters are restored.
- Numerically reversible mode (`keep=1`, `base_correction=True`) achieves maximum absolute error below `1e-13` in the test case, **not bitwise identity**.
- Independent decoding of stored packet produced exactly equal NumPy arrays to in-process decoding.
- **24 regression tests passed** in local execution.

These numbers are restricted to this small synthetic evaluation. The **symbolic** virtual address volume is `10^18000000` sites; the theoretical recursion budget `(10^24)^(10^24)` is **not executed**. A learned ANN branch, fixed mathematical agents, and a finite architecture search do not constitute human-equivalent cognition or unlimited self-evolution. No parallel GPU/cloud/distributed hardware acceleration has been benchmarked.

## Security and authorization

- Agent roles in this implementation are *mathematical transformations*, not open-ended autonomous actors.
- The system performs no external tool calls or privileged operating-system actuation.
- Training updates are candidate-gated; stability is measured on finite selected datasets, **not guaranteed on unobserved cases**.
- Decoder enforces tensor and packet-size caps and a SHA-256 checksum for accidental integrity corruption. Checksum is **not authenticity, signatures, encryption, or adversarial security**.
- For unattended or safety-critical deployment, require independent human authorization, signed artifacts, provenance, privacy controls, threat modeling, comprehensive tests on held-out real distributions, resource quotas and monitoring.
