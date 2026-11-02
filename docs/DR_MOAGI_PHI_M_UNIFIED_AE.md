# Phi_M Unified 3D Autoencoder Research Candidate

**Status:** draft-only synthetic research. The existing mathematical Phi_M operator is authoritative. This optional module is **not** automatically connected to the production Jarvis-X runtime or codec.

## End-to-end pipeline

Input is a moving Gaussian Phi_M field sampled on a finite 3D mesh, not real user data.

    parameters (amplitude, modulation, time)
      -> Phi_M^depth on G^3 coordinates
      -> Y in R^(G x G x G x 3)
      -> 3D Conv / GELU downsampling
      -> latent vector in R^L
      -> Linear / ConvTranspose3D decoder / tanh
      -> Yhat
      -> MSE(Yhat,Y) / held-out evaluation / checkpoint

The geometric map is nonlinear and not generally invertible. The neural model is trained to reconstruct that map's **output fields**, not to decode its source coordinates exactly, compress arbitrary documents losslessly, or learn a general 3D intelligence.

The map's historical two-cycle behavior means a depth parameter must represent a literal number of iterations, **not** a fixed-point guarantee. The dataset uses disjoint random seeds for training and validation, but both datasets share the same synthetic generator family.

## Running from repository root

    pip install -e '.[phi_m_ae]'
    python -m jarvisx.dr_moagi_phi_m_autoencoder --mode audit
    python -m jarvisx.dr_moagi_phi_m_autoencoder --mode train --out ./runs/phiM --iters 1000 --grid 16 --latent 48 --train-samples 48 --val-samples 12 --batch 8
    python -m jarvisx.dr_moagi_phi_m_autoencoder --mode encode --model ./runs/phiM/model.pt --field ./runs/phiM/demo_field.npy --out ./runs/phiM/code.npy
    python -m jarvisx.dr_moagi_phi_m_autoencoder --mode decode --model ./runs/phiM/model.pt --code ./runs/phiM/code.npy --out ./runs/phiM/reconstructed.npy

Shape bounds: G in {8,16}, batch not exceeding train_samples, validated finite inputs, at most 4096 geometric points per call (operator budget). The 3D convolutional bottleneck is grid-aware: G=8 → 1³ and G=16 → 2³. The model rejects incompatible tensor shapes.

Optimizer: AdamW, finite warmup and cosine schedule, gradient clipping with nonfinite detection, exponential moving-average of parameters, best validation checkpoint, deterministic seeded minibatch sampling, and explicit 1000 *executed* optimizer steps. Serialization saves a tensor-only state_dict, loaded with weights_only=True and strict shape checks; load only trusted checkpoints.

The research reference provides a shadow-only API: FieldGenerator, PhiMAutoEncoder, train, load_model, encode_field and decode_code. Neither trained weights nor latent coordinates are automatically passed to Jarvis-X runtime control.

## Evidence (local CPU, 29 Oct 2026)

A separately tested local, more extensive reference (source package supplied in the conversation) trained the same 16³ convolutional architecture for 1,000 steps using 48 training and 12 independently seeded validation fields; validation MSE ~0.00045295 versus zero-output MSE ~0.31756. Train-loop time ~6.48 s on that CPU, excluding field generation and graphics. This is **not** a benchmark across other hardware, user documents, visual media or tasks.

A separate held-out latent probe on that local reference measured R² ~0.970 predicting time but ~-0.008 predicting amplitude with ridge regression. This negative value invalidates any claim that both generating parameters were reliably inferred. Independent SciPy and complex-step numerical checks passed but are not mathematical proofs of trained model accuracy.

The GitHub-side independent check is the dedicated phi-m-autoencoder.yml workflow: 8³ exact 1000-update synthetic training, distinct validation benchmark, optional-model I/O checks, and reproducibility artifact. Confirm GitHub CI before promoting this candidate.

## Limitations

- Not a guaranteed compressor (codebook framing, floating-point payload, model weights and headers are excluded).
- Not a demonstrated 1,000× speedup; the fast analytical Phi_M kernel affects only part of the pipeline.
- The model does not provide a unique inverse of Phi_M or exact bitstream reconstruction.
- No self-modifying source, continuous unsupervised deployment, device control, banking, clinical automation or live inference.
- Four local diagnostic figures and independent latent-probe results ship in the accompanying local source archive; the lean GitHub baseline is deliberately separate from those optional visualization extensions.
