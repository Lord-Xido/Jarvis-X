# Neural Feedback Auto-Training Runtime

## Purpose

`jarvisx.neural_feedback_runtime` is a bounded reference implementation of the Jarvis-X neural feedback loop:

```text
OBSERVE -> ENCODE -> SIMULATE -> PREDICT -> VERIFY -> REPLAY -> TRAIN SHADOW -> PROMOTE -> REPEAT
```

The live web is treated as an **untrusted measurement source**, never as direct training truth. Retrieved text is converted into deterministic numeric features; retrieved content is not executed. Parameter updates occur only after a verification gate accepts evidence and a shadow candidate improves held-out reconstruction loss.

## State transition

For runtime state

```text
S_t = (X_t, Z_t, Xhat_t, Sigma_t, E_t, C_t, R_t, Theta_t, Omega_t)
```

the implementation realizes the bounded transition

```text
S_{t+1} = U_theta o U_omega o R o C o W o P o S o E(S_t)
```

where `E` hashes/encodes observations, `S` produces perturbation simulations, `P` reconstructs/predicts, `W` ingests a web JSON feed, `C/R` score independent supporting and counter-evidence, `U_omega` appends accepted vectors to replay, and `U_theta` trains/promotes a shadow model.

## Runtime invariants

1. **No direct web-to-gradient path.** Web records must cross the verification threshold before entering replay.
2. **No source execution.** Web text is featurized as data only.
3. **Independent-source weighting.** Duplicate source groups do not count as independent confirmations.
4. **Counterevidence penalty.** High counterevidence lowers the verification score and can block training.
5. **Simulation before update.** Accepted observations generate bounded dropout/noise variants used to test local stability.
6. **Shadow promotion.** Three learning-rate candidates are trained from the production checkpoint; only a candidate with lower held-out loss is promoted.
7. **Bounded memory.** Replay is a fixed-capacity FIFO buffer.
8. **Checkpointed state.** Production parameters and generation counters serialize to a compressed NumPy checkpoint.
9. **SSRF boundary.** Feed URLs must be HTTP(S), contain no credentials, and resolve only to public IP addresses. Optional domain allowlisting is supported.

## Quick start

Install the repository and run the offline deterministic simulation feed:

```bash
python -m pip install -e .
python -m jarvisx.neural_feedback_cli --cycles 5
```

A checkpoint named `jarvisx-neural-feedback.npz` is written by default.

Resume:

```bash
python -m jarvisx.neural_feedback_cli \
  --resume jarvisx-neural-feedback.npz \
  --cycles 5 \
  --checkpoint jarvisx-neural-feedback.npz
```

Machine-readable telemetry:

```bash
python -m jarvisx.neural_feedback_cli --cycles 3 --json
```

## Live web feed

The runtime deliberately does not implement a general crawler. A search/indexing layer should normalize retrieved public-web evidence into a bounded JSON feed and then hand it to the runtime.

Example:

```json
{
  "observations": [
    {
      "claim": "example-claim-id",
      "content": "Observed statement or normalized evidence text",
      "source": "https://research.example/paper/123",
      "source_group": "research.example",
      "authority": 0.9,
      "counterevidence": 0.05,
      "timestamp": "2026-09-08T07:00:00+02:00"
    }
  ]
}
```

Run against an explicitly selected feed:

```bash
python -m jarvisx.neural_feedback_cli \
  --feed-url https://research.example/jarvisx-feed.json \
  --allow-domain research.example \
  --cycles 5
```

`authority` and `counterevidence` must be in `[0, 1]`. A production search adapter should derive these values from auditable provenance rules rather than model intuition alone.

## Verification score

For a claim group, the reference gate computes

```text
R = clip(
      0.32 * authority
    + 0.24 * independence
    + 0.16 * cross_source_support
    + 0.16 * simulation_consistency
    + 0.12 * predictive_agreement
    - 0.35 * counterevidence,
    0, 1)
```

Only `R >= verification_threshold` reaches replay.

## Auto-evolution mechanism

Auto-evolution is parameter evolution, not autonomous source-code mutation:

1. snapshot the production autoencoder;
2. split replay into training and held-out validation sets;
3. train three shadow candidates at `0.5x`, `1.0x`, and `1.5x` the configured learning rate;
4. choose the candidate with the lowest validation MSE;
5. promote only if `candidate_loss + promotion_margin < production_loss`;
6. increment the production generation counter.

This preserves a rollback boundary and prevents a failed experimental update from replacing the production parameters.

## Tests

```bash
python -m pytest tests/test_neural_feedback_runtime.py -q
```

Coverage includes deterministic encoding, verification acceptance/rejection, candidate promotion, checkpoint round-trip, and rejection of local/private feed targets.

## Scope

This module is a reproducible runtime substrate, not a claim that arbitrary web text can safely or correctly train a frontier model online. A production deployment still needs a real retrieval system, provenance store, licensing/privacy policy, evaluation suites, model-specific optimizers, observability, and human governance around parameter promotion.
