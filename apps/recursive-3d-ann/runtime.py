#!/usr/bin/env python3
"""Inference from the weights embedded in the Jarvis-X recursive 3D ANN web app.

This is a trained, synthetic-vocabulary ANN and 3D convolutional autoencoder,
not a large language model, physical 3D computer, or lossless compressor.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

SIDE = 4
TOKENS = SIDE ** 3
LATENT = 24


class NextTokenANN(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 24)
        self.network = nn.Sequential(nn.Linear(24, 48), nn.GELU(), nn.Linear(48, vocab_size))

    def forward(self, previous: torch.Tensor) -> torch.Tensor:
        return self.network(self.embedding(previous))


class SpatialAutoencoder(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 12)
        self.down1 = nn.Conv3d(12, 24, 3, stride=2, padding=1)
        self.down2 = nn.Conv3d(24, LATENT, 2, stride=2)
        self.up1 = nn.ConvTranspose3d(LATENT, 24, 2, stride=2)
        self.up2 = nn.ConvTranspose3d(24, 24, 2, stride=2)
        self.readout = nn.Conv3d(24, vocab_size, 1)
        self.vocab_size = vocab_size

    def encode_embedded(self, embedded: torch.Tensor) -> torch.Tensor:
        grid = embedded.transpose(1, 2).reshape(-1, 12, SIDE, SIDE, SIDE)
        return self.down2(F.gelu(self.down1(grid)))

    def encode(self, ids: torch.Tensor) -> torch.Tensor:
        return self.encode_embedded(self.embedding(ids))

    def decode_logits(self, latent: torch.Tensor) -> torch.Tensor:
        expanded = F.gelu(self.up2(F.gelu(self.up1(latent))))
        return self.readout(expanded).permute(0, 2, 3, 4, 1).reshape(-1, TOKENS, self.vocab_size)

    def forward(self, ids: torch.Tensor):
        latent = self.encode(ids)
        return self.decode_logits(latent), latent

    def reencode_soft(self, logits: torch.Tensor) -> torch.Tensor:
        expected_embeddings = F.softmax(logits, -1) @ self.embedding.weight
        return self.encode_embedded(expected_embeddings)


def extract_json(document: str, script_id: str):
    pattern = rf'<script\s+id="{re.escape(script_id)}"\s+type="application/json">(.*?)</script>'
    match = re.search(pattern, document, re.DOTALL)
    if not match:
        raise ValueError(f'Missing embedded model payload: {script_id}')
    return json.loads(match.group(1))


def load_models(html_path: Path):
    document = html_path.read_text(encoding='utf-8')
    model_data = extract_json(document, 'model-data')
    reference = extract_json(document, 'results-data')
    vocabulary = model_data['vocabulary']
    arrays = model_data['weights']
    models = {'next_token': NextTokenANN(len(vocabulary)),
              'autoencoder': SpatialAutoencoder(len(vocabulary))}
    for prefix, model in models.items():
        state = model.state_dict()
        for name, original in state.items():
            key = prefix + '.' + name
            if key not in arrays:
                raise ValueError(f'Missing trained weight tensor: {key}')
            data = torch.tensor(arrays[key], dtype=original.dtype)
            if data.numel() != original.numel():
                raise ValueError(f'Shape mismatch in {key}')
            state[name] = data.reshape(original.shape)
        model.load_state_dict(state, strict=True)
        model.eval()
    return models['next_token'], models['autoencoder'], vocabulary, reference


@torch.inference_mode()
def generate(model, streams: int, seed: int, temperature: float, topk: int):
    if streams < 1 or not 0.0 < temperature or topk < 1:
        raise ValueError('streams and topk must be positive, temperature > 0')
    count = model.embedding.num_embeddings
    previous = torch.zeros(streams, dtype=torch.long)
    tokens = torch.empty((streams, TOKENS), dtype=torch.long)
    random = torch.Generator().manual_seed(seed)
    t0 = time.perf_counter()
    for step in range(TOKENS):
        logits = model(previous) / temperature
        # The browser sampler excludes BOS from generated tokens.
        logits[:, 0] = -torch.inf
        values, indices = logits.topk(min(topk, count - 1), dim=-1)
        probabilities = F.softmax(values, dim=-1)
        sampled = torch.multinomial(probabilities, 1, generator=random)
        previous = indices.gather(1, sampled).flatten()
        tokens[:, step] = previous
    return tokens, time.perf_counter() - t0


@torch.inference_mode()
def feedback(model, ids: torch.Tensor, iterations: int = 8, gain: float = 0.25):
    if not (0 <= iterations <= 1000 and 0 < gain <= 1):
        raise ValueError('invalid feedback settings')
    initial = model.encode(ids)
    latent = initial.clone()
    history = []
    for stage in range(iterations + 1):
        logits = model.decode_logits(latent)
        reconstructed = logits.argmax(-1)
        history.append({'stage': stage,
                        'accuracy': float((reconstructed == ids).float().mean()),
                        'latent_norm': float(torch.linalg.vector_norm(latent)),
                        'delta_from_initial': float(torch.linalg.vector_norm(latent - initial))})
        if stage < iterations:
            latent = (1.0 - gain) * latent + gain * model.reencode_soft(logits)
            if not torch.isfinite(latent).all():
                raise ArithmeticError('nonfinite latent state')
    return initial, latent, history


@torch.inference_mode()
def run(html_path: Path, streams: int = 1000, seed: int = 20261029,
        temperature: float = 0.8, topk: int = 4, iterations: int = 8):
    predictor, autoencoder, vocabulary, reference = load_models(html_path)
    ids, generation_seconds = generate(predictor, streams, seed, temperature, topk)
    logits, latent = autoencoder(ids)
    decoded = logits.argmax(-1)
    original_bytes = ids.to(torch.uint8)
    approximate_bytes = decoded.to(torch.uint8)
    residual = original_bytes ^ approximate_bytes
    lossless = bool(torch.equal(approximate_bytes ^ residual, original_bytes))
    _, recurrent, trajectory = feedback(autoencoder, ids[:1], iterations)
    return {'kind': 'learned synthetic ANN + 3D conv autoencoder',
            'vocabulary_size': len(vocabulary), 'streams': streams,
            'generated_tokens': ids.numel(), 'generation_seconds': generation_seconds,
            'generation_only_tokens_per_second': ids.numel() / generation_seconds,
            'decoder_accuracy': float((decoded == ids).float().mean()),
            'residual_roundtrip_exact': lossless,
            'latent_shape': list(latent.shape), 'finite_latent': bool(torch.isfinite(recurrent).all()),
            'feedback_history': trajectory, 'first_stream_tokens': ids[0].tolist(),
            'first_stream_text': ''.join(vocabulary[i] for i in ids[0].tolist()),
            'reference_seed': reference.get('reproducible_seed'),
            'limitations': ['one-token context and 27-symbol vocabulary',
                            'generation-only timing; no GPU benchmark',
                            'XOR residual requires extra storage',
                            'no mathematically established fixed-point contraction']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--html', type=Path, default=Path(__file__).with_name('index.html'))
    parser.add_argument('--streams', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=20261029)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--topk', type=int, default=4)
    parser.add_argument('--iterations', type=int, default=8)
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()
    torch.set_num_threads(min(2, torch.get_num_threads()))
    result = run(args.html, args.streams, args.seed, args.temperature, args.topk, args.iterations)
    encoded = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(encoded + '\n', encoding='utf-8')
    print(encoded)


if __name__ == '__main__':
    main()
