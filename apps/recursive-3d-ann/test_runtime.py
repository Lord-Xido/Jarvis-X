import math
from pathlib import Path

import torch
import runtime

ROOT = Path(__file__).resolve().parent


def test_embedded_pytorch_weights_roundtrip_reference():
    _, model, vocabulary, reference = runtime.load_models(ROOT / 'index.html')
    sample = torch.tensor([reference['selected_streams'][0]['token_ids']])
    with torch.no_grad():
        logits, latent = model(sample)
    assert logits.argmax(-1)[0].tolist() == reference['volume_example']['decoded_ids']
    expected = torch.tensor(reference['volume_example']['input_latent_24'])
    assert torch.allclose(latent.flatten(), expected, rtol=1e-4, atol=1e-4)
    assert len(vocabulary) == 27


def test_complete_1000_stream_pipeline():
    out = runtime.run(ROOT / 'index.html', streams=1000, iterations=8)
    assert out['generated_tokens'] == 64000
    assert out['residual_roundtrip_exact']
    assert out['finite_latent']
    assert out['latent_shape'] == [1000, 24, 1, 1, 1]
    assert len(out['feedback_history']) == 9
    assert 0 < out['decoder_accuracy'] < 1
    assert all(math.isfinite(s['latent_norm']) for s in out['feedback_history'])


def test_seeded_sampling_determinism_and_validation():
    predictor, _, _, _ = runtime.load_models(ROOT / 'index.html')
    a, _ = runtime.generate(predictor, 8, 52, 0.8, 4)
    b, _ = runtime.generate(predictor, 8, 52, 0.8, 4)
    assert torch.equal(a, b)
    assert a.shape == (8, 64)
    assert (a > 0).all()
