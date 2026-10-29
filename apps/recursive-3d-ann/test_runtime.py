import math
from pathlib import Path

import pytest

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
    assert out['pipeline_wall_seconds_excluding_json'] >= out['generation_seconds']
    assert out['pipeline_tokens_per_second_excluding_json'] > 0
    assert out['pipeline_tokens_per_second_excluding_json'] <= (
        out['generation_only_tokens_per_second']
    )
    assert out['storage_floor']['input_token_id_bytes'] == 64000
    assert out['storage_floor']['float32_latent_bytes'] == 1000 * 24 * 4
    assert out['storage_floor']['xor_residual_bytes'] == 64000
    assert out['storage_floor']['payload_floor_ratio'] == 2.5
    assert not out['storage_floor']['compressed_without_model_weights']
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


def test_latent_dynamics_reports_one_and_two_step_residuals():
    _, autoencoder, _, reference = runtime.load_models(ROOT / 'index.html')
    ids = torch.tensor([reference['selected_streams'][0]['token_ids']])
    initial, final, history = runtime.feedback(autoencoder, ids, iterations=3)
    assert initial.shape == final.shape == (1, 24, 1, 1, 1)
    assert [h['stage'] for h in history] == list(range(4))
    assert history[0]['one_step_residual_norm'] is None
    assert history[0]['two_step_residual_norm'] is None
    assert history[1]['one_step_residual_norm'] is not None
    assert history[1]['two_step_residual_norm'] is None
    for h in history[2:]:
        assert math.isfinite(h['one_step_residual_norm'])
        assert math.isfinite(h['two_step_residual_norm'])
        assert h['one_step_residual_norm'] >= 0
        assert h['two_step_residual_norm'] >= 0


@pytest.mark.parametrize(
    'streams,temperature,topk',
    [(0, 0.8, 4), (1, 0, 4), (1, float('inf'), 4),
     (1, float('nan'), 4), (1, 0.8, 0)],
)
def test_generation_rejects_invalid_inputs(streams, temperature, topk):
    predictor, _, _, _ = runtime.load_models(ROOT / 'index.html')
    with pytest.raises(ValueError):
        runtime.generate(predictor, streams, 123, temperature, topk)


@pytest.mark.parametrize('iterations,gain', [(-1, 0.25), (1001, 0.25), (1, 0), (1, 1.1)])
def test_feedback_rejects_invalid_control_parameters(iterations, gain):
    _, model, _, _ = runtime.load_models(ROOT / 'index.html')
    with pytest.raises(ValueError):
        runtime.feedback(model, torch.zeros((1, 64), dtype=torch.long), iterations, gain)


def test_output_vocabulary_and_sampled_sequence_are_bounded():
    predictor, _, vocabulary, _ = runtime.load_models(ROOT / 'index.html')
    ids, _ = runtime.generate(predictor, 4, 20261029, 0.8, 4)
    assert ids.shape == (4, 64)
    assert torch.all(ids >= 1)
    assert torch.all(ids < len(vocabulary))
