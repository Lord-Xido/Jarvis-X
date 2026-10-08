from pathlib import Path
import numpy as np
import pytest
import torch
from dm3d_multimodal.data import SceneDataset, collate_one, decode_text, save_dataset_archive
from dm3d_multimodal.model import DM3DMultiVAE, ModelConfig, compute_losses
from dm3d_multimodal.training import save_checkpoint, load_checkpoint, train, set_seed
from dm3d_multimodal.store import MemoryStore
from dm3d_multimodal.media import save_media

@pytest.fixture(scope='module')
def model():
    set_seed(4)
    return DM3DMultiVAE()


def test_four_modalities_and_scene_alignment():
    ex = SceneDataset(4)[1]
    assert ex['text'].shape == (8,)
    assert ex['image'].shape == (3, 16, 16)
    assert ex['audio'].shape == (256,)
    assert ex['video'].shape == (4, 3, 16, 16)
    assert torch.allclose(ex['image'], ex['video'][0])
    assert 'blue' in decode_text(ex['text'])


def test_encoder_3d_and_512_latent(model):
    sample = collate_one(SceneDataset(1)[0])
    mu, lv, h = model.encode(sample)
    assert mu.shape == (1, 512) and lv.shape == (1, 512)
    assert h.shape == (1, 8, 4, 4, 4)
    out = model.decode(mu)
    assert out['text_logits'].shape == (1, 8, 16)
    assert out['image'].shape == sample['image'].shape
    assert out['audio'].shape == sample['audio'].shape
    assert out['video'].shape == sample['video'].shape


def test_real_backprop_and_optimizer_step(model):
    sample = collate_one(SceneDataset(1)[0])
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003)
    before = model.to_mu.weight.detach().clone()
    output, mu, logvar = model(sample, sample=True)
    losses = compute_losses(output, sample, mu, logvar)
    assert all(torch.isfinite(value).item() for value in losses.values())
    optimizer.zero_grad()
    losses['total'].backward()
    assert model.to_mu.weight.grad is not None
    assert torch.isfinite(model.to_mu.weight.grad).all().item()
    optimizer.step()
    assert not torch.equal(model.to_mu.weight.detach(), before)


def test_mask_validates_one_present_modality(model):
    sample = collate_one(SceneDataset(1)[0])
    mu, lv, h = model.encode(sample, torch.tensor([[1., 0., 1., 0.]]))
    assert torch.isfinite(mu).all()
    with pytest.raises(ValueError):
        model.encode(sample, torch.zeros(1, 4))


def test_anchored_refinement_is_finite(model):
    sample = collate_one(SceneDataset(1)[0])
    output, z, metrics = model.refine(sample, alpha=.3, max_steps=9, tolerance=.01)
    assert z.shape == (1, 512)
    assert 1 <= metrics['steps'] <= 9
    assert np.isfinite(metrics['residuals']).all()
    assert metrics['converged'] == (metrics['residuals'][-1] <= .01)
    assert torch.isfinite(output['audio']).all()


def test_checkpoint_roundtrip(tmp_path, model):
    model.eval()
    ckpt = tmp_path / 'model.pt'
    save_checkpoint(model, ckpt)
    loaded = load_checkpoint(ckpt)
    x = collate_one(SceneDataset(1)[0])
    with torch.no_grad():
        assert torch.equal(model.encode(x)[0], loaded.encode(x)[0])


def test_sqlite_vector_index_and_persistence(tmp_path):
    path = tmp_path / 'memory.sqlite'
    with MemoryStore(path) as store:
        store.add(1, [1, 3, 2], 'red scene', 'train', 'source.npz', np.ones(512, np.float32))
        store.add(2, [1, 4, 2], 'blue scene', 'validation', 'source.npz', -np.ones(512, np.float32))
        store.react(1, 'manual_test', 0.1)
        rows = store.top_k(np.ones(512), k=2)
        assert [r['sample_id'] for r in rows] == [1, 2]
        assert abs(rows[0]['similarity'] - 1) < 1e-5
        assert store.counts() == {'Messages': 2, 'Media': 6, 'Reactions': 1, 'Embeddings': 2}
    with MemoryStore(path) as store:
        assert store.counts()['Embeddings'] == 2
        with pytest.raises(ValueError):
            store.top_k(np.ones(10), k=1)


def test_media_file_formats(tmp_path, model):
    x = collate_one(SceneDataset(1)[0])
    with torch.no_grad():
        out = model.decode(model.encode(x)[0])
    paths = save_media(tmp_path / 'decoded', out)
    assert Path(paths['image_png']).stat().st_size > 0
    assert Path(paths['audio_wav']).stat().st_size > 0
    assert Path(paths['video_gif']).stat().st_size > 0
    if paths['video_mp4']:
        assert Path(paths['video_mp4']).stat().st_size > 0


def test_small_real_training_and_archive(tmp_path):
    set_seed(3)
    trainset, validset = SceneDataset(12), SceneDataset(4, start=12)
    model = DM3DMultiVAE(ModelConfig(latent_dim=32))
    history = train(model, trainset, validset, epochs=1, batch_size=4, out_dir=tmp_path)
    assert len(history) == 2
    assert np.isfinite(history[1]['validation']['total'])
    assert (tmp_path / 'dm3d_weights.pt').exists()
    count = save_dataset_archive(tmp_path / 'sources.npz', [trainset, validset])
    assert count == 16
    with np.load(tmp_path / 'sources.npz') as archive:
        assert archive['video'].shape == (16, 4, 3, 16, 16)


def test_train_split_filter_and_latent_recovery(tmp_path):
    with MemoryStore(tmp_path / 'retrieval.sqlite') as store:
        store.add(7, [1, 3, 2], 'train-red', 'train', 'sources.npz', np.ones(512))
        store.add(8, [1, 4, 2], 'validation-red', 'validation', 'sources.npz', np.ones(512)*.9)
        rows = store.top_k(np.ones(512), k=5, split='train')
        assert len(rows) == 1 and rows[0]['sample_id'] == 7
        assert np.allclose(store.embedding(7), np.ones(512))
        with pytest.raises(KeyError):
            store.embedding(10)


def test_real_file_ingestion_of_exported_media(tmp_path, model):
    from dm3d_multimodal.inputs import load_local
    with torch.no_grad():
        x = collate_one(SceneDataset(1)[0])
        media = save_media(tmp_path / 'source', model.decode(model.encode(x)[0]))
    if media['video_mp4'] is None:
        pytest.skip('FFmpeg MP4 support not available')
    result = load_local('red circle scene tone fast dark',
                        media['image_png'], media['audio_wav'], media['video_mp4'])
    assert result['text'].shape == (1, 8)
    assert result['image'].shape == (1, 3, 16, 16)
    assert result['audio'].shape == (1, 256)
    assert result['video'].shape == (1, 4, 3, 16, 16)
    with torch.no_grad():
        mu, _, _ = model.encode(result)
        assert torch.isfinite(mu).all()
