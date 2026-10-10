import json
import torch
from dm3d_multimodal.data import RealSceneDataset, SceneDataset, collate_one
from dm3d_multimodal.model import DM3DMultiVAE, ModelConfig
from dm3d_multimodal.media import save_media


def test_real_dataset_manifest_and_typechecking(tmp_path):
    model = DM3DMultiVAE(ModelConfig(latent_dim=32)).eval()
    scene = collate_one(SceneDataset(1)[0])
    with torch.no_grad():
        result = model.decode(model.encode(scene)[0])
    assets = save_media(tmp_path/'user_source', result)
    rows = [{
        'split': 'train' if i<8 else 'validation',
        'text': 'blue circle moving tone fast bright',
        'image': assets['image_png'], 'audio': assets['audio_wav'],
        'video': assets['video_mp4'] or assets['video_gif'],
        'label': 'real ingestion test'
    } for i in range(12)]
    manifest = tmp_path / 'aligned.json'
    manifest.write_text(json.dumps(rows))
    train = RealSceneDataset(manifest, split='train')
    valid = RealSceneDataset(manifest, split='validation')
    assert len(train)==8 and len(valid)==4
    entry = valid[0]
    assert entry['text'].shape == (8,)
    assert entry['video'].shape == (4,3,16,16)
    with torch.no_grad():
        z, logvar, grid = model.encode(collate_one(entry))
        assert z.shape == (1,32)
        assert grid.shape == (1,8,4,4,4)
