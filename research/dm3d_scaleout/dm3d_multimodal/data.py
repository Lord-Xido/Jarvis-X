"""Reproducible, explicitly synthetic, aligned text/image/audio/video scenes."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

VOCAB = ['<pad>', '<bos>', '<eos>', 'red', 'blue', 'green', 'yellow',
         'circle', 'square', 'moving', 'tone', 'slow', 'fast', 'bright', 'dark', 'scene']
STOI = {v: i for i, v in enumerate(VOCAB)}
SEQ_LEN = 8
IMAGE_SIDE = 16
AUDIO_SAMPLES = 256
FRAMES = 4
COLORS = np.array([[0.93, 0.18, 0.13], [0.12, 0.42, 0.95],
                   [0.15, 0.86, 0.37], [0.92, 0.78, 0.12]], dtype=np.float32)


def generate_scene(index: int, seed: int = 7):
    """One aligned artificial event; no copyrighted, external or private data."""
    rng = np.random.default_rng(seed + 104729 * index)
    color_id = index % 4
    shape_id = (index // 4) % 2
    moving = (index // 8) % 2
    bright = (index // 16) % 2
    speed = (index // 32) % 2
    terms = [
        '<bos>', VOCAB[STOI['red'] + color_id],
        'circle' if shape_id == 0 else 'square',
        'moving' if moving else 'scene',
        'tone', 'fast' if speed else 'slow',
        'bright' if bright else 'dark', '<eos>',
    ]
    text = np.array([STOI[t] for t in terms], dtype=np.int64)
    yy, xx = np.mgrid[0:IMAGE_SIDE, 0:IMAGE_SIDE].astype(np.float32)
    dy = yy - (7.5 + float(rng.uniform(-0.5, 0.5)))
    phase = float(rng.uniform(-0.45, 0.45))
    radius = 3.5 + 0.25 * bright
    v = []
    for frame in range(FRAMES):
        dx = xx - (7.5 + phase + (frame - 1.5) * (0.8 if moving else 0))
        if shape_id == 0:
            mask = (dx ** 2 + dy ** 2 < radius ** 2)
        else:
            mask = (np.abs(dx) < radius) & (np.abs(dy) < radius)
        canvas = np.full((IMAGE_SIDE, IMAGE_SIDE, 3), 0.035, np.float32)
        canvas[mask] = COLORS[color_id] * (1.0 if bright else 0.68)
        v.append(canvas.transpose(2, 0, 1))
    video = np.stack(v).astype(np.float32)
    image = video[0].copy()
    # Physically constructed short synthetic waveform at 1024 Hz.
    t = np.arange(AUDIO_SAMPLES, dtype=np.float32) / 1024
    freq = [36, 52, 68, 84][color_id] + shape_id * 5
    carrier = np.sin(2 * np.pi * freq * t + phase)
    second = 0.18 * np.sin(2 * np.pi * (2 if speed else 1) * freq * t)
    envelope = (0.6 + 0.4 * np.cos(2 * np.pi * (4 if speed else 2) * t))
    audio = np.clip((0.68 if bright else 0.45) * (carrier + second) * envelope, -1, 1).astype(np.float32)
    label = f"{terms[1]} {terms[2]} {'moving' if moving else 'static'}"
    return {'text': text, 'image': image, 'audio': audio, 'video': video,
            'label': label, 'sample_id': int(index)}


class SceneDataset(Dataset):
    def __init__(self, count: int = 96, start: int = 0, seed: int = 7):
        if count < 1 or start < 0:
            raise ValueError('count must be positive and start nonnegative')
        self.count, self.start, self.seed = count, start, seed

    def __len__(self):
        return self.count

    def __getitem__(self, pos):
        if pos < 0 or pos >= self.count:
            raise IndexError(pos)
        sample = generate_scene(self.start + pos, self.seed)
        return {k: torch.from_numpy(sample[k]) if k in ('text', 'image', 'audio', 'video')
                else sample[k] for k in sample}


def collate_one(scene):
    return {k: scene[k].unsqueeze(0) for k in ('text', 'image', 'audio', 'video')}


def decode_text(token_ids):
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.detach().cpu().tolist()
    words = []
    for i in token_ids:
        if int(i) == STOI['<eos>']:
            break
        if int(i) not in (STOI['<pad>'], STOI['<bos>']):
            words.append(VOCAB[int(i)] if 0 <= int(i) < len(VOCAB) else '<?>')
    return ' '.join(words)


class RealSceneDataset(Dataset):
    """Aligned user-supplied media records, read locally without network access."""
    def __init__(self, manifest_path, split='train'):
        from .inputs import load_local
        path = Path(manifest_path)
        self.base = path.parent
        payload = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(payload, list) or len(payload) > 4096:
            raise ValueError('manifest must be a list of at most 4096 records')
        self.items = [(i, row) for i, row in enumerate(payload) if row.get('split') == split]
        if not self.items:
            raise ValueError(f'no {split} rows in manifest')
        for i, row in self.items:
            if not all(key in row for key in ('text', 'image', 'audio', 'video', 'split')):
                raise ValueError(f'incomplete record at manifest index {i}')

    def __len__(self):
        return len(self.items)

    def __getitem__(self, pos):
        if pos < 0 or pos >= len(self.items):
            raise IndexError(pos)
        from .inputs import load_local
        sample_id, row = self.items[pos]
        def source_path(key):
            v = Path(row[key])
            return str(v if v.is_absolute() else self.base / v)
        aligned = load_local(row['text'], source_path('image'), source_path('audio'), source_path('video'))
        return {**{k: aligned[k][0] for k in ('text', 'image', 'audio', 'video')},
                'label': row.get('label', row['text'][:96]), 'sample_id': sample_id}


def save_dataset_archive(path, datasets):
    """Persist preprocessed source tensors from either synthetic or real datasets."""
    scenes = [ds[i] for ds in datasets for i in range(len(ds))]
    arrays = {name: np.stack([np.asarray(s[name]) for s in scenes])
              for name in ('text', 'image', 'audio', 'video')}
    arrays['sample_id'] = np.array([int(s['sample_id']) for s in scenes], dtype=np.int64)
    arrays['label'] = np.array([s['label'] for s in scenes])
    np.savez_compressed(path, **arrays)
    return len(scenes)
