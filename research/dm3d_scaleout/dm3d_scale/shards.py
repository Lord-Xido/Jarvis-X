"""Disk-backed, aligned multimodal sample shards. Constant number of mapped files per worker."""
from __future__ import annotations
import bisect
import json
from collections import OrderedDict
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from dm3d_multimodal.data import IMAGE_SIDE, AUDIO_SAMPLES, FRAMES, SEQ_LEN, SceneDataset, RealSceneDataset

SCHEMA = {
    'text': ('int64', (SEQ_LEN,)),
    'image': ('float32', (3, IMAGE_SIDE, IMAGE_SIDE)),
    'audio': ('float32', (AUDIO_SAMPLES,)),
    'video': ('float32', (FRAMES, 3, IMAGE_SIDE, IMAGE_SIDE)),
    'sample_id': ('int64', ()),
}


def write_shards(dataset, directory, shard_size=256):
    """Write one dataset split into bounded sized, memory-mapped .npy shards.

    Each shard is published only after the arrays have been flushed and a metadata file
    is written. This is intentionally a single-writer, offline preparation step.
    """
    if not 1 <= shard_size <= 65536:
        raise ValueError('shard_size must be in 1..65536')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / 'index.json').exists():
        raise FileExistsError('refusing to overwrite existing shard index')
    shards = []
    for start in range(0, len(dataset), shard_size):
        size = min(shard_size, len(dataset)-start)
        name = f'shard-{len(shards):06d}'
        target = directory / name
        target.mkdir(exist_ok=False)
        fields = {}
        for key, (dtype, shape) in SCHEMA.items():
            fields[key] = np.lib.format.open_memmap(
                target / f'{key}.npy', mode='w+', dtype=dtype, shape=(size, *shape))
        for j in range(size):
            sample = dataset[start+j]
            for key, (_, shape) in SCHEMA.items():
                a = np.asarray(sample[key])
                if a.shape != shape:
                    raise ValueError(f'invalid shape for {key}: {a.shape} instead of {shape}')
                if key in ('image', 'audio', 'video') and not np.isfinite(a).all():
                    raise ValueError('non-finite input')
                fields[key][j] = a
        for arr in fields.values():
            arr.flush()
        fields.clear()
        shards.append({'name': name, 'count': size})
    manifest = {'version': 1, 'schema': SCHEMA, 'shards': shards,
                'total': sum(s['count'] for s in shards), 'shard_size': shard_size}
    tmp = directory / 'index.json.tmp'
    tmp.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    tmp.replace(directory / 'index.json')
    return manifest


class MMapSceneDataset(Dataset):
    """Map-style dataset: only a bounded LRU of small shard files is mapped per worker."""
    def __init__(self, directory, cache_shards=2):
        self.directory = Path(directory)
        manifest = json.loads((self.directory / 'index.json').read_text(encoding='utf-8'))
        if manifest['version'] != 1 or manifest['total'] <= 0:
            raise ValueError('invalid shard index')
        self.shards = manifest['shards']
        self.cumulative = [0]
        for shard in self.shards:
            self.cumulative.append(self.cumulative[-1]+shard['count'])
        self.limit = max(1, min(int(cache_shards), 8))
        self._mapped = OrderedDict()

    def __len__(self):
        return self.cumulative[-1]

    def __getstate__(self):
        state=self.__dict__.copy()
        state['_mapped']=OrderedDict()
        return state

    def __getitem__(self, i):
        if not 0 <= i < len(self):
            raise IndexError(i)
        shard_idx=bisect.bisect_right(self.cumulative, i)-1
        local_idx=i-self.cumulative[shard_idx]
        if shard_idx not in self._mapped:
            root=self.directory/self.shards[shard_idx]['name']
            mapped={key:np.load(root/f'{key}.npy',mmap_mode='r',allow_pickle=False)
                    for key in SCHEMA}
            self._mapped[shard_idx]=mapped
            if len(self._mapped)>self.limit:
                self._mapped.popitem(last=False)
        self._mapped.move_to_end(shard_idx)
        fields=self._mapped[shard_idx]
        return {key:torch.from_numpy(np.array(fields[key][local_idx],copy=True))
                for key in SCHEMA}


def prepare(out, *, train_count=96, validation_count=24, shard_size=16, seed=7,
            manifest=None, manifest_jsonl=None):
    out=Path(out)
    if manifest and manifest_jsonl:
        raise ValueError("use only one manifest type")
    if manifest_jsonl:
        train=AlignedJSONLDataset(manifest_jsonl,"train")
        valid=AlignedJSONLDataset(manifest_jsonl,"validation")
    elif manifest:
        train=RealSceneDataset(manifest,'train')
        valid=RealSceneDataset(manifest,'validation')
    else:
        train=SceneDataset(train_count,0,seed)
        valid=SceneDataset(validation_count,train_count,seed)
    a=write_shards(train,out/'train',shard_size)
    b=write_shards(valid,out/'validation',shard_size)
    return {'train':a['total'],'validation':b['total'], 'shard_size':shard_size,
            'source':'real-jsonl' if manifest_jsonl else ('real-manifest' if manifest else 'procedural-synthetic')}


class AlignedJSONLDataset(Dataset):
    """Lazy local JSONL manifest: O(number of records) 64-bit offsets; media stays on disk.

    One line per scene: {split, text, image, audio, video, sample_id?}.
    Uses per-item file open to avoid shared file handles across DataLoader workers.
    """
    def __init__(self, manifest_path, split):
        self.path=Path(manifest_path)
        self.base=self.path.parent
        self.offsets=[]
        with self.path.open('rb') as fp:
            while True:
                offset=fp.tell()
                line=fp.readline()
                if not line:break
                if not line.strip():continue
                meta=json.loads(line)
                if meta.get('split')==split:
                    self.offsets.append(offset)
        if not self.offsets:
            raise ValueError(f'no {split} records in JSONL manifest')

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, i):
        if not 0<=i<len(self.offsets):
            raise IndexError(i)
        from dm3d_multimodal.inputs import load_local
        with self.path.open('rb') as fp:
            fp.seek(self.offsets[i])
            row=json.loads(fp.readline())
        for key in ('text','image','audio','video'):
            if key not in row:
                raise ValueError(f'JSONL missing {key}')
        def local(name):
            path=Path(row[name])
            return str(path if path.is_absolute() else self.base/path)
        sample=load_local(row['text'],local('image'),local('audio'),local('video'))
        return {**{k:sample[k][0] for k in ('text','image','audio','video')},
                'sample_id':int(row.get('sample_id',self.offsets[i]))}
