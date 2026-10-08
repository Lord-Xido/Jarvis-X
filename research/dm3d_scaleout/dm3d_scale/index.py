"""Bounded-memory persistent exact-cosine retrieval using immutable float32 segments.

Not an ANN index. A verified baseline which can be replaced with Faiss IVF/PQ.
"""
from __future__ import annotations
import heapq
import json
from pathlib import Path
import numpy as np

class SegmentIndex:
    def __init__(self, root, dimension=512):
        self.root=Path(root)
        self.root.mkdir(parents=True,exist_ok=True)
        self.dimension=int(dimension)
        if self.dimension<1 or self.dimension>4096:
            raise ValueError('invalid dimension')
        self.manifest_path=self.root/'segments.json'
        if self.manifest_path.exists():
            self.manifest=json.loads(self.manifest_path.read_text())
            if self.manifest['dimension']!=self.dimension:
                raise ValueError('embedding dimension mismatch')
        else:
            self.manifest={'version':1,'dimension':self.dimension,'segments':[]}

    def append(self, ids, vectors):
        ids=np.asarray(ids,dtype=np.int64)
        vectors=np.asarray(vectors,dtype=np.float32)
        if vectors.ndim!=2 or vectors.shape!=(len(ids),self.dimension) or not len(ids):
            raise ValueError('invalid vectors shape')
        if not np.isfinite(vectors).all() or len(set(ids.tolist()))!=len(ids):
            raise ValueError('duplicate ids or non-finite vectors')
        for item in self.manifest['segments']:
            old=np.load(self.root/item['ids'],mmap_mode='r')
            if np.intersect1d(old,ids).size:
                raise ValueError('duplicate id across segments')
        norm=np.linalg.norm(vectors,axis=1,keepdims=True)
        vectors=vectors/np.maximum(norm,1e-12)
        idx=len(self.manifest['segments'])
        filename=f'segment-{idx:06d}'
        np.save(self.root/f'{filename}-ids.npy',ids)
        np.save(self.root/f'{filename}-vectors.npy',vectors)
        np.save(self.root/f'{filename}-norms.npy',norm.squeeze(1))
        entry={'ids':f'{filename}-ids.npy','vectors':f'{filename}-vectors.npy',
               'norms':f'{filename}-norms.npy','count':len(ids)}
        # Atomic index publication, after all segment data has been written.
        next_manifest={**self.manifest,'segments':[*self.manifest['segments'],entry]}
        temp=self.root/'segments.json.tmp'
        temp.write_text(json.dumps(next_manifest,indent=2))
        temp.replace(self.manifest_path)
        self.manifest=next_manifest

    def topk(self, query, k=5, block=4096, exclude=None):
        q=np.asarray(query,dtype=np.float32)
        if q.shape!=(self.dimension,) or not np.isfinite(q).all() or not 1<=k<=1000 or block<1:
            raise ValueError('invalid query')
        q=q/max(float(np.linalg.norm(q)),1e-12)
        heap=[]
        for segment in self.manifest['segments']:
            ids=np.load(self.root/segment['ids'],mmap_mode='r')
            vectors=np.load(self.root/segment['vectors'],mmap_mode='r')
            for start in range(0,len(ids),block):
                scores=np.asarray(vectors[start:start+block])@q
                chunk_ids=np.asarray(ids[start:start+block])
                # Vectorized per-block candidates rather than a Python loop over every record.
                # Lexsort ensures deterministic smaller-ID tie breaking.
                candidates=np.lexsort((chunk_ids,-scores))[:k + (1 if exclude is not None else 0)]
                for j in candidates:
                    key=int(chunk_ids[j])
                    if key==exclude:
                        continue
                    item=(float(scores[j]),-key,key)
                    if len(heap)<k:
                        heapq.heappush(heap,item)
                    elif item>heap[0]:
                        heapq.heapreplace(heap,item)
        return [{'id':item[2],'cosine':item[0]}
                for item in sorted(heap,reverse=True)]


    def embedding(self, sample_id):
        """Reconstruct the original encoder vector for conditioning (not normalized)."""
        for segment in self.manifest['segments']:
            ids=np.load(self.root/segment['ids'],mmap_mode='r')
            matching=np.flatnonzero(ids==sample_id)
            if matching.size:
                j=int(matching[0])
                vec=np.load(self.root/segment['vectors'],mmap_mode='r')[j]
                scale=(float(np.load(self.root/segment['norms'],mmap_mode='r')[j])
                       if 'norms' in segment else 1.0)
                return np.asarray(vec,dtype=np.float32).copy()*scale
        raise KeyError(sample_id)
