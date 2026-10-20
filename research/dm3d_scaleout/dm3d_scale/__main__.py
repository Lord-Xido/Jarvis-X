"""Commands: shard, train, index, query, footprint."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .shards import prepare, MMapSceneDataset
from .index import SegmentIndex
from .tiles import estimate_virtual


def main():
    p=argparse.ArgumentParser(description='DM3D scale-out bounded-memory engine')
    commands=p.add_subparsers(dest='command',required=True)
    s=commands.add_parser('shard',help='prepare disk-backed aligned four-modality shards')
    s.add_argument('--output',required=True)
    s.add_argument('--manifest')
    s.add_argument('--manifest-jsonl',help='scalable line-oriented aligned local media manifest')
    s.add_argument('--train-count',type=int,default=96)
    s.add_argument('--validation-count',type=int,default=24)
    s.add_argument('--shard-size',type=int,default=16)
    s.add_argument('--seed',type=int,default=7)
    t=commands.add_parser('train',help='train CPU, GPU or DDP distributed workers')
    t.add_argument('--output',required=True)
    t.add_argument('--epochs',type=int,default=2)
    t.add_argument('--batch-size',type=int,default=4)
    t.add_argument('--accumulation',type=int,default=2)
    t.add_argument('--workers',type=int,default=0)
    t.add_argument('--latent-dim',type=int,default=512)
    t.add_argument('--lr',type=float,default=.001)
    t.add_argument('--seed',type=int,default=7)
    t.add_argument('--amp',choices=['auto','none','bf16','fp16'],default='auto')
    t.add_argument('--resume',action='store_true')
    f2=commands.add_parser('train-fsdp2',help='multi-rank fully sharded FSDP2 model training')
    f2.add_argument('--output',required=True)
    f2.add_argument('--epochs',type=int,default=1)
    f2.add_argument('--batch-size',type=int,default=2)
    f2.add_argument('--latent-dim',type=int,default=512)
    f2.add_argument('--lr',type=float,default=.001)
    f2.add_argument('--seed',type=int,default=7)
    f2.add_argument('--resume',action='store_true')
    f2.add_argument('--export-full',action='store_true',help='gather model to CPU rank 0 for retrieval; requires enough host RAM')
    ix=commands.add_parser('index',help='compute persistent segmented exact vector index')
    ix.add_argument('--output',required=True)
    ix.add_argument('--segment-size',type=int,default=128)
    q=commands.add_parser('query',help='lookup sample by its training embedding')
    q.add_argument('--output',required=True)
    q.add_argument('--sample-id',type=int,default=0)
    q.add_argument('--top-k',type=int,default=5)
    gen=commands.add_parser('generate',help='retrieve, refine, decode and export four modalities')
    gen.add_argument('--output',required=True)
    gen.add_argument('--sample-id',type=int,required=True)
    gen.add_argument('--split',choices=['train','validation'],default='validation')
    gen.add_argument('--top-k',type=int,default=3)
    gen.add_argument('--alpha',type=float,default=.3)
    gen.add_argument('--tolerance',type=float,default=.002)
    gen.add_argument('--max-steps',type=int,default=12)
    f=commands.add_parser('footprint',help='calculate memory for a virtual image lattice')
    f.add_argument('--height',type=int,default=8000000)
    f.add_argument('--width',type=int,default=8000000)
    f.add_argument('--tile',type=int,default=2048)
    f.add_argument('--slots',type=int,default=122)
    a=p.parse_args()
    if a.command=='shard':
        print(json.dumps(prepare(Path(a.output)/'shards',train_count=a.train_count,
          validation_count=a.validation_count,shard_size=a.shard_size,seed=a.seed,manifest=a.manifest,
          manifest_jsonl=a.manifest_jsonl),indent=2))
    elif a.command=='train':
        from .distributed import run
        run(a.output,epochs=a.epochs,batch_size=a.batch_size,
            accumulation=a.accumulation,workers=a.workers,latent_dim=a.latent_dim,
            lr=a.lr,seed=a.seed,resume=a.resume,amp=a.amp)
    elif a.command=='train-fsdp2':
        from .fsdp2 import train_fsdp2
        train_fsdp2(a.output,epochs=a.epochs,batch_size=a.batch_size,
                    latent_dim=a.latent_dim,lr=a.lr,seed=a.seed,resume=a.resume,export_full=a.export_full)
    elif a.command=='index':
        from dm3d_multimodal.model import DM3DMultiVAE, ModelConfig
        state=torch.load(Path(a.output)/'checkpoint.pt',map_location='cpu',weights_only=True)
        model=DM3DMultiVAE(ModelConfig(latent_dim=state['latent_dim']))
        model.load_state_dict(state['model'])
        model.eval()
        dataset=MMapSceneDataset(Path(a.output)/'shards'/'train')
        if a.segment_size<1 or a.segment_size>8192:raise ValueError('invalid segment size')
        index=SegmentIndex(Path(a.output)/'index',dimension=model.config.latent_dim)
        with torch.no_grad():
            for batch in DataLoader(dataset,batch_size=a.segment_size):
                x={k:batch[k] for k in ('text','image','audio','video')}
                z,_,_=model.encode(x)
                index.append(batch['sample_id'].numpy(),z.numpy())
        print(json.dumps({'segments':len(index.manifest['segments']),
             'vectors':sum(s['count'] for s in index.manifest['segments'])}))
    elif a.command=='query':
        import bisect
        from dm3d_multimodal.model import DM3DMultiVAE, ModelConfig
        checkpoint=torch.load(Path(a.output)/'checkpoint.pt',map_location='cpu',weights_only=True)
        model=DM3DMultiVAE(ModelConfig(latent_dim=checkpoint['latent_dim']))
        model.load_state_dict(checkpoint['model']);model.eval()
        dataset=MMapSceneDataset(Path(a.output)/'shards'/'train')
        sample=None
        for i in range(len(dataset)):
            row=dataset[i]
            if int(row['sample_id'])==a.sample_id:
                sample=row;break
        if sample is None:raise ValueError('unknown sample_id')
        with torch.no_grad():
            x={k:sample[k].unsqueeze(0) for k in ('text','image','audio','video')}
            mu,_,_=model.encode(x)
        index=SegmentIndex(Path(a.output)/'index',dimension=model.config.latent_dim)
        print(json.dumps(index.topk(mu[0].numpy(),k=a.top_k,exclude=a.sample_id),indent=2))
    elif a.command=='generate':
        from .generation import generate
        print(json.dumps(generate(a.output,a.sample_id,split=a.split,
             alpha=a.alpha,tolerance=a.tolerance,max_steps=a.max_steps,topk=a.top_k),indent=2))
    elif a.command=='footprint':
        print(json.dumps(estimate_virtual(a.height,a.width,a.tile,active_slots=a.slots),indent=2))

if __name__=='__main__':
    main()
