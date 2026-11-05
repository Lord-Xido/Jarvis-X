"""Torchrun data-parallel training with measurable held-out validation.

This module uses DDP (replicated parameters), not FSDP parameter sharding.
"""
from __future__ import annotations
import contextlib
import json
import os
import random
from pathlib import Path
import numpy as np
import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Subset
from dm3d_multimodal.model import DM3DMultiVAE, ModelConfig, compute_losses
from .shards import MMapSceneDataset

KEYS=('text','image','audio','video')

def environment():
    world=int(os.environ.get('WORLD_SIZE','1'))
    rank=int(os.environ.get('RANK','0'))
    local=int(os.environ.get('LOCAL_RANK','0'))
    if world<1 or not 0<=rank<world:
        raise ValueError('invalid distributed environment')
    device=torch.device(f'cuda:{local}' if torch.cuda.is_available() else 'cpu')
    if device.type=='cuda':
        torch.cuda.set_device(device)
    if world>1 and not dist.is_initialized():
        dist.init_process_group('nccl' if device.type=='cuda' else 'gloo',init_method='env://')
    return world,rank,device

def rank_sum(x,device):
    t=torch.tensor(float(x),device=device,dtype=torch.float64)
    if dist.is_initialized():
        dist.all_reduce(t,op=dist.ReduceOp.SUM)
    return float(t.item())

def to_device(batch,device):
    return {key:batch[key].to(device,non_blocking=True) for key in KEYS}

@torch.no_grad()
def validate(model,loader,device):
    """Exact once-per-validation-sample distributed reduction; no padding duplicates."""
    model.eval()
    keys=['total','text_ce','image_mse_weighted','audio_mse','video_mse_weighted','kl','token_accuracy']
    totals={key:0. for key in keys};seen=0
    for batch in loader:
        x=to_device(batch,device)
        decoded,mu,logvar=model(x,sample=False)
        ls=compute_losses(decoded,x,mu,logvar)
        ls['token_accuracy']=(decoded['text_logits'].argmax(-1)==x['text']).float().mean()
        n=x['text'].size(0)
        for key in keys:
            totals[key]+=float(ls[key])*n
        seen+=n
    values=torch.tensor([seen]+[totals[k] for k in keys],device=device,dtype=torch.float64)
    if dist.is_initialized():
        dist.all_reduce(values,op=dist.ReduceOp.SUM)
    if values[0]<=0:
        raise ValueError('empty validation')
    return {key:float(values[i+1]/values[0]) for i,key in enumerate(keys)}

def atomic_checkpoint(state,path):
    temp=Path(str(path)+'.tmp')
    torch.save(state,temp)
    os.replace(temp,path)

def run(out,epochs=2,batch_size=4,accumulation=2,lr=1e-3,latent_dim=512,
        workers=0,seed=7,resume=False,amp='auto'):
    if epochs<1 or epochs>100000 or batch_size<1 or accumulation<1 or workers<0 or not 0<lr<.1:
        raise ValueError('invalid training settings')
    if amp not in ('auto','none','bf16','fp16'):
        raise ValueError('invalid amp mode')
    torch.set_num_threads(min(2,max(1,torch.get_num_threads())))
    world,rank,device=environment()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    out=Path(out)
    out.mkdir(parents=True,exist_ok=True)
    trainset=MMapSceneDataset(out/'shards'/'train')
    valset=MMapSceneDataset(out/'shards'/'validation')
    sampler=DistributedSampler(trainset,num_replicas=world,rank=rank,
                               shuffle=True,seed=seed,drop_last=False)
    trainloader=DataLoader(trainset,batch_size=batch_size,sampler=sampler,
                           num_workers=workers,pin_memory=device.type=='cuda')
    # Validation uses strided, disjoint partitions (DistributedSampler pads otherwise).
    val_part=Subset(valset,range(rank,len(valset),world))
    valloader=DataLoader(val_part,batch_size=batch_size,shuffle=False,num_workers=workers)
    base=DM3DMultiVAE(ModelConfig(latent_dim=latent_dim)).to(device)
    optimizer=torch.optim.AdamW(base.parameters(),lr=lr)
    checkpoint=out/'checkpoint.pt'
    first_epoch=1
    history=[]
    if resume:
        state=torch.load(checkpoint,map_location=device,weights_only=True)
        base.load_state_dict(state['model'],strict=True)
        optimizer.load_state_dict(state['optimizer'])
        first_epoch=int(state['epoch'])+1
        history=state.get('history',[])
        if int(state['latent_dim'])!=latent_dim or int(state['world_size'])!=world:
            raise ValueError('resume configuration changed')
    wrapped=DistributedDataParallel(base,device_ids=[device.index] if device.type=='cuda' else None) if world>1 else base
    dtype=(torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if amp=='auto' and device.type=='cuda' else (torch.bfloat16 if amp=='bf16' else torch.float16)
    autocast_on=device.type=='cuda' and amp!='none'
    scaler=torch.amp.GradScaler('cuda',enabled=autocast_on and dtype==torch.float16)
    if not resume:
        initial=validate(base,valloader,device)
        if rank==0:
            history.append({'epoch':0,'validation':initial})
            print(f'initial_val={initial["total"]:.6f} world={world} device={device}',flush=True)
    try:
        for epoch in range(first_epoch,epochs+1):
            sampler.set_epoch(epoch)
            wrapped.train()
            optimizer.zero_grad(set_to_none=True)
            running=0.;observed=0
            count_batches=len(trainloader)
            for i,batch in enumerate(trainloader):
                x=to_device(batch,device)
                # Normalize by effective microbatches in the current accumulation group.
                group_start=(i//accumulation)*accumulation
                group_size=min(accumulation,count_batches-group_start)
                sync=(i+1)%accumulation==0 or i+1==count_batches
                ctx=wrapped.no_sync() if world>1 and not sync else contextlib.nullcontext()
                with ctx:
                    with torch.autocast(device_type='cuda',dtype=dtype,enabled=autocast_on):
                        decoded,mu,logvar=wrapped(x,sample=True)
                        losses=compute_losses(decoded,x,mu,logvar)
                    if not torch.isfinite(losses['total']).all():
                        raise RuntimeError('non-finite loss')
                    scaler.scale(losses['total']/group_size).backward()
                running+=float(losses['total'].detach())*x['text'].size(0)
                observed+=x['text'].size(0)
                if sync:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(base.parameters(),max_norm=2.)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            loss_sum=rank_sum(running,device)
            count_sum=rank_sum(observed,device)
            validation=validate(base,valloader,device)
            if rank==0:
                stats={'epoch':epoch,'train_objective':loss_sum/count_sum,
                       'validation':validation,'global_samples_seen_with_sampler_padding':int(count_sum)}
                history.append(stats)
                atomic_checkpoint({'model':base.state_dict(),'optimizer':optimizer.state_dict(),
                                   'epoch':epoch,'history':history,'latent_dim':latent_dim,
                                   'world_size':world,'strategy':'ddp'},checkpoint)
                (out/'history.json').write_text(json.dumps(history,indent=2))
                print(f'epoch={epoch} train={stats["train_objective"]:.6f} '
                      f'val={validation["total"]:.6f} accuracy={validation["token_accuracy"]:.4f}',flush=True)
            if dist.is_initialized():
                dist.barrier()
        return history
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
