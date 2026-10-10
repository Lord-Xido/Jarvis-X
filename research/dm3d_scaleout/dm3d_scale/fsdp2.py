"""Multi-process FSDP2 parameter sharding + torch.distributed.checkpoint.

This separate path has been exercised on two CPU workers; CUDA/NCCL scale is supported
by the PyTorch backend but requires real GPU-cluster qualification.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
import torch
from torch import distributed as dist
from torch.distributed.fsdp import fully_shard
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (get_state_dict, set_state_dict,
    get_model_state_dict, StateDictOptions)
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from dm3d_multimodal.model import DM3DMultiVAE,ModelConfig,compute_losses
from .shards import MMapSceneDataset
from .distributed import environment,to_device,rank_sum,KEYS

class PaddedValidation(Dataset):
    """Equal forward counts per rank with an explicit validity bit for duplicate padding."""
    def __init__(self,base,rank,world):
        self.base=base
        self.indices=list(range(rank,len(base),world))
        self.expected=math.ceil(len(base)/world)
        if len(base)==0:raise ValueError('empty validation dataset')
    def __len__(self):return self.expected
    def __getitem__(self,i):
        valid=i<len(self.indices)
        row=self.base[self.indices[i] if valid else 0]
        row['valid']=torch.tensor(valid)
        return row

def _eval_fsdp(model,loader,device):
    model.eval()
    totals=torch.zeros(3,device=device,dtype=torch.float64)
    with torch.no_grad():
        for batch in loader:
            x=to_device(batch,device)
            # All ranks run the same number of FSDP2 forwards, including padded batches.
            reconstruction,mu,lv=model(x,sample=False)
            valid=batch['valid'].to(device).bool()
            n=int(valid.sum())
            if n==0:continue
            rx={k:v[valid] for k,v in reconstruction.items()}
            xx={k:v[valid] for k,v in x.items()}
            losses=compute_losses(rx,xx,mu[valid],lv[valid])
            accuracy=(rx['text_logits'].argmax(-1)==xx['text']).float().mean()
            totals+=torch.tensor([n,n*float(losses['total']),n*float(accuracy)],device=device,dtype=torch.float64)
    dist.all_reduce(totals,op=dist.ReduceOp.SUM)
    return {'total':float(totals[1]/totals[0]),'token_accuracy':float(totals[2]/totals[0])}

def _distributed_state(model,optimizer):
    md,od=get_state_dict(model,optimizer)
    return {'model':md,'optimizer':od}

def train_fsdp2(out,epochs=1,batch_size=2,latent_dim=512,lr=.001,resume=False,seed=7,export_full=False):
    if not 1<=epochs<=10000 or batch_size<1 or not 0<lr<.1:
        raise ValueError('invalid hyperparameters')
    world,rank,device=environment()
    if world<2:
        raise ValueError('FSDP2 strategy expects torchrun with >=2 processes')
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    trainset=MMapSceneDataset(out/'shards'/'train')
    validset=MMapSceneDataset(out/'shards'/'validation')
    sampler=DistributedSampler(trainset,num_replicas=world,rank=rank,seed=seed,shuffle=True)
    tr=DataLoader(trainset,batch_size=batch_size,sampler=sampler,num_workers=0)
    vl=DataLoader(PaddedValidation(validset,rank,world),batch_size=batch_size,shuffle=False)
    model=DM3DMultiVAE(ModelConfig(latent_dim=latent_dim)).to(device)
    # Shard each large block before the root to overlap all-gathers with compute.
    for layer in (model.text_transformer,model.image_stem,model.audio_stem,
                  model.video_stem,model.fuse,model.decoder3d,
                  model.text_head,model.image_head,model.audio_head,model.video_head):
        fully_shard(layer)
    fully_shard(model)
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr)
    history=[];start_epoch=1
    metadata=out/'fsdp2_status.json'
    if resume:
        saved=json.loads(metadata.read_text())
        start_epoch=int(saved['epoch'])+1
        history=saved['history']
        if saved['latent_dim']!=latent_dim:raise ValueError('latent dim differs from saved run')
        payload=_distributed_state(model,optimizer)
        dcp.load(state_dict=payload,checkpoint_id=str(out/saved.get('checkpoint_id','fsdp2_checkpoint')))
        set_state_dict(model,optimizer,model_state_dict=payload['model'],optim_state_dict=payload['optimizer'])
    try:
        for epoch in range(start_epoch,epochs+1):
            sampler.set_epoch(epoch)
            model.train();total=0.;seen=0
            for batch in tr:
                x=to_device(batch,device)
                optimizer.zero_grad(set_to_none=True)
                reconstruction,mu,lv=model(x,sample=True)
                loss=compute_losses(reconstruction,x,mu,lv)['total']
                if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
                loss.backward();optimizer.step()
                n=x['text'].shape[0];total+=float(loss.detach())*n;seen+=n
            train_mean=rank_sum(total,device)/rank_sum(seen,device)
            validation=_eval_fsdp(model,vl,device)
            entry={'epoch':epoch,'train_objective':train_mean,'validation':validation}
            # All ranks participate in distributed saving; checkpoints support resharding.
            payload=_distributed_state(model,optimizer)
            checkpoint_name=f'fsdp2_checkpoint_{epoch:06d}'
            dcp.save(state_dict=payload,checkpoint_id=str(out/checkpoint_name))
            dist.barrier()
            if rank==0:
                history.append(entry)
                status={'epoch':epoch,'latent_dim':latent_dim,'history':history,
                        'strategy':'fsdp2','checkpoint_id':checkpoint_name}
                temporary=out/'fsdp2_status.tmp'
                temporary.write_text(json.dumps(status,indent=2))
                temporary.replace(metadata)
                print(f'FSDP2 epoch={epoch} val={validation["total"]:.6f} '
                      f'token_accuracy={validation["token_accuracy"]:.4f}',flush=True)
            dist.barrier()
        if export_full:
            # Optional full-model export requires enough host RAM for model parameters.
            # Every rank enters the collective; only rank 0 receives CPU tensors.
            full=get_model_state_dict(model,options=StateDictOptions(
                full_state_dict=True,cpu_offload=True))
            if rank==0:
                torch.save({'model':full,'latent_dim':latent_dim,'strategy':'fsdp2-export'},out/'checkpoint.pt')
                print('FSDP2 full checkpoint exported for indexing',flush=True)
            dist.barrier()
    finally:
        if dist.is_initialized():dist.destroy_process_group()
    return history
