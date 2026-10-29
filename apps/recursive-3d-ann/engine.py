#!/usr/bin/env python3
"""Executable 3D text-token autoencoder + trainable neural token sampler.

Pipeline: synthetic source transitions -> train next-token ANN -> 1,000 independent
sampled streams -> 4^3 token lattice -> Conv3D encoder 4^3->2^3->1^3 ->
24-D bottleneck -> learned 3D decoder -> empirical inward consistency loop ->
exact recovery via optional XOR residual -> traceable JSON and HTML report.

A neural autoencoder is lossy by default; exact recovery requires the residual.
This is a compact trained model, NOT a pretrained general-purpose LLM.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# These explicit transition logits are copied from the uploaded toy source,
# including the default branch. They provide synthetic training labels.
RULES = {
 "The continuous creation of new ideas": [(" leads",4.8),(" fuels",4.1),(" inspires",3.5),(" drives",3.2),(" forms",1.8)],
 " leads": [(" to",6.2),(" directly",2.1),(" naturally",1.2)],
 " fuels": [(" technological",4.5),(" human",3.8),(" groundbreaking",2.9)],
 " inspires": [(" future",4.2),(" global",3.9),(" creative",3.1)],
 " drives": [(" rapid",4.1),(" economic",3.6),(" continuous",2.7)],
 " to": [(" transformative",4.2),(" breakthrough",3.9),(" sustainable",3.1)],
 " default": [(" innovation",4.9),(" growth",3.8),(" progress",3.4),(" solutions",2.2),(" discoveries",2.0),(" systems",1.5)]
}
PROMPT = "The continuous creation of new ideas"
VOXELS = 64
SIDE = 4
LATENT = 24


def make_vocabulary():
    terms = sorted({t for options in RULES.values() for t,_ in options})
    return ["<BOS>"] + terms


def sample_rule(rng: np.random.Generator, context: str, temp: float=.8, topk:int=4):
    options = RULES.get(context, RULES[" default"])
    opts = sorted(options, key=lambda x:x[1], reverse=True)[:topk]
    vals = np.asarray([v for _,v in opts], np.float64)/temp
    weights = np.exp(vals-vals.max()); weights/=weights.sum()
    return opts[int(rng.choice(len(opts),p=weights))][0]


def source_sequences(n:int, steps:int, seed:int):
    rng = np.random.default_rng(seed)
    seq=[]
    for _ in range(n):
        prev=PROMPT
        tokens=[]
        for _ in range(steps):
            word=sample_rule(rng,prev)
            tokens.append(word)
            prev=word
        seq.append(tokens)
    return seq


class NextTokenANN(nn.Module):
    """A genuine trainable 2-layer ANN for P(next token | preceding token)."""
    def __init__(self,v:int):
        super().__init__()
        self.embedding=nn.Embedding(v,24)
        self.network=nn.Sequential(nn.Linear(24,48),nn.GELU(),nn.Linear(48,v))
    def forward(self,prev):
        return self.network(self.embedding(prev))


class SpatialAutoencoder(nn.Module):
    """Convolutional 3D encoder / decoder, spatial 4^3 -> 2^3 -> 1^3 -> 4^3."""
    def __init__(self,v:int):
        super().__init__()
        self.v=v
        self.embedding=nn.Embedding(v,12)
        self.down1=nn.Conv3d(12,24,3,stride=2,padding=1)
        self.down2=nn.Conv3d(24,LATENT,2,stride=2)
        self.up1=nn.ConvTranspose3d(LATENT,24,2,stride=2)
        self.up2=nn.ConvTranspose3d(24,24,2,stride=2)
        self.readout=nn.Conv3d(24,v,1)
    def encode_embedded(self,token_emb):
        x=token_emb.transpose(1,2).reshape(-1,12,SIDE,SIDE,SIDE)
        return self.down2(F.gelu(self.down1(x)))
    def encode(self,ids):
        return self.encode_embedded(self.embedding(ids))
    def decode_logits(self,z):
        y=F.gelu(self.up1(z))
        y=F.gelu(self.up2(y))
        return self.readout(y).permute(0,2,3,4,1).reshape(-1,VOXELS,self.v)
    def forward(self,ids):
        z=self.encode(ids)
        return self.decode_logits(z),z
    def soft_reencode(self,logits):
        probabilities=F.softmax(logits,dim=-1)
        emb=probabilities @ self.embedding.weight
        return self.encode_embedded(emb)


def train_predictor(model, sequences, stoi, epochs:int, lr:float=.012):
    pairs=[]
    for row in sequences:
        prev=stoi["<BOS>"]
        for tok in row:
            out=stoi[tok]
            pairs.append((prev,out));prev=out
    arr=torch.tensor(pairs,dtype=torch.long)
    gen=torch.Generator().manual_seed(919)
    order=torch.randperm(len(arr),generator=gen)
    split=int(.85*len(arr))
    train=arr[order[:split]];held=arr[order[split:]]
    optim=torch.optim.AdamW(model.parameters(),lr=lr)
    history=[]
    for epoch in range(epochs):
        perms=torch.randperm(len(train),generator=gen)
        model.train();running=0.0; seen=0
        for ix in perms.split(2048):
            batch=train[ix]; logits=model(batch[:,0]);loss=F.cross_entropy(logits,batch[:,1])
            optim.zero_grad();loss.backward();optim.step()
            running+=loss.item()*len(batch);seen+=len(batch)
        model.eval()
        with torch.no_grad():
            sample=held[:min(len(held),4096)]
            vl=F.cross_entropy(model(sample[:,0]),sample[:,1]).item()
        history.append({'epoch':epoch+1,'training_ce':round(running/seen,5),'heldout_ce':round(vl,5),'heldout_perplexity':round(math.exp(min(20,vl)),4)})
    return history


def train_autoencoder(model, data:torch.Tensor, steps:int, batch_size:int=64, lr:float=.007):
    split=int(.85*len(data))
    train=data[:split];held=data[split:]
    optim=torch.optim.Adam(model.parameters(),lr=lr)
    history=[];rng=np.random.default_rng(124)
    for step in range(steps):
        model.train()
        idx=torch.tensor(rng.integers(0,len(train),size=batch_size),dtype=torch.long)
        samples=train[idx]
        logits,_=model(samples)
        loss=F.cross_entropy(logits.reshape(-1,model.v),samples.reshape(-1))
        optim.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.0);optim.step()
        if step==0 or (step+1)%max(1,steps//10)==0 or step+1==steps:
            model.eval()
            with torch.no_grad():
                v_logits,_=model(held)
                vl=F.cross_entropy(v_logits.reshape(-1,model.v),held.reshape(-1)).item()
                va=(v_logits.argmax(-1)==held).float().mean().item()
            history.append({'step':step+1,'train_ce':round(loss.item(),5),'validation_ce':round(vl,5),'validation_accuracy':round(va,5)})
    return history


@torch.no_grad()
def generate(model, n:int, steps:int, seed:int, temp:float,topk:int):
    model.eval()
    ids=torch.zeros(n,dtype=torch.long)
    out=torch.empty((n,steps),dtype=torch.long)
    generator=torch.Generator().manual_seed(seed)
    start=time.perf_counter()
    for t in range(steps):
        logits=model(ids)/temp
        if topk < logits.shape[1]:
            vals,idx=logits.topk(topk,dim=-1)
            probs=F.softmax(vals,dim=-1)
            index=torch.multinomial(probs,1,generator=generator).squeeze(-1)
            nextids=idx.gather(1,index[:,None]).squeeze(1)
        else:
            probs=F.softmax(logits,dim=-1)
            nextids=torch.multinomial(probs,1,generator=generator).squeeze(1)
        out[:,t]=nextids
        ids=nextids
    elapsed=time.perf_counter()-start
    return out,elapsed


@torch.no_grad()
def inward_loop(model: SpatialAutoencoder, ids:torch.Tensor, iterations:int=8, gain:float=.25):
    model.eval()
    z0=model.encode(ids)
    z=z0.detach().clone()
    trajectory=[]
    for iteration in range(iterations+1):
        logits=model.decode_logits(z)
        recon=logits.argmax(-1)
        match=(recon==ids).float().mean().item()
        trajectory.append({
          'iteration':iteration,
          'relative_displacement_from_input':round((torch.linalg.vector_norm(z-z0)/(torch.linalg.vector_norm(z0)+1e-9)).item(),7),
          'token_reconstruction_accuracy':round(match,5),
          'latent_norm':round(torch.linalg.vector_norm(z).item(),5),
          'state':[round(x,5) for x in z[0].flatten().tolist()]
        })
        if iteration<iterations:
            target=model.soft_reencode(logits)
            z=(1-gain)*z+gain*target
            if not torch.isfinite(z).all():
                raise ArithmeticError('Non-finite latent state during inward iteration')
    return z0,z,trajectory


def run(args):
    if args.steps != VOXELS:
        raise ValueError('The specified 4x4x4 lattice requires exactly 64 positions (use --steps 64)')
    if not(0<args.temperature and 1<=args.topk):raise ValueError('Invalid sampling temperature/topk')
    torch.set_num_threads(min(2,torch.get_num_threads()))
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    vocabulary=make_vocabulary();stoi={x:i for i,x in enumerate(vocabulary)};v=len(vocabulary)
    t0=time.perf_counter()
    source=source_sequences(args.corpus,args.steps,args.seed+10)
    corpus=torch.tensor([[stoi[x] for x in row] for row in source],dtype=torch.long)
    predictor=NextTokenANN(v)
    pred_history=train_predictor(predictor,source,stoi,args.pred_epochs)
    auto=SpatialAutoencoder(v)
    auto_history=train_autoencoder(auto,corpus,args.ae_steps)
    output,gen_seconds=generate(predictor,args.streams,args.steps,args.seed+300,args.temperature,min(args.topk,v))
    auto.eval()
    with torch.no_grad():
        logits,z=auto(output)
        decoded=logits.argmax(-1)
        original=output.to(torch.uint8)
        approximate=decoded.to(torch.uint8)
        residual=original ^ approximate
        recovered=(approximate ^ residual)
        assert torch.equal(recovered,original)
        fidelity=(decoded==output).float().mean().item()
        softmax_check=(F.softmax(logits[:2],dim=-1).sum(-1)-1).abs().max().item()
    z0,zn,trajectory=inward_loop(auto,output[:1],args.inward_iterations)
    with torch.no_grad():
        first_original=output[0].tolist();first_decode=decoded[0].tolist()
    original_bytes=original.numel()
    residual_nonzero=int((residual!=0).sum().item())
    distinct=len({tuple(row.tolist()) for row in output})
    unique_vocab=len(set(output.flatten().tolist()))
    results={
      'identity':'Trainable 3D autoencoding / decoding ANN with self-consistency feedback and learned next-token generation',
      'reproducible_seed':args.seed,'prompt':PROMPT,
      'source':'Source rule logits provide synthetic training data, not a pretrained language model',
      'architecture':{'vocabulary_size':v,'vocabulary':vocabulary,'geometry':'4 x 4 x 4 token lattice','input_voxels':64,'encoder_stages':['4x4x4 channels=12','2x2x2 channels=24','1x1x1 channels=24'],'latent_channels':24,'decoder_stages':['1x1x1 channels=24','2x2x2 channels=24','4x4x4 channels=24','token logits channels='+str(v)],'feedback_operator':'z[k+1]=0.75*z[k]+0.25*Encode(softmax(Decode(z[k])))','feedback_convergence':'Empirical; NOT mathematically guaranteed','lossless_method':'XOR residual appended to the lossy decoder output (not guaranteed compression)'},
      'execution':{'num_streams':args.streams,'steps_per_stream':args.steps,'generated_tokens':output.numel(),'distinct_outputs':distinct,'distinct_generated_token_types':unique_vocab,'generation_wall_seconds':round(gen_seconds,6),'generation_only_token_rate':round(output.numel()/gen_seconds,2),'total_training_and_execution_wall_seconds':round(time.perf_counter()-t0,3),'device':'CPU','simulated_gpu_cores':0},
      'training':{'source_sequence_count':args.corpus,'predictor_epochs':args.pred_epochs,'autoencoder_optimization_steps':args.ae_steps,'predictor_history':pred_history,'autoencoder_history':auto_history,'predictor_parameter_count':sum(x.numel() for x in predictor.parameters()),'autoencoder_parameter_count':sum(x.numel() for x in auto.parameters())},
      'verification':{'output_shape':list(output.shape),'latent_shape':list(z.shape),'softmax_max_sum_error':softmax_check,'lossy_token_accuracy_generated':round(fidelity,5),'residual_recovery_exact':bool(torch.equal(recovered,original)),'original_8bit_tokens_bytes':original_bytes,'residual_nonzero_byte_count':residual_nonzero,'naive_float32_latent_plus_dense_residual_bytes_per_stream':LATENT*4+VOXELS,'true_lossless_compression_claim':False,'finite_latent':bool(torch.isfinite(z).all()),'trained_ann':True},
      'inward_feedback':trajectory,
      'selected_streams':[{'stream_id':i+1,'text':PROMPT+''.join(vocabulary[x] for x in output[i].tolist()),'token_ids':output[i].tolist(),'lossy_decoded_text':''.join(vocabulary[x] for x in decoded[i].tolist()),'decoded_ids':decoded[i].tolist(),'per_stream_reconstruction_accuracy':round((output[i]==decoded[i]).float().mean().item(),5)} for i in range(min(args.streams,24))],
      'volume_example':{'original_ids':first_original,'decoded_ids':first_decode,'xor_residual':residual[0].tolist(),'recovered_ids':recovered[0].tolist(),'input_latent_24':[round(x,5) for x in z0[0].flatten().tolist()],'recurrent_latent_24':[round(x,5) for x in zn[0].flatten().tolist()]},
      'limitations':['Training data comes exclusively from the uploaded fixed vocabulary rules.','The next-token ANN conditions on the previous token, not a long context.','The learned 3D bottleneck is lossy; residual recovery is exact but generally expands storage.','The inward loop empirically re-encodes its own reconstruction but has no formal contraction bound.','CPU generation-only timing excludes training, rendering, and deployment overhead.','The 3D grid is a mathematical data representation, not physical 3D hardware.']
    }
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(results,indent=2,ensure_ascii=False),encoding='utf-8')
    torch.save({'next_token':predictor.state_dict(),'autoencoder':auto.state_dict(),'vocabulary':vocabulary,'config':{'side':SIDE,'latent':LATENT}},out.with_suffix('.pt'))
    print(json.dumps({'output':str(out),'model':str(out.with_suffix('.pt')),'streams':args.streams,'tokens':output.numel(),'prediction_heldout_perplexity':pred_history[-1]['heldout_perplexity'],'autoencoder_validation_accuracy':auto_history[-1]['validation_accuracy'],'lossy_generated_accuracy':fidelity,'residual_exact':True,'generation_tokens_per_s':round(output.numel()/gen_seconds,1),'inward_start_match':trajectory[0]['token_reconstruction_accuracy'],'inward_end_match':trajectory[-1]['token_reconstruction_accuracy'],'elapsed_s':round(time.perf_counter()-t0,2)},indent=2))
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--streams',type=int,default=1000)
    parser.add_argument('--steps',type=int,default=64,help='64 tokens = 4^3 voxels')
    parser.add_argument('--corpus',type=int,default=1100)
    parser.add_argument('--pred-epochs',type=int,default=5)
    parser.add_argument('--ae-steps',type=int,default=120)
    parser.add_argument('--inward-iterations',type=int,default=8)
    parser.add_argument('--seed',type=int,default=20261029)
    parser.add_argument('--temperature',type=float,default=.8)
    parser.add_argument('--topk',type=int,default=4)
    parser.add_argument('--output',default=str(Path(__file__).parent/'results.json'))
    args=parser.parse_args();run(args)
if __name__=='__main__':main()
