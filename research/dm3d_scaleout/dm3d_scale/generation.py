"""End-to-end indexed reconstruction, retrieval conditioning, anchored refinement and media export."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from dm3d_multimodal.data import decode_text
from dm3d_multimodal.media import save_media
from dm3d_multimodal.model import DM3DMultiVAE,ModelConfig
from .shards import MMapSceneDataset
from .index import SegmentIndex


def generate(out,sample_id,split='validation',alpha=.3,tolerance=.002,max_steps=12,topk=3):
    if split not in ('train','validation') or not 1<=topk<=50:
        raise ValueError('invalid generation request')
    out=Path(out)
    model_state=torch.load(out/'checkpoint.pt',map_location='cpu',weights_only=True)
    model=DM3DMultiVAE(ModelConfig(latent_dim=int(model_state['latent_dim'])))
    model.load_state_dict(model_state['model'],strict=True)
    model.eval()
    dataset=MMapSceneDataset(out/'shards'/split)
    sample=None
    for i in range(len(dataset)):
        candidate=dataset[i]
        if int(candidate['sample_id'])==int(sample_id):
            sample=candidate;break
    if sample is None:
        raise ValueError(f'sample_id {sample_id} not in {split} data')
    x={k:sample[k].unsqueeze(0) for k in ('text','image','audio','video')}
    with torch.no_grad():
        reconstructed,z,diagnostics=model.refine(x,alpha=alpha,tolerance=tolerance,max_steps=max_steps)
        anchor,_,_=model.encode(x)
        idx=SegmentIndex(out/'index',dimension=int(model_state['latent_dim']))
        hits=idx.topk(anchor[0].numpy(),k=topk,
                      exclude=sample_id if split=='train' else None)
        if hits:
            memories=np.stack([idx.embedding(hit['id']) for hit in hits])
            prior=torch.as_tensor(memories.mean(axis=0),dtype=torch.float32)[None]
            conditioned=model.decode(.65*z+.35*prior)
        else:
            conditioned=model.decode(z)
        generator=torch.Generator().manual_seed(129)
        unconditional=model.decode(torch.randn((1,z.shape[1]),generator=generator))
    destination=out/'generation'/f'{split}_{sample_id}'
    destination.mkdir(parents=True,exist_ok=True)
    files_recon=save_media(destination/'reconstruction',reconstructed)
    files_conditioned=save_media(destination/'retrieval_conditioned',conditioned)
    files_unconditional=save_media(destination/'unconditional_prior',unconditional)
    result={'sample_id':sample_id,'split':split,'reconstructed_text':decode_text(reconstructed['text_logits'].argmax(-1)[0]),
            'retrieval_text':decode_text(conditioned['text_logits'].argmax(-1)[0]),
            'neighbors':hits,'inward_diagnostics':diagnostics,'reconstruction_files':files_recon,
            'retrieval_files':files_conditioned,
            'unconditional_text':decode_text(unconditional['text_logits'].argmax(-1)[0]),
            'unconditional_files':files_unconditional,
            'warning':'Small-vocabulary reconstruction, not a general media foundation model'}
    (destination/'result.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    return result
