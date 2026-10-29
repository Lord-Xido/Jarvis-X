#!/usr/bin/env python3
"""Regenerate the standalone browser application from a trained checkpoint and report."""
import json
from pathlib import Path
import torch
root=Path(__file__).resolve().parent
checkpoint=torch.load(root/'results.pt',map_location='cpu',weights_only=True)
weights={'vocabulary':checkpoint['vocabulary'],'weights':{name+'.'+key:value.detach().cpu().reshape(-1).tolist() for name,sd in [('next_token',checkpoint['next_token']),('autoencoder',checkpoint['autoencoder'])] for key,value in sd.items()}}
(root/'weights.json').write_text(json.dumps(weights,separators=(',',':')))
s=(root/'dashboard_template.html').read_text()
s=s.replace('__DATA__',(root/'results.json').read_text().replace('</','<\\/')).replace('__WEIGHTS__',(root/'weights.json').read_text().replace('</','<\\/'))
p=root/'3D_ANN_Live_Engine.html';p.write_text(s)
print(f'Created {p} ({p.stat().st_size:,} bytes)')
