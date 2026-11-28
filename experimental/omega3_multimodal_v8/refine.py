import json, time, copy, sys, hashlib
from pathlib import Path
import torch,numpy as np
from neural import setup,Field,target,generate
from engine import ingest,export
from independent import decode_file
from hybrid import choose,pack,read8
P=Path(__file__).resolve().parent;O=P/'results';G=P/'generated';G.mkdir(exist_ok=True);src=Path(sys.argv[1]) if len(sys.argv)>1 else P.parent/'omega7'/'outputs';r=json.loads((P/'verification.json').read_text());r['hybrid']={};start=time.perf_counter()
for name,q in r['regression'].items():
 a,m=decode_file(src/(name+'.s3m'));h=choose(a,m,src/(name+'.s3m'),O/('media_'+name+'.s3b'),q['tile_counts'],O/(name+'.s3h'));r['hybrid'][name]=h;print('HYBRID',name,h['format'],h['selected_bytes'],flush=True)
# Fixed independent validation sample; checkpoint selection on aggregate and per-mode tolerance.
setup();checkpoint=torch.load(P/'generator.pt',map_location='cpu',weights_only=True);net=Field();net.load_state_dict(checkpoint['field']);vg=torch.Generator().manual_seed(8801);val=torch.rand(12288,3,generator=vg);vk=torch.arange(len(val))%3;vy=target(val,vk)
def metric():
 with torch.no_grad():
  e=((net(val,vk)-vy)**2).mean(1);return [float(e.mean())]+[float(e[vk==i].mean()) for i in range(3)]
initial=metric();best=initial;state=copy.deepcopy(net.state_dict());opt=torch.optim.Adam(net.parameters(),lr=.0007);curve=[];accepted=0
for i in range(1,1001):
 x=torch.rand(1536,3);k=torch.arange(len(x))%3;loss=((net(x,k)-target(x,k))**2).mean();opt.zero_grad();loss.backward();opt.step()
 if i%100==0:
  e=metric();accept=e[0]<best[0] and all(e[j]<=initial[j]*1.02 for j in range(1,4));curve.append(dict(step=i,mse=e,accepted=accept))
  if accept:best=e;state=copy.deepcopy(net.state_dict());accepted+=1
  else:net.load_state_dict(state);opt=torch.optim.Adam(net.parameters(),lr=.0007)
checkpoint['field']=state;torch.save(checkpoint,P/'generator_refined.pt');r['neural_refinement']={'steps':1000,'initial_mse':initial,'final_mse':best,'labels':['aggregate','visual','audio','geometry'],'accepted_checkpoints':accepted,'curve':curve,'validation_samples':len(val),'selection_guard':'aggregate decrease; each modality <= 102% initial MSE','note':'fixed selection validation set; not independent final generalization test'}
print('REFINE',r['neural_refinement'],flush=True)
r['generation']=generate(P/'generator_refined.pt',G,240,phase=.31);print('GEN',r['generation'],flush=True);r['new_generated_tests']={}
for kind,file in [('image','generated.png'),('video','generated.mp4'),('audio','generated.wav'),('text','generated.txt'),('geometry','generated_geometry.npy')]:
 a,m=ingest(kind,G/file);q=pack(a,m,G/(kind+'.s3h'));b,md=read8(G/(kind+'.s3h'));assert np.array_equal(a,b);out=G/('recovered_'+file);export(b,md,out);re,_=ingest(kind,out);assert np.array_equal(re,a);r['new_generated_tests'][kind]=dict(q,raw_bytes=a.size,roundtrip_errors=0,export_reingest_errors=0);print('NEW',kind,q,flush=True)
r['refinement_seconds']=time.perf_counter()-start;(P/'verification.json').write_text(json.dumps(r,indent=2));print('ALL PASS',flush=True)
