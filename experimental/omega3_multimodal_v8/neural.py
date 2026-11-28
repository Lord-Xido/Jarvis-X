import math,json,wave,subprocess,argparse
from pathlib import Path
import numpy as np
import torch
from torch import nn
from PIL import Image
from engine import features

def setup():
 torch.manual_seed(710);torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
def train_predictor(vols,out):
 setup();rng=np.random.default_rng(710);xs=[];ys=[]
 for a in vols:
  for i in np.linspace(1,len(a)-1,min(24,len(a)-1),dtype=int):
   x=features(a[i-1]).reshape(-1,5);y=a[i].reshape(-1);j=rng.choice(len(y),min(512,len(y)),replace=False);xs.append(x[j]);ys.append(y[j])
 X=torch.tensor(np.concatenate(xs),dtype=torch.float32)/255;Y=torch.tensor(np.concatenate(ys),dtype=torch.float32)[:,None]/255
 net=nn.Sequential(nn.Linear(5,8),nn.ReLU(),nn.Linear(8,1));opt=torch.optim.Adam(net.parameters(),lr=.01);loss=nn.MSELoss();start=float(loss(net(X),Y).detach())
 for _ in range(220):
  j=torch.randint(len(X),(1024,));l=loss(net(X[j]),Y[j]);opt.zero_grad();l.backward();opt.step()
 end=float(loss(net(X),Y).detach());q=lambda t,s:torch.round(t.detach()*s).to(torch.int64).tolist();m={'w1':q(net[0].weight.T,256),'b1':q(net[0].bias,255*256),'w2':q(net[2].weight.flatten(),256),'b2':q(net[2].bias[0],255*256)};Path(out).write_text(json.dumps(m));return dict(parameters=57,training_samples=len(X),steps=220,initial_mse=start,final_mse=end)
FREQ=torch.tensor([1.,2.,4.,8.,16.,32.,64.,110.,220.,330.])
class Field(nn.Module):
 def __init__(self):super().__init__();self.net=nn.Sequential(nn.Linear(66,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU(),nn.Linear(64,3),nn.Sigmoid())
 def forward(self,x,k):
  a=x[:,:,None]*FREQ[None,None,:]*2*math.pi;gate=((k[:,None,None]==1)|(FREQ[None,None,:]<=8)).float();return self.net(torch.cat([x,(torch.sin(a)*gate).flatten(1),(torch.cos(a)*gate).flatten(1),torch.nn.functional.one_hot(k,3).float()],1))
def target(x,k):
 u,v,t=x.T;out=torch.zeros((len(x),3));cx=.5+.26*torch.cos(2*math.pi*t);cy=.5+.22*torch.sin(2*math.pi*t);r=((u-cx)**2+(v-cy)**2).sqrt();r2=((u-(1-cx))**2+(v-(1-cy))**2).sqrt();g=torch.exp(-r*r*95);h=torch.exp(-r2*r2*130);ring=torch.exp(-((r-.12)*45)**2);scene=torch.stack([.04+.12*v+.72*g+.18*ring,.07+.14*u+.72*h+.25*ring,.16+.14*v+.40*g+.64*h],1).clamp(0,1);out[k==0]=scene[k==0];sig=.5+.26*torch.sin(2*math.pi*220*u)+.12*torch.sin(2*math.pi*330*u);out[k==1]=sig[k==1,None].expand(-1,3);dist=((u-.5)**2+(v-.5)**2+(t-.5)**2).sqrt();occ=torch.sigmoid((.32-dist)*60);out[k==2]=occ[k==2,None].expand(-1,3);return out
CORPUS=['the geometric engine encodes the frame and verifies the residual','the neural engine predicts the next frame and reconstructs the scene','the scanner reads the pixel and maps the temporal state','the machine retains verified memory and selects the efficient kernel','the engine encodes audio and reconstructs the signal','the engine generates geometry and projects the volume','the machine verifies the frame and commits the state','the neural field generates a moving scene and preserves temporal continuity']
class TextNet(nn.Module):
 def __init__(self,n):super().__init__();self.emb=nn.Embedding(n,16);self.net=nn.Sequential(nn.Linear(48,32),nn.Tanh(),nn.Linear(32,n))
 def forward(self,x):return self.net(self.emb(x).flatten(1))
def train_generator(out):
 setup();model=Field();opt=torch.optim.Adam(model.parameters(),lr=.003);val=torch.rand(4096,3);vk=torch.arange(len(val))%3;vy=target(val,vk);init=float(((model(val,vk)-vy)**2).mean().detach())
 for _ in range(1000):
  x=torch.rand(1536,3);k=torch.arange(len(x))%3;l=((model(x,k)-target(x,k))**2).mean();opt.zero_grad();l.backward();opt.step()
 with torch.no_grad():err=((model(val,vk)-vy)**2).mean(1);final=float(err.mean());per={name:float(err[vk==i].mean()) for i,name in enumerate(['video','audio','geometry'])}
 tok=sorted(set(' '.join(CORPUS).split()))+['<start>','<end>'];lookup={t:i for i,t in enumerate(tok)};rows=[];labels=[]
 for line in CORPUS:
  a=[lookup['<start>']]*3+[lookup[t] for t in line.split()]+[lookup['<end>']]
  for i in range(3,len(a)):rows.append(a[i-3:i]);labels.append(a[i])
 X=torch.tensor(rows);Y=torch.tensor(labels);txt=TextNet(len(tok));o=torch.optim.Adam(txt.parameters(),lr=.02);ce=nn.CrossEntropyLoss();li=float(ce(txt(X),Y).detach())
 for _ in range(250):
  l=ce(txt(X),Y);o.zero_grad();l.backward();o.step()
 lf=float(ce(txt(X),Y).detach());torch.save({'field':model.state_dict(),'text':txt.state_dict(),'vocab':tok},out)
 return dict(field_parameters=sum(p.numel() for p in model.parameters()),field_steps=1000,initial_validation_mse=init,final_validation_mse=final,per_modality_validation_mse=per,text_parameters=sum(p.numel() for p in txt.parameters()),text_training_examples=len(X),text_steps=250,text_initial_cross_entropy=li,text_final_cross_entropy=lf)
@torch.no_grad()
def generate(modelpath,outdir,frames=240,phase=.12):
 setup();o=Path(outdir);o.mkdir(parents=True,exist_ok=True);d=torch.load(modelpath,map_location='cpu',weights_only=True);m=Field();m.load_state_dict(d['field']);m.eval();w,h=256,144;yy,xx=torch.meshgrid(torch.linspace(0,1,h),torch.linspace(0,1,w),indexing='ij');xy=torch.stack((xx.flatten(),yy.flatten()),1);k=torch.zeros(len(xy),dtype=torch.long);arr=[]
 for i in range(frames):
  t=torch.full((len(xy),1),(i/frames+phase)%1);v=m(torch.cat((xy,t),1),k);arr.append((v.reshape(h,w,3).numpy()*255+.5).astype(np.uint8))
 a=np.stack(arr);Image.fromarray(a[0]).save(o/'generated.png');subprocess.run(['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{w}x{h}','-r','60','-i','pipe:0','-c:v','libx264rgb','-crf','0','-preset','ultrafast','-threads','2',str(o/'generated.mp4')],input=a.tobytes(),check=True)
 n=16000;x=torch.arange(n).float()/n;coords=torch.stack((x,torch.zeros(n),torch.zeros(n)),1);sig=((m(coords,torch.ones(n,dtype=torch.long))[:,0].numpy()-.5)*2*32767).clip(-32768,32767).astype('<i2');sig=np.tile(sig,math.ceil(frames/60))[:round(frames/60*n)]
 with wave.open(str(o/'generated.wav'),'wb') as f:f.setparams((1,2,n,len(sig),'NONE','not compressed'));f.writeframes(sig.tobytes())
 grid=torch.linspace(0,1,40);zz,yy,xx=torch.meshgrid(grid,grid,grid,indexing='ij');coords=torch.stack((xx.flatten(),yy.flatten(),zz.flatten()),1);vals=m(coords,torch.full((len(coords),),2,dtype=torch.long))[:,0].reshape(40,40,40).numpy();geom=(vals>=.5).astype(np.uint8)*255;np.save(o/'generated_geometry.npy',geom);z,y,x=np.nonzero(geom);(o/'generated_geometry.ply').write_text('ply\nformat ascii 1.0\nelement vertex '+str(len(x))+'\nproperty float x\nproperty float y\nproperty float z\nend_header\n'+'\n'.join(f'{a/39:.5f} {b/39:.5f} {c/39:.5f}' for a,b,c in zip(x,y,z))+'\n')
 txt=TextNet(len(d['vocab']));txt.load_state_dict(d['text']);tok=d['vocab'];ids=[tok.index('<start>')]*3;words=[]
 for _ in range(25):
  logits=txt(torch.tensor([ids[-3:]]))[0];logits[tok.index('<start>')]=-1e9;j=int(torch.multinomial(torch.softmax(logits/.8,0),1));ids.append(j)
  if tok[j]=='<end>':break
  words.append(tok[j])
 (o/'generated.txt').write_text(' '.join(words).capitalize()+'.\n');subprocess.run(['ffmpeg','-y','-v','error','-i',str(o/'generated.mp4'),'-i',str(o/'generated.wav'),'-c:v','copy','-c:a','aac','-shortest',str(o/'generated_av.mp4')],check=True)
 return dict(video_frames=frames,video_dimensions=[w,h],audio_samples=len(sig),geometry_shape=list(geom.shape),occupied_voxels=int(np.count_nonzero(geom)),generated_text=' '.join(words),video_unique_frames=len({__import__('hashlib').sha256(f).digest() for f in a}),audio_peak=int(np.max(abs(sig.astype(np.int32)))))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('model');p.add_argument('out');p.add_argument('--train',action='store_true');p.add_argument('--frames',type=int,default=240);a=p.parse_args();
 if a.train:print(json.dumps(train_generator(a.model),indent=2))
 print(json.dumps(generate(a.model,a.out,a.frames),indent=2))
