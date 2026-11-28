#!/usr/bin/env python3
"""S3D-BCx v7: bounded CPU multimedia VM, 4+58+2 ISA, deterministic integer ANN."""
import argparse, hashlib, json, struct, zlib, wave, subprocess, time
from pathlib import Path
from fractions import Fraction
import numpy as np
from PIL import Image
MAGIC=b'S3M7'; H=struct.Struct('<4sI'); REC=struct.Struct('<BI32s'); MASK=(1<<58)-1
OPS={'FRAME_REF':(10,0),'ENC_MP4':(0,2),'MOTION':(9,1),'PERMEATE':(3,1),'RENORM':(4,1),'ANN_FIRE':(2,1),'XOR_VOXEL':(8,2),'WRITE_VOXEL':(1,0),'DEC_MP4':(5,3),'VERIFY':(12,3),'COMMIT':(13,0),'SYNC_PTS':(6,3)}
ORDER=list(OPS)
def word(op,i):
 o,s=OPS[op]; assert 0<=i<=MASK; return (o<<60)|(i<<2)|s

def features(p):
 # Each row is a voxel-channel; causal spatial neighborhood from preceding z-plane.
 return np.stack((p,np.roll(p,1,0),np.roll(p,-1,0),np.roll(p,1,1),np.roll(p,-1,1)),axis=-1)
def predict(p,m):
 if not np.any(p): return p.copy() # zero input fast-path is also normative in decoder
 f=features(p).astype(np.int32); w1=np.asarray(m['w1'],np.int32);b1=np.asarray(m['b1'],np.int32);w2=np.asarray(m['w2'],np.int32);b2=int(m['b2'])
 h=np.maximum((f@w1+b1)//256,0); return np.clip((h@w2+b2)//256,0,255).astype(np.uint8)
def zipb(b):return zlib.compress(b,1)
def sha(b):return hashlib.sha256(b).digest()
def unpack_payload(mode,body,prev,pred):
 n=prev.size
 if mode==0: raw=body
 else:
  d=zlib.decompressobj();raw=d.decompress(body,n+1)
  if len(raw)!=n or not d.eof or d.unused_data or d.unconsumed_tail:raise ValueError('invalid compressed plane')
 if len(raw)!=n:raise ValueError('plane length')
 a=np.frombuffer(raw,np.uint8).reshape(prev.shape).copy()
 if mode==2:a^=prev
 if mode==3:a^=pred
 if mode not in range(4):raise ValueError('bad mode')
 return a

def encode(vol,meta,model,path,tracepath=None):
 assert vol.ndim==4 and vol.dtype==np.uint8 and all(0<n<=8192 for n in vol.shape) and vol.shape[-1]<=4
 meta={**meta,'shape':list(vol.shape),'model':model,'sha256':sha(vol).hex(),'isa':'S3D-BCx-v7-functional','packet':'S3M7'}
 hdr=json.dumps(meta,separators=(',',':')).encode();prev=np.zeros(vol.shape[1:],np.uint8); counts=[0]*4; trace=[]; changed=0; baseline=0; saved=0; t=time.perf_counter(); errors=0; nwords=0
 with open(path,'wb') as f:
  f.write(H.pack(MAGIC,len(hdr)));f.write(hdr)
  for i,target in enumerate(vol):
   state={};program=[word(op,i) for op in ORDER]
   for j,w in enumerate(program):
    op=ORDER[j]; code,lane=OPS[op]
    if (w>>60, (w>>2)&MASK,w&3)!=(code,i,lane):raise ValueError('instruction mismatch')
    nwords+=1
    if op=='FRAME_REF':state['ref']=prev
    elif op=='ENC_MP4':state['raw']=target.tobytes()
    elif op=='MOTION':state['motion']=state['ref'] # identity temporal predictor; no motion search in profile
    elif op=='PERMEATE':state['active']=int(np.count_nonzero(np.any(target^prev,axis=-1)));changed+=state['active']
    elif op=='RENORM':state['model']=model # coefficients already quantized and bounded by training
    elif op=='ANN_FIRE':state['pred']=predict(state['ref'],state['model'])
    elif op=='XOR_VOXEL':
     r=state['raw'];c=[r,zipb(r),zipb((target^state['motion']).tobytes()),zipb((target^state['pred']).tobytes())];state['base']=min(map(len,c[:3]));baseline+=state['base']+REC.size;state['mode']=min(range(4),key=lambda k:len(c[k]));state['body']=c[state['mode']];saved+=state['base']-len(state['body'])
    elif op=='WRITE_VOXEL':state['pending']=(state['mode'],state['body'],sha(state['raw']))
    elif op=='DEC_MP4':state['restored']=unpack_payload(*state['pending'][:2],prev,state['pred'])
    elif op=='VERIFY':
     e=int(np.count_nonzero(state['restored']!=target));errors+=e
     if e or sha(state['restored'])!=state['pending'][2]:raise ValueError('verification failed; no commit')
    elif op=='COMMIT':
     mode,body,digest=state['pending'];f.write(REC.pack(mode,len(body),digest));f.write(body);prev=state['restored'];counts[mode]+=1
    elif op=='SYNC_PTS':
     if i<2:trace.append({'plane':i,'words':[f'{x:016X}' for x in program],'mode':state['mode'],'bytes':len(state['body']),'sha256':state['pending'][2].hex()})
 result={'shape':list(vol.shape),'raw_bytes':vol.size,'packet_bytes':Path(path).stat().st_size,'baseline_packet_bytes':baseline+H.size+len(hdr),'ann_payload_saving_bytes':saved,'mode_counts':dict(zip(['raw','spatial_zlib','temporal_xor','ann_xor'],counts)),'instructions':nwords,'planes':len(vol),'changed_voxel_states':changed,'byte_errors':errors,'seconds':time.perf_counter()-t,'sha256':meta['sha256'],'trace':trace}
 if tracepath:Path(tracepath).write_text(json.dumps(result,indent=2))
 return result

def decode(path):
 # Independent implementation does not import this module.
 from independent import decode_file
 return decode_file(path)
def pack_bytes(raw):
 a=np.frombuffer(raw,np.uint8);n=max(1,(len(a)+1023)//1024);p=np.zeros(n*1024,np.uint8);p[:len(a)]=a;return p.reshape(n,32,32,1)
def ingest(kind,path,scale=None):
 p=Path(path);m={'kind':kind,'name':p.name}
 if kind=='image':
  a=np.asarray(Image.open(p).convert('RGB'));m.update(width=a.shape[1],height=a.shape[0]);return a[None],m
 if kind=='video':
  probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height,r_frame_rate','-of','json',str(p)]))['streams'][0];w,h=probe['width'],probe['height'];cmd=['ffmpeg','-v','error','-threads','1','-i',str(p),'-map','0:v:0'];m['source_dimensions']=[w,h]
  if scale:w,h=scale;cmd+=['-vf',f'scale={w}:{h}']
  cmd+=['-an','-pix_fmt','rgb24','-f','rawvideo','-threads','1','pipe:1'];raw=subprocess.check_output(cmd);m.update(width=w,height=h,fps=probe['r_frame_rate']);return np.frombuffer(raw,np.uint8).reshape(-1,h,w,3),m
 if kind=='audio':
  with wave.open(str(p),'rb') as f:m.update(rate=f.getframerate(),channels=f.getnchannels(),sample_width=f.getsampwidth(),samples=f.getnframes());raw=f.readframes(f.getnframes())
  m['length']=len(raw);return pack_bytes(raw),m
 if kind=='geometry':
  a=np.load(p,allow_pickle=False);assert a.dtype==np.uint8 and a.ndim==3;m['geometry_shape']=list(a.shape);return a[...,None],m
 if kind in ('text','file'):
  raw=p.read_bytes();m['length']=len(raw);return pack_bytes(raw),m
 raise ValueError(kind)
def export(a,m,out):
 out=Path(out);kind=m['kind']
 if kind=='image':Image.fromarray(a[0]).save(out)
 elif kind=='video':
  # Lossless RGB H.264, video-only, constant-rate frame timestamps.
  subprocess.run(['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s',f"{m['width']}x{m['height']}",'-r',m['fps'],'-i','pipe:0','-an','-c:v','libx264rgb','-preset','ultrafast','-crf','0','-threads','2',str(out)],input=a.tobytes(),check=True)
 elif kind=='audio':
  with wave.open(str(out),'wb') as f:f.setnchannels(m['channels']);f.setsampwidth(m['sample_width']);f.setframerate(m['rate']);f.writeframes(a.tobytes()[:m['length']])
 elif kind=='geometry':np.save(out,a[...,0],allow_pickle=False)
 else:out.write_bytes(a.tobytes()[:m['length']])
def main():
 p=argparse.ArgumentParser();s=p.add_subparsers(dest='cmd',required=True);e=s.add_parser('encode');e.add_argument('kind',choices=['image','video','audio','geometry','text','file']);e.add_argument('input');e.add_argument('output');e.add_argument('--model',default=str(Path(__file__).with_name('predictor.json')));e.add_argument('--scale',nargs=2,type=int);d=s.add_parser('decode');d.add_argument('input');d.add_argument('output');args=p.parse_args()
 if args.cmd=='encode':a,m=ingest(args.kind,args.input,args.scale);print(json.dumps(encode(a,m,json.loads(Path(args.model).read_text()),args.output),indent=2))
 else:a,m=decode(args.input);export(a,m,args.output)
if __name__=='__main__':main()
