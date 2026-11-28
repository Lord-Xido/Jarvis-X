import json,hashlib,subprocess,time,sys,zlib
from pathlib import Path
import numpy as np
from independent import decode_file
from verify8 import decode
P=Path(__file__).resolve().parent; O=P/'results';O.mkdir(exist_ok=True); source=Path(sys.argv[1]) if len(sys.argv)>1 else P.parent/'omega7'/'outputs'
names=['image','video','audio','text','geometry','file','gen_image','gen_video','gen_audio','gen_text','gen_geometry']
def hash8(a):return hashlib.sha256(np.ascontiguousarray(a)).hexdigest()
def native(a,tag,it=None,ex=None,z=True,opt=True):
 a=np.ascontiguousarray(a);n=len(a);it=it or n;ex=ex or n;raw=O/(tag+'.raw');p=O/(tag+'.s3b');a.tofile(raw);t=time.perf_counter();r=json.loads(subprocess.check_output([str(P/'core'),str(raw),str(p),*map(str,a.shape[1:]),str(it),str(ex),str(int(z)),str(int(opt)),'710']));r['wall_seconds']=time.perf_counter()-t;raw.unlink();b=decode(p);expected=a[np.arange(ex)%n];assert np.array_equal(b,expected);r.update(independent_blocks=ex,independent_bytes=b.size,sha256=hash8(b),packet_file_bytes=p.stat().st_size);return r,b

def tiles(a,shape):
 z,y,x,c=a.shape;d,h,w=shape;zp=(z+d-1)//d;yp=(y+h-1)//h;xp=(x+w-1)//w
 p=np.pad(a,((0,zp*d-z),(0,yp*h-y),(0,xp*w-x),(0,0)));t=p.reshape(zp,d,yp,h,xp,w,c).transpose(2,4,0,1,3,5,6).reshape(-1,d,h,w,c);return t,(zp,yp,xp)
def untile(t,shape,counts):
 z,y,x,c=shape;zp,yp,xp=counts;_,d,h,w,_=t.shape
 return t.reshape(yp,xp,zp,d,h,w,c).transpose(2,3,0,4,1,5,6).reshape(zp*d,yp*h,xp*w,c)[:z,:y,:x,:]
def samples(a,kind):
 if kind=='image':
  v=np.ascontiguousarray(a[0]).reshape(-1);k=np.linspace(0,len(v)-64,256,dtype=int);return np.stack([v[i:i+64].reshape(4,4,4,1) for i in k])
 t,_=tiles(a[...,0:1],(4,4,4));k=np.linspace(0,len(t)-1,256,dtype=int);return t[k]

def main():
 start=time.perf_counter();report={'benchmark':{},'regression':{},'negative':{}};A={}
 for name in names:
  a,m=decode_file(source/(name+'.s3m'));assert hash8(a)==m['sha256'];A[name]=(a,m)
 for name in names[:5]:
  a=A[name][0];s=samples(a,name);r,b=native(s,'million_'+name,200000,256,False);report['benchmark'][name]=r;print('BENCH',name,r,flush=True)
 report['benchmark_total']={k:sum(q[k] for q in report['benchmark'].values()) for k in ['iterations','instructions','candidate_evaluations','accepted','verified_byte_comparisons','record_bytes','baseline_bytes','fixed_model_bytes','seconds','independent_blocks','independent_bytes']}
 (P/'benchmark.json').write_text(json.dumps(report,indent=2))
 for name,(a,m) in A.items():
  t,counts=tiles(a,(min(4,len(a)),32,32));r,b=native(t,'media_'+name);rest=untile(b,a.shape,counts);assert np.array_equal(rest,a);r.update(original_bytes=a.size,original_shape=list(a.shape),tile_counts=counts,source_sha256=hash8(rest),padding_bytes=t.size-a.size,legacy_packet_bytes=(source/(name+'.s3m')).stat().st_size,metadata=m);report['regression'][name]=r;print('MEDIA',name,r['packet_file_bytes'],r['legacy_packet_bytes'],r['seconds'],flush=True)
  r['v8_metadata_bytes']=len(json.dumps({'metadata':m,'tile_counts':counts},separators=(',',':')).encode());r['v8_total_with_metadata']=r['packet_file_bytes']+r['v8_metadata_bytes'];r['selected_format']='v8' if r['v8_total_with_metadata']<r['legacy_packet_bytes'] else 'v7';r['selected_bytes']=min(r['v8_total_with_metadata'],r['legacy_packet_bytes'])
 p=O/'million_image.s3b';raw=p.read_bytes()
 for name,b in {'truncation':raw[:-1],'trailing':raw+b'x','magic':b'BAD!'+raw[4:],'mode':raw[:24]+bytes([255])+raw[25:],'crc':raw[:29]+bytes([raw[29]^1])+raw[30:],'payload':raw[:-1]+bytes([raw[-1]^1])}.items():
  q=O/'bad';q.write_bytes(b)
  try:decode(q)
  except (ValueError,zlib.error):report['negative'][name]='rejected'
  else:raise AssertionError(name)
 q.unlink();report['seconds_total']=time.perf_counter()-start;(P/'verification.json').write_text(json.dumps(report,indent=2));print('PASS',report['benchmark_total'],flush=True)
if __name__=='__main__':main()
