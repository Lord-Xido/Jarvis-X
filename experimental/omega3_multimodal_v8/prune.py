import json,subprocess,time,hashlib,sys,statistics
from pathlib import Path
from verify8 import decode
from independent import decode_file
from run8 import samples
P=Path(__file__).parent;src=Path(sys.argv[1]) if len(sys.argv)>1 else P.parent/'omega7'/'outputs';O=P/'results';out={}
for name in ('image','video','audio','text','geometry'):
 a,_=decode_file(src/(name+'.s3m'));a=samples(a,name);raw=O/'prune.raw';a.tofile(raw);q=O/'prune.s3b';rr={1:[],2:[]};hashes={}
 for k in (1,2,2,1,1,2):
  r=json.loads(subprocess.check_output([str(P/'core'),str(raw),str(q),'4','4','4','1','200000','256','0',str(k),'710']));rr[k].append(r);hashes[k]=hashlib.sha256(q.read_bytes()).hexdigest()
 assert hashes[1]==hashes[2];x,y=rr[1][0],rr[2][0]
 for field in ('record_bytes','baseline_bytes','fixed_model_bytes','accepted','hash_chain','weights','modes'):assert x[field]==y[field],field
 out[name]={'standard':x,'pruned':y,'standard_median_seconds':statistics.median(v['seconds'] for v in rr[1]),'pruned_median_seconds':statistics.median(v['seconds'] for v in rr[2]),'sample_packet_sha256':hashes[1]};print(name,out[name]['standard_median_seconds'],out[name]['pruned_median_seconds'],y['skipped'],flush=True)
 raw.unlink();q.unlink()
(P/'pruning.json').write_text(json.dumps(out,indent=2))
