#!/usr/bin/env python3
"""JARVIS-X sparse 3D PDF VM: deterministic reference, not a full LLM."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse, hashlib, io, json, re, sqlite3, time, zlib
from pathlib import Path
import numpy as np
import fitz

SIDE=1000; MAX_PAGES=SIDE**2; CHUNK=128; D=128; H=32
TOKEN=re.compile(r'\w+|[^\w\s]',re.UNICODE)

def sha(b): return hashlib.sha256(b).hexdigest()
def tokens(t): return TOKEN.findall(t.casefold())
def address(p, offset):
    if not 0<=p<MAX_PAGES or offset<0: raise ValueError('address out of range')
    slab,z=divmod(offset,SIDE)
    return p//SIDE,p%SIDE,z,slab

def inverse(x,y,z,slab):
    if not all(0<=v<SIDE for v in (x,y,z)) or slab<0: raise ValueError('bad coordinate')
    return x*SIDE+y, slab*SIDE+z

def pack(b):
    # Predict byte i from byte i-1; XOR residual, then lossless DEFLATE.
    a=np.frombuffer(b,dtype=np.uint8)
    r=a.copy()
    if len(a)>1: r[1:]=a[1:]^a[:-1]
    return zlib.compress(r.tobytes(),6)

def unpack(c):
    r=np.frombuffer(zlib.decompress(c),dtype=np.uint8)
    return np.bitwise_xor.accumulate(r).tobytes()

def feat(ts):
    a=np.zeros(D,dtype=np.float32)
    for t in ts:
        if not any(c.isalnum() for c in t): continue
        h=hashlib.blake2b(t.encode(),digest_size=8).digest()
        k=int.from_bytes(h,'little'); a[k%D]+=1 if (k>>8)&1 else -1
    n=np.linalg.norm(a)
    return a/n if n else a

def unit(a): return a/np.maximum(np.linalg.norm(a,axis=-1,keepdims=True),1e-12)

class ANN:
    """Trainable 128 -> 32 -> 128 tanh / linear autoencoder."""
    def __init__(self, blob=None):
        if blob is None:
            rng=np.random.default_rng(42)
            self.W=(rng.standard_normal((D,H))*0.06).astype('f4')
            self.V=(rng.standard_normal((H,D))*0.06).astype('f4')
        else:
            arr=np.frombuffer(blob,dtype='<f4')
            if arr.size!=2*D*H: raise ValueError('invalid ANN state')
            self.W=arr[:D*H].reshape(D,H).copy(); self.V=arr[D*H:].reshape(H,D).copy()
    def blob(self): return self.W.astype('<f4').tobytes()+self.V.astype('<f4').tobytes()
    def encode(self,x): return np.tanh(x@self.W)
    def decode(self,z): return z@self.V
    def loss(self,x): return float(np.mean((self.decode(self.encode(x))-x)**2))
    def step(self,x,lr=.2):
        h=self.encode(x); e=(h@self.V-x)*(2/x.size)
        dw=x.T@((e@self.V.T)*(1-h*h)); dv=h.T@e
        self.W-=lr*dw; self.V-=lr*dv

SCHEMA='''
CREATE TABLE IF NOT EXISTS docs(id TEXT PRIMARY KEY, name TEXT, raw BLOB, nbytes INTEGER, npages INTEGER);
CREATE TABLE IF NOT EXISTS pages(p INTEGER PRIMARY KEY CHECK(p>=0 AND p<1000000), doc TEXT REFERENCES docs(id), local INTEGER, text BLOB, digest TEXT, UNIQUE(doc,local));
CREATE TABLE IF NOT EXISTS chunks(p INTEGER REFERENCES pages(p), off INTEGER, n INTEGER, f BLOB, body TEXT, PRIMARY KEY(p,off));
CREATE TABLE IF NOT EXISTS state(k TEXT PRIMARY KEY,v BLOB);
'''
class VM:
    def __init__(self,path):
        self.db=sqlite3.connect(path); self.db.execute('PRAGMA foreign_keys=ON'); self.db.executescript(SCHEMA)
        row=self.db.execute("SELECT v FROM state WHERE k='ann'").fetchone(); self.ann=ANN(row[0] if row else None)
    def close(self): self.db.close()
    def ingest(self,path):
        raw=Path(path).read_bytes(); d=sha(raw)
        if self.db.execute('SELECT 1 FROM docs WHERE id=?',(d,)).fetchone(): return d
        with fitz.open(stream=raw,filetype='pdf') as pdf, self.db:
            if pdf.needs_pass: raise ValueError('encrypted PDF requires prior decryption')
            p=self.db.execute('SELECT COALESCE(MAX(p)+1,0) FROM pages').fetchone()[0]
            if p+len(pdf)>MAX_PAGES: raise ValueError('virtual page capacity exceeded')
            enc=pack(raw)
            if unpack(enc)!=raw: raise RuntimeError('binary round-trip failed')
            self.db.execute('INSERT INTO docs VALUES(?,?,?,?,?)',(d,Path(path).name,enc,len(raw),len(pdf)))
            for local,page in enumerate(pdf):
                text=page.get_text(sort=True); b=text.encode(); ts=tokens(text)
                self.db.execute('INSERT INTO pages VALUES(?,?,?,?,?)',(p,d,local,pack(b),sha(b)))
                off=0
                while off<len(ts):
                    n=min(CHUNK,SIDE-off%SIDE,len(ts)-off); part=ts[off:off+n]
                    self.db.execute('INSERT INTO chunks VALUES(?,?,?,?,?)',(p,off,n,feat(part).astype('<f4').tobytes(),' '.join(part)))
                    off+=n
                p+=1
        return d
    def raw(self,d):
        row=self.db.execute('SELECT raw,nbytes FROM docs WHERE id=?',(d,)).fetchone()
        if not row: raise KeyError(d)
        raw=unpack(row[0]);
        if len(raw)!=row[1] or sha(raw)!=d: raise ValueError('document integrity failure')
        return raw
    def verify(self):
        count=0; nc=0; nb=0
        for d, in self.db.execute('SELECT id FROM docs'):
            raw=self.raw(d); nb+=len(raw)
            with fitz.open(stream=raw,filetype='pdf') as pdf:
                rows=self.db.execute('SELECT p,local,text,digest FROM pages WHERE doc=? ORDER BY local',(d,)).fetchall()
                if len(rows)!=len(pdf): raise ValueError('missing page')
                for expected,(p,local,c,digest) in enumerate(rows):
                    b=unpack(c)
                    if expected!=local or sha(b)!=digest or b!=pdf[local].get_text(sort=True).encode(): raise ValueError('page integrity failure')
                    ts=tokens(b.decode()); off=0; count+=1
                    for start,n,f,body in self.db.execute('SELECT off,n,f,body FROM chunks WHERE p=? ORDER BY off',(p,)):
                        if start!=off or n!=min(CHUNK,SIDE-start%SIDE,len(ts)-start) or n<=0: raise ValueError('chunk coverage failure')
                        part=ts[start:start+n]
                        if inverse(*address(p,start))!=(p,start) or body!=' '.join(part) or f!=feat(part).astype('<f4').tobytes(): raise ValueError('chunk integrity failure')
                        off+=n; nc+=1
                    if off!=len(ts): raise ValueError('missing tokens')
        return {'pages':count,'chunks':nc,'source_bytes':nb,'bit_errors':0}
    def matrix(self,limit=4096):
        rows=self.db.execute('SELECT f FROM chunks ORDER BY p,off LIMIT ?',(limit,)).fetchall()
        if not rows: raise ValueError('no text chunks')
        return np.stack([np.frombuffer(r[0],dtype='<f4') for r in rows])
    def save_ann(self,ann):
        with self.db: self.db.execute("INSERT OR REPLACE INTO state VALUES('ann',?)",(ann.blob(),))
        self.ann=ann
    def query(self,q,k=5,ann=None,banks=0):
        if k<1 or banks<0: raise ValueError('invalid retrieval limits')
        ann=ann or self.ann; qt=tokens(q); qf=feat(qt); qz=unit(ann.encode(qf)); terms={t for t in qt if t.isalnum()}
        # Optional hierarchical coarse bank selection; exact search by default.
        selected=None
        if banks:
            sums={}
            for p,f in self.db.execute('SELECT p,f FROM chunks'):
                b=p//SIDE; sums.setdefault(b,np.zeros(D,dtype='f4')); sums[b]+=np.frombuffer(f,dtype='<f4')
            selected=set(sorted(sums,key=lambda b:(-float(unit(sums[b])@qf),b))[:banks])
        import heapq
        heap=[]; cur=self.db.execute('SELECT p,off,n,f,body FROM chunks ORDER BY p,off')
        while True:
            rows=cur.fetchmany(512)
            if not rows: break
            if selected is not None: rows=[r for r in rows if r[0]//SIDE in selected]
            if not rows: continue
            f=np.stack([np.frombuffer(r[3],dtype='<f4') for r in rows]); scores=.45*(unit(ann.encode(f))@qz)+.35*(f@qf)
            for r,s in zip(rows,scores):
                p,off,n,_,body=r; lex=len(terms.intersection(body.split()))/max(1,len(terms)); score=float(s)+.2*lex
                item=(score,-p,-off,n)
                if len(heap)<k: heapq.heappush(heap,item)
                elif item>heap[0]: heapq.heapreplace(heap,item)
        out=[]
        for s,np_,no,n in sorted(heap,reverse=True):
            p,off=-np_,-no; d,local,c,digest=self.db.execute('SELECT doc,local,text,digest FROM pages WHERE p=?',(p,)).fetchone(); b=unpack(c)
            if sha(b)!=digest: raise ValueError('retrieved page integrity failure')
            # Verify original document identity before returning citation.
            with fitz.open(stream=self.raw(d),filetype="pdf") as pdf:
                if pdf[local].get_text(sort=True).encode()!=b: raise ValueError("retrieved source mismatch")
            text=b.decode(); matches=list(TOKEN.finditer(text));
            if len(matches)==len(tokens(text)):
                snippet=text[matches[off].start():matches[off+n-1].end()]
            else: snippet=' '.join(tokens(text)[off:off+n])
            x,y,z,slab=address(p,off)
            out.append({'score':s,'xyz':[x,y,z],'slab':slab,'document_sha256':d,'pdf_page':local+1,'token_offset':off,'text':snippet})
        return out
    def train(self,steps=100):
        x=self.matrix(); a=ANN(self.ann.blob()); before=a.loss(x)
        for _ in range(steps): a.step(x)
        after=a.loss(x)
        if not np.isfinite(after) or after>before: raise ValueError('candidate rejected')
        self.save_ann(a); return {'before':before,'after':after,'samples':len(x)}

def make_fixture(path,source):
    import textwrap
    pdf=fitz.open(); page=pdf.new_page(); page.insert_text((48,65),'JARVIS-X SELF SPECIFICATION',fontsize=16)
    page.insert_textbox(fitz.Rect(48,95,550,780),'ZEPHYRNEEDLE: exact binary recovery uses XOR residual and SHA256 verification.\n\nThe virtual machine maps pages into a 1000 by 1000 address space. Each page uses token positions and continuation slabs.\n\nQUARTZANCHOR: neural autoencoding uses a tanh encoder and linear decoder, and accepted states are committed after verification.',fontsize=12)
    # Force continuation in a page with >1000 tokens.
    page=pdf.new_page(); page.insert_textbox(fitz.Rect(25,25,570,810),' '.join(['continuation']*1050),fontsize=6)
    # Self-description as visible source and exact source as an embedded attachment.
    lines=[]
    for l in source.decode().splitlines(): lines.extend(textwrap.wrap(l,width=100,replace_whitespace=False,drop_whitespace=False) or [''])
    for i in range(0,len(lines),65):
        page=pdf.new_page(); page.insert_text((25,25),'\n'.join(lines[i:i+65]),fontname='cour',fontsize=8)
    pdf.embfile_add('vm.py',source,filename='vm.py'); pdf.save(path,garbage=4,deflate=True,no_new_id=True); pdf.close()

OP={1:'LOAD',3:'MAP_3D',17:'ENCODE',64:'DECODE',80:'VERIFY',96:'CTR',255:'EMIT'}
PROGRAM=bytes(OP)

def seal(vm):
    h=hashlib.sha256()
    for table,order in [('docs','id'),('pages','p'),('chunks','p,off')]:
        for row in vm.db.execute(f'SELECT * FROM {table} ORDER BY {order}'):
            for v in row:
                b=v if isinstance(v,bytes) else str(v).encode()
                h.update(len(b).to_bytes(8,'little')); h.update(b)
    return h.hexdigest()

def execute(vm,program,c):
    if program!=PROGRAM: raise ValueError('invalid bytecode program')
    path=[]
    for op in program:
        path.append(OP[op])
        if op==1:
            old=vm.ann.blob(); raw=vm.raw(c['d'])
        elif op==3:
            if seal(vm)!=c['seal']: raise ValueError('corpus changed')
            for p,off in c['addresses']:
                if inverse(*address(p,off))!=(p,off): raise ValueError('address failure')
        elif op==17:
            candidate=ANN(old); candidate.step(c['x'])
            if c['inject']: candidate.W[:]=np.nan
        elif op==64:
            restored=ANN(unpack(pack(old)))
            if restored.blob()!=old or unpack(pack(candidate.blob()))!=candidate.blob(): raise ValueError('model round-trip failure')
            decoded=candidate.decode(candidate.encode(c['x']))
        elif op==80:
            with fitz.open(stream=raw,filetype='pdf') as pdf:
                if pdf.embfile_get('vm.py')!=c['source']: raise ValueError('self source mismatch')
            loss=float(np.mean((decoded-c['x'])**2))
            valid=bool(np.isfinite(loss) and loss<=restored.loss(c['x']) and c['rank'](candidate)==2)
        elif op==96:
            if valid: vm.save_ann(candidate); path.append('COMMIT')
            else:
                if vm.ann.blob()!=old: raise ValueError('rollback failure')
                path.append('ROLLBACK')
        elif op==255:
            return {'opcode_path':path,'accepted':valid,'loss':vm.ann.loss(c['x']),'model_sha256':sha(vm.ann.blob()),'retrieval_probes_passed':c['rank'](vm.ann)}

def selftest(out,iterations):
    out=Path(out); out.mkdir(parents=True,exist_ok=True); db=out/'selftest.sqlite'; db.unlink(missing_ok=True)
    source=Path(__file__).read_bytes(); fixture=out/'self.pdf'; fixture.unlink(missing_ok=True); make_fixture(fixture,source)
    vm=VM(db); d=vm.ingest(fixture); assert vm.ingest(fixture)==d
    with fitz.open(stream=vm.raw(d),filetype='pdf') as pdf: assert pdf.embfile_get('vm.py')==source
    assert inverse(*address(999999,2001))==(999999,2001)
    assert vm.db.execute('SELECT COUNT(*) FROM chunks WHERE off>=1000').fetchone()[0]>0
    for b in (b'',b'\0',bytes(range(256))*4): assert unpack(pack(b))==b
    x=vm.matrix(); before=vm.ann.loss(x); accept=reject=0; start=time.perf_counter(); trace=out/'trace.jsonl'
    probes=['ZEPHYRNEEDLE','QUARTZANCHOR']; expected=[1,1]
    def rank(a): return sum(bool(r and r[0]['pdf_page']==p) for r,p in zip([vm.query(q,1,a) for q in probes],expected))
    assert rank(vm.ann)==len(probes)
    checks=vm.verify(); digest=seal(vm); addresses=vm.db.execute('SELECT p,off FROM chunks').fetchall()
    (out/'program.bin').write_bytes(PROGRAM)
    with trace.open('w') as f:
        for t in range(1,iterations+1):
            rec=execute(vm,PROGRAM,{'d':d,'source':source,'x':x,'seal':digest,'addresses':addresses,'rank':rank,'inject':t==max(1,iterations//2)})
            accept+=int(rec['accepted']); reject+=int(not rec['accepted'])
            f.write(json.dumps({'iteration':t,**rec,**checks,'corpus_seal':digest},allow_nan=False)+'\n')
    # Negative integrity test: alter a stored digest within an uncommitted transaction.
    vm.db.execute('UPDATE pages SET digest=? WHERE p=0',('0'*64,)); caught=False
    try: vm.verify()
    except ValueError: caught=True
    finally: vm.db.rollback()
    assert caught
    final=vm.ann.blob(); vm.close(); vm=VM(db); assert vm.ann.blob()==final
    result={'iterations':iterations,'accepted':accept,'rolled_back':reject,'initial_loss':before,'final_loss':vm.ann.loss(x),'source_sha256':sha(source),'pdf_sha256':d,'corruption_detected':caught,'restart_model_verified':True,'probes_passed':rank(vm.ann),'virtual_page_capacity':MAX_PAGES,'corpus_seal':digest,'bytecode_hex':PROGRAM.hex(),'elapsed_seconds':round(time.perf_counter()-start,3),**vm.verify()}
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2)); vm.close()

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--db',default='omega.sqlite'); sub=p.add_subparsers(dest='cmd',required=True)
    a=sub.add_parser('ingest'); a.add_argument('pdf',nargs='+')
    a=sub.add_parser('query'); a.add_argument('text'); a.add_argument('-k',type=int,default=5); a.add_argument('--banks',type=int,default=0)
    a=sub.add_parser('restore'); a.add_argument('sha256'); a.add_argument('output')
    a=sub.add_parser('train'); a.add_argument('--steps',type=int,default=100)
    sub.add_parser('verify')
    a=sub.add_parser('selftest'); a.add_argument('--out',default='test_output'); a.add_argument('--iterations',type=int,default=1000)
    a=p.parse_args()
    if a.cmd=='selftest':
        if a.iterations<1: p.error('iterations must be positive')
        return selftest(a.out,a.iterations)
    vm=VM(a.db)
    try:
        if a.cmd=='ingest': result=[vm.ingest(f) for f in a.pdf]
        elif a.cmd=='query': result=vm.query(a.text,a.k,banks=a.banks)
        elif a.cmd=='restore': Path(a.output).write_bytes(vm.raw(a.sha256)); result={'restored':a.output}
        elif a.cmd=='train': result=vm.train(a.steps)
        else: result=vm.verify()
        print(json.dumps(result,indent=2,ensure_ascii=False))
    finally: vm.close()
if __name__=='__main__': main()
