#!/usr/bin/env python3
"""JARVIS-X web VM: sparse 3D address space, verified snapshots, neural retrieval."""
import argparse, base64, hashlib, html, ipaddress, json, math, socket, sqlite3, time
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit, urljoin, urlencode
from urllib.request import Request, build_opener, HTTPRedirectHandler, ProxyHandler
from pdf_core import ANN, address, inverse, pack, unpack, sha, tokens, feat, unit, np, SIDE, MAX_PAGES, CHUNK

MAX_BYTES=8*1024*1024
PROGRAM=bytes([1,3,17,49,64,80,96,255])
OP=dict(zip(PROGRAM,['LOAD','MAP_3D','ENCODE','SEARCH','DECODE','VERIFY','CTR','EMIT']))
SCHEMA='''
CREATE TABLE IF NOT EXISTS web_pages(p INTEGER PRIMARY KEY CHECK(p>=0 AND p<1000000), url TEXT NOT NULL, digest TEXT NOT NULL, raw BLOB NOT NULL, size INTEGER NOT NULL, text BLOB NOT NULL, text_digest TEXT NOT NULL, fetched TEXT NOT NULL, UNIQUE(url,digest));
CREATE TABLE IF NOT EXISTS web_chunks(id INTEGER PRIMARY KEY, p INTEGER NOT NULL REFERENCES web_pages(p), off INTEGER NOT NULL, n INTEGER NOT NULL, f BLOB NOT NULL, body TEXT NOT NULL, UNIQUE(p,off));
CREATE VIRTUAL TABLE IF NOT EXISTS web_fts USING fts5(body, tokenize='unicode61');
CREATE TABLE IF NOT EXISTS web_state(k TEXT PRIMARY KEY,v BLOB NOT NULL);
'''
class Extract(HTMLParser):
    def __init__(self): super().__init__(convert_charrefs=True); self.parts=[]; self.skip=[]; self.links=[]
    def handle_starttag(self,tag,attrs):
        if tag in ('script','style','template','noscript'): self.skip.append(tag)
        if not self.skip:
            if tag=='a':
                href=dict(attrs).get('href')
                if href: self.links.append(href)
            if tag in ('p','div','br','h1','h2','h3','li','td','pre','article','section'): self.parts.append(' ')
    def handle_endtag(self,tag):
        if self.skip and tag==self.skip[-1]: self.skip.pop()
        if not self.skip and tag in ('p','div','h1','h2','h3','li','td','pre','article','section'): self.parts.append(' ')
    def handle_data(self,data):
        if not self.skip: self.parts.append(data)
def extract(raw):
    p=Extract(); p.feed(raw.decode('utf-8',errors='replace')); p.close(); return ' '.join(''.join(p.parts).split()),p.links

def validate_url(url,local=False):
    u=urlsplit(url)
    if u.scheme not in ('http','https') or not u.hostname or u.username or u.password: raise ValueError('only HTTP(S) URLs without credentials')
    if local and u.hostname=='127.0.0.1': return
    for r in socket.getaddrinfo(u.hostname,u.port or (443 if u.scheme=='https' else 80),type=socket.SOCK_STREAM):
        if not ipaddress.ip_address(r[4][0]).is_global: raise ValueError('non-public address blocked')
class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs): return None

def fetch(url,local=False,expected='html'):
    # Explicit URLs only; no automatic redirects, cookies or JavaScript execution.
    # DNS validation is best effort, not a hardened SSRF sandbox (see README).
    validate_url(url,local); op=build_opener(ProxyHandler({}),NoRedirect())
    with op.open(Request(url,headers={'User-Agent':'JarvisWebVM/1.0','Accept':'text/html,application/xhtml+xml' if expected=='html' else 'application/json'}),timeout=12) as r:
        mime=r.headers.get_content_type()
        if mime not in (('text/html','application/xhtml+xml') if expected=='html' else ('application/json',)): raise ValueError('unexpected content type: '+mime)
        if expected=='html' and r.headers.get_content_charset() not in (None,'utf-8','utf8','us-ascii'): raise ValueError('only UTF-8 HTML supported')
        raw=r.read(MAX_BYTES+1)
        if len(raw)>MAX_BYTES: raise ValueError('response too large')
        return raw

def search_provider(endpoint,q,k=5,local=False):
    # Configurable JSON metasearch gateway with {results:[{url:...}]} contract.
    sep='&' if '?' in endpoint else '?'; data=json.loads(fetch(endpoint+sep+urlencode({'q':q,'format':'json'}),local,'json'))
    result=[]
    for item in data['results']:
        url=item.get('url')
        if isinstance(url,str) and urlsplit(url).scheme in ('https','http') and url not in result: result.append(url)
        if len(result)>=k: break
    return result

class WebVM:
    def __init__(self,path):
        self.db=sqlite3.connect(path); self.db.execute('PRAGMA foreign_keys=ON'); self.db.executescript(SCHEMA)
        r=self.db.execute("SELECT v FROM web_state WHERE k='ann'").fetchone(); self.ann=ANN(r[0] if r else None)
    def close(self): self.db.close()
    def ingest(self,url,raw,at=None):
        if len(raw)>MAX_BYTES: raise ValueError('document too large')
        d=sha(raw); text,_=extract(raw); b=text.encode(); ts=tokens(text)
        with self.db:
            self.db.execute('BEGIN IMMEDIATE')
            row=self.db.execute('SELECT p FROM web_pages WHERE url=? AND digest=?',(url,d)).fetchone()
            if row: return row[0]
            p=self.db.execute('SELECT COALESCE(MAX(p)+1,0) FROM web_pages').fetchone()[0]
            if p>=MAX_PAGES: raise ValueError('capacity exceeded')
            enc=pack(raw)
            if unpack(enc)!=raw: raise ValueError('roundtrip failure')
            self.db.execute('INSERT INTO web_pages VALUES(?,?,?,?,?,?,?,?)',(p,url,d,enc,len(raw),pack(b),sha(b),at or time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
            off=0
            while off<len(ts):
                n=min(CHUNK,SIDE-off%SIDE,len(ts)-off); part=ts[off:off+n]; body=' '.join(part)
                cur=self.db.execute('INSERT INTO web_chunks(p,off,n,f,body) VALUES(?,?,?,?,?)',(p,off,n,feat(part).astype('<f4').tobytes(),body))
                self.db.execute('INSERT INTO web_fts(rowid,body) VALUES(?,?)',(cur.lastrowid,body)); off+=n
        return p
    def raw(self,p):
        url,d,c,n,at=self.db.execute('SELECT url,digest,raw,size,fetched FROM web_pages WHERE p=?',(p,)).fetchone()
        b=unpack(c)
        if len(b)!=n or sha(b)!=d: raise ValueError('snapshot integrity failure')
        return b
    def verify(self):
        pages=chunks=total=0
        for p,c,d in self.db.execute('SELECT p,text,text_digest FROM web_pages ORDER BY p'):
            b=self.raw(p); text=extract(b)[0]; ts=tokens(text); total+=len(b); pages+=1
            if unpack(c)!=text.encode() or sha(text.encode())!=d: raise ValueError('text mismatch')
            off=0
            for i,start,n,f,body in self.db.execute('SELECT id,off,n,f,body FROM web_chunks WHERE p=? ORDER BY off',(p,)):
                part=ts[off:off+n]
                if start!=off or n!=min(CHUNK,SIDE-off%SIDE,len(ts)-off) or n<=0 or inverse(*address(p,start))!=(p,start) or body!=' '.join(part) or f!=feat(part).astype('<f4').tobytes(): raise ValueError('chunk mismatch')
                if self.db.execute('SELECT body FROM web_fts WHERE rowid=?',(i,)).fetchone()!=(body,): raise ValueError('index mismatch')
                off+=n; chunks+=1
            if off!=len(ts): raise ValueError('incomplete coverage')
        if self.db.execute('SELECT count(*) FROM web_fts').fetchone()[0]!=chunks: raise ValueError('orphan index')
        return {'pages':pages,'chunks':chunks,'source_bytes':total,'bit_errors':0}
    def seal(self):
        h=hashlib.sha256()
        for tab,order in [('web_pages','p'),('web_chunks','id'),('web_fts','rowid')]:
            for row in self.db.execute(f'SELECT * FROM {tab} ORDER BY {order}'):
                for v in row:
                    b=v if isinstance(v,bytes) else str(v).encode(); h.update(len(b).to_bytes(8,'little')); h.update(b)
        return h.hexdigest()
    def matrix(self,limit=4096):
        r=self.db.execute('SELECT f FROM web_chunks ORDER BY id LIMIT ?',(limit,)).fetchall()
        if not r: raise ValueError('empty corpus')
        return np.stack([np.frombuffer(x[0],dtype='<f4') for x in r])
    def query(self,q,k=5,ann=None,exact=False):
        if not 1<=k<=100: raise ValueError('k must be 1..100')
        a=ann or self.ann; ts=tokens(q); terms=sorted({t for t in ts if t.isalnum()})[:64]; qf=feat(ts); qz=unit(a.encode(qf))
        rows=[]; mode='fts5+ann'
        if terms and not exact:
            expr=' OR '.join('"'+t+'"' for t in terms)
            rows=self.db.execute('SELECT c.id,c.p,c.off,c.n,c.f,c.body FROM web_fts JOIN web_chunks c ON c.id=web_fts.rowid WHERE web_fts MATCH ? ORDER BY bm25(web_fts),c.id LIMIT 2048',(expr,)).fetchall()
        if exact or not rows:
            mode='full-scan'; cur=self.db.execute('SELECT id,p,off,n,f,body FROM web_chunks ORDER BY id')
        else: cur=None
        import heapq
        heap=[]
        while True:
            batch=cur.fetchmany(512) if cur is not None else rows
            if not batch: break
            f=np.stack([np.frombuffer(r[4],dtype='<f4') for r in batch]); s=.45*(unit(a.encode(f))@qz)+.35*(f@qf)
            for r,v in zip(batch,s):
                i,p,off,n,_,body=r; score=float(v)+.2*len(set(terms)&set(body.split()))/max(1,len(terms)); item=(score,-i,p,off,n)
                if len(heap)<k: heapq.heappush(heap,item)
                elif item>heap[0]: heapq.heapreplace(heap,item)
            if cur is None: break
        out=[]
        for score,_,p,off,n in sorted(heap,reverse=True):
            url,d,at=self.db.execute('SELECT url,digest,fetched FROM web_pages WHERE p=?',(p,)).fetchone(); b=self.raw(p); ts=tokens(extract(b)[0]); x,y,z,slab=address(p,off)
            out.append({'url':url,'sha256':d,'fetched_utc':at,'xyz':[x,y,z],'slab':slab,'token_offset':off,'score':score,'text':' '.join(ts[off:off+n]),'retrieval_mode':mode})
        return out
    def cycle(self,x,probes=(),inject=False,seal=None):
        path=[]
        for op in PROGRAM:
            path.append(OP[op])
            if op==1: old=self.ann.blob(); baseline=self.ann.loss(x)
            elif op==3:
                if seal is not None and self.seal()!=seal: raise ValueError('corpus changed')
                for p,o in self.db.execute('SELECT p,off FROM web_chunks'):
                    if inverse(*address(p,o))!=(p,o): raise ValueError('address mismatch')
            elif op==17:
                a=ANN(old); a.step(x)
                if inject: a.W[:]=np.nan
            elif op==49:
                rank=lambda model: sum(bool(r and r[0]['url']==u) for q,u in probes for r in [self.query(q,1,model)])
                hits=rank(a) if np.isfinite(a.W).all() else -1
            elif op==64:
                restored=unpack(pack(a.blob())); decoded=a.decode(a.encode(x))
                if restored!=a.blob() or unpack(pack(old))!=old: raise ValueError('model byte mismatch')
            elif op==80:
                loss=float(np.mean((decoded-x)**2)); ok=bool(np.isfinite(loss) and loss<=baseline and hits==len(probes))
            elif op==96:
                if ok:
                    with self.db: self.db.execute("INSERT OR REPLACE INTO web_state VALUES('ann',?)",(restored,))
                    self.ann=a
                elif self.ann.blob()!=old: raise ValueError('rollback failed')
                path.append('COMMIT' if ok else 'ROLLBACK')
            elif op==255: return {'accepted':ok,'loss':self.ann.loss(x),'model_sha256':sha(self.ann.blob()),'probe_hits':rank(self.ann),'opcode_path':path}

def selftest(out,iterations):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from threading import Thread
    from urllib.parse import parse_qs
    out=Path(out); out.mkdir(parents=True,exist_ok=True); db=out/'web.sqlite'; db.unlink(missing_ok=True)
    source=Path(__file__).read_bytes(); core=Path(__file__).with_name('pdf_core.py').read_bytes()
    self_html=(b'<html><body><h1>Self source archive</h1><pre id="web-source">'+base64.b64encode(source)+b'</pre><pre id="pdf-core">'+base64.b64encode(core)+b'</pre></body></html>')
    fixtures={'/alpha':b'<html><h1>NEBULAFORGE</h1><p>Ocean thermal energy conversion.</p><script>ignored malicious content</script></html>', '/beta':b'<html><h1>QUARTZHELIX</h1><p>Recursive neural autoencoding uses latent reconstruction.</p></html>', '/long':b'<html><p>'+b'continuation '*1100+b'</p></html>', '/self':self_html}
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args): pass
        def do_GET(self):
            p=urlsplit(self.path).path
            if p=='/search':
                q=parse_qs(urlsplit(self.path).query).get('q',[''])[0]; paths=['/beta','/alpha'] if 'neural' in q else list(fixtures)
                b=json.dumps({'results':[{'url':base+x} for x in paths]}).encode(); mime='application/json'
            elif p in fixtures: b=fixtures[p]; mime='text/html; charset=utf-8'
            elif p=='/redirect': self.send_response(302); self.send_header('Location',base+'/alpha'); self.end_headers(); return
            else: self.send_error(404); return
            self.send_response(200); self.send_header('Content-Type',mime); self.send_header('Content-Length',str(len(b))); self.end_headers(); self.wfile.write(b)
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler); base=f'http://127.0.0.1:{server.server_port}'; thread=Thread(target=server.serve_forever,daemon=True); thread.start()
    vm=WebVM(db); start=time.perf_counter()
    try:
        urls=search_provider(base+'/search','all',4,True); assert len(urls)==4
        for u in urls: vm.ingest(u,fetch(u,True),'2026-01-01T00:00:00Z')
        assert search_provider(base+'/search','neural',1,True)==[base+'/beta']
        assert vm.ingest(base+'/alpha',fixtures['/alpha'])==0
        assert len(vm.query('NEBULAFORGE',1))==1
        assert 'ignored malicious content' not in extract(vm.raw(0))[0]
        try: fetch(base+'/alpha'); raise AssertionError('private host accepted')
        except ValueError: pass
        try: fetch(base+'/redirect',True); raise AssertionError('redirect accepted')
        except Exception as e:
            from urllib.error import HTTPError
            assert isinstance(e,HTTPError) and e.code==302
        p=3; b=vm.raw(p); assert b==self_html
        import re
        for tag,original in [('web-source',source),('pdf-core',core)]: assert base64.b64decode(re.search(b'<pre id="'+tag.encode()+b'">(.*?)</pre>',b).group(1))==original
        (out/'self.html').write_bytes(b)
        assert inverse(*address(999999,2001))==(999999,2001)
        assert vm.db.execute('SELECT count(*) FROM web_chunks WHERE off>=1000').fetchone()[0]>0
        probes=[('NEBULAFORGE',base+'/alpha'),('QUARTZHELIX',base+'/beta')]
        for q,u in probes: assert vm.query(q,1)[0]['url']==u and vm.query(q,1,exact=True)[0]['url']==u
        checked=vm.verify(); seal=vm.seal(); x=vm.matrix(); before=vm.ann.loss(x); accepted=0; prev=before
        (out/'program.bin').write_bytes(PROGRAM)
        with (out/'trace.jsonl').open('w') as f:
            for i in range(1,iterations+1):
                rec=vm.cycle(x,probes,i==max(1,iterations//2),seal); assert rec['loss']<=prev and rec['probe_hits']==len(probes); prev=rec['loss']; accepted+=rec['accepted']; f.write(json.dumps({'iteration':i,**rec,'corpus_seal':seal},allow_nan=False)+'\n')
        vm.db.execute('UPDATE web_pages SET text_digest=? WHERE p=0',('0'*64,))
        caught=False
        try: vm.verify()
        except ValueError: caught=True
        finally: vm.db.rollback()
        assert caught; model=vm.ann.blob(); vm.close(); vm=WebVM(db); assert vm.ann.blob()==model; assert vm.seal()==seal; assert vm.verify()==checked
        result={'iterations':iterations,'accepted':accepted,'rolled_back':iterations-accepted,'initial_loss':before,'final_loss':vm.ann.loss(x),'virtual_pages':MAX_PAGES,'base_token_slots':SIDE**3,'search_provider':'local HTTP fixture','live_public_web_tested':False,'self_source_roundtrip':True,'pdf_core_source_roundtrip':True,'continuation_test':True,'private_host_blocked':True,'redirect_blocked':True,'corruption_detected':caught,'restart_verified':True,'retrieval_probes':len(probes),'bytecode_hex':PROGRAM.hex(),'elapsed_seconds':round(time.perf_counter()-start,3),**checked}
        (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))
    finally: vm.close(); server.shutdown(); server.server_close(); thread.join()

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--db',default='omega_web.sqlite'); s=p.add_subparsers(dest='cmd',required=True)
    a=s.add_parser('fetch'); a.add_argument('url',nargs='+')
    a=s.add_parser('import-html'); a.add_argument('url'); a.add_argument('file')
    a=s.add_parser('web-search'); a.add_argument('query'); a.add_argument('--endpoint',required=True); a.add_argument('--k',type=int,default=5)
    a=s.add_parser('query'); a.add_argument('query'); a.add_argument('--k',type=int,default=5); a.add_argument('--exact',action='store_true')
    a=s.add_parser('train'); a.add_argument('--steps',type=int,default=100)
    s.add_parser('verify'); a=s.add_parser('restore'); a.add_argument('page',type=int); a.add_argument('output')
    a=s.add_parser('selftest'); a.add_argument('--iterations',type=int,default=1000); a.add_argument('--out',default='test_output')
    args=p.parse_args()
    if args.cmd=='selftest':
        if args.iterations<1: p.error('iterations must be positive')
        return selftest(args.out,args.iterations)
    vm=WebVM(args.db)
    try:
        if args.cmd=='fetch': result=[{'url':u,'page':vm.ingest(u,fetch(u))} for u in args.url]
        elif args.cmd=='import-html': result={'page':vm.ingest(args.url,Path(args.file).read_bytes())}
        elif args.cmd=='web-search':
            if not 1<=args.k<=100: p.error('k must be 1..100')
            errors=[]; fetched=[]
            for u in search_provider(args.endpoint,args.query,args.k):
                try: fetched.append({'url':u,'page':vm.ingest(u,fetch(u))})
                except Exception as e: errors.append({'url':u,'error':str(e)})
            result={'fetched':fetched,'errors':errors,'evidence':vm.query(args.query,args.k)}
        elif args.cmd=='query': result=vm.query(args.query,args.k,exact=args.exact)
        elif args.cmd=='verify': result=vm.verify()
        elif args.cmd=='restore': Path(args.output).write_bytes(vm.raw(args.page)); result={'output':args.output,'sha256':sha(vm.raw(args.page))}
        elif args.cmd=='train':
            if args.steps<1: p.error('steps must be positive')
            checked=vm.verify(); x=vm.matrix(); seal=vm.seal(); before=vm.ann.loss(x); count=0
            for _ in range(args.steps): count+=vm.cycle(x,seal=seal)['accepted']
            result={'accepted':count,'steps':args.steps,'initial_loss':before,'final_loss':vm.ann.loss(x),'verification':checked,'retrieval_probe_count':0}
        print(json.dumps(result,indent=2,allow_nan=False))
    finally: vm.close()
if __name__=='__main__': main()
