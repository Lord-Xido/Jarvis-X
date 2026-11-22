#!/usr/bin/env python3
"""Synthetic-only PDM3D P3D2 internet receiver. NOT a private or production v6 service."""
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hashlib,hmac,json,os,struct,threading,time,zlib
from uuid import UUID

VERSION='pdm3d-public-synthetic-edge-v1'
TOKEN='PDM3D_PUBLIC_SYNTHETIC_INTEROP_TOKEN_ONLY_2026'
UID=UUID('12345678-1234-5678-1234-567812345678').bytes
DATA=bytes((i*71+13)&255 for i in range(42000))
CHUNKS=[DATA[i:i+1024] for i in range(0,len(DATA),1024)]
DIGEST=hashlib.sha256(DATA).digest()
KEY=hashlib.sha256(b'PDM3D-v2-HMAC-context\x00'+TOKEN.encode()).digest()
HEADER=struct.Struct('!4sBBH16s12sIQH32sI16s')
assert HEADER.size==102 and len(CHUNKS)==42
LOCK=threading.RLock()
STATE={'received':{},'started':0.0,'accepted':0,'rejected':0,'duplicates':0,'completed':0}
REQUESTS=deque()

def decode(frame):
    if not 102<=len(frame)<=1126:raise ValueError('frame size')
    magic,version,flags,reserved,uid,coords,n,total,length,digest,crc,mac=HEADER.unpack_from(frame)
    body=frame[102:]
    if (magic,version,flags,reserved)!=(b'P3D2',2,0,0):raise ValueError('version')
    if uid!=UID or n!=42 or total!=42000 or digest!=DIGEST:raise ValueError('fixture')
    if coords[0]&252:raise ValueError('address padding')
    addr=int.from_bytes(coords,'big')
    x,y,z=(addr>>60)&((1<<30)-1),(addr>>30)&((1<<30)-1),addr&((1<<30)-1)
    if (x,y)!=(7,5) or z>=42:raise ValueError('xyz')
    if length!=len(body) or body!=CHUNKS[z]:raise ValueError('synthetic only')
    if not hmac.compare_digest(mac,hmac.new(KEY,frame[:86]+body,hashlib.sha256).digest()[:16]):
        raise ValueError('HMAC')
    if crc!=zlib.crc32(frame[:82]+body):raise ValueError('CRC')
    return z,body

def expire():
    if STATE['received'] and time.monotonic()-STATE['started']>300:
        STATE['received'].clear();STATE['started']=0.0

class Handler(BaseHTTPRequestHandler):
    server_version='PDM3DSyntheticEdge/1'
    sys_version=''
    def log_message(self,*args):pass
    def respond(self,status,data):
        raw=(json.dumps(data,separators=(',',':'),sort_keys=True)+'\n').encode()
        self.send_response(status)
        for k,v in [('Content-Type','application/json'),('Content-Length',str(len(raw))),('Cache-Control','no-store'),('X-Content-Type-Options','nosniff')]:
            self.send_header(k,v)
        self.end_headers();self.wfile.write(raw)
    def throttle(self):
        with LOCK:
            now=time.monotonic()
            while REQUESTS and now-REQUESTS[0]>60:REQUESTS.popleft()
            if len(REQUESTS)>=250:
                self.respond(429,{'error':'rate_limited'});return True
            REQUESTS.append(now);return False
    def do_GET(self):
        if self.path=='/health':
            return self.respond(200,{'ok':True,'version':VERSION,'mode':'public_synthetic_test_only',
                'permanent_data_store':False,'full_cloudbitfabric_v6':False})
        if self.path=='/v1/feedback':
            with LOCK:
                expire()
                return self.respond(200,{'received':len(STATE['received']),
                    'missing':[i for i in range(42) if i not in STATE['received']],
                    'accepted':STATE['accepted'],'rejected':STATE['rejected'],
                    'duplicates':STATE['duplicates'],'completed':STATE['completed'],
                    'mode':'synthetic_only'})
        return self.respond(404,{'error':'not_found'})
    def do_POST(self):
        if self.throttle():return
        try:
            n=int(self.headers.get('Content-Length','-1'))
            if not 0<=n<=1126:return self.respond(413,{'error':'too_large'})
            body=self.rfile.read(n)
            if len(body)!=n:raise ValueError('truncated')
            if self.path=='/v1/frame':
                z,payload=decode(body)
                with LOCK:
                    expire()
                    if not STATE['received']:STATE['started']=time.monotonic()
                    if z in STATE['received']:STATE['duplicates']+=1
                    else:STATE['received'][z]=payload;STATE['accepted']+=1
                    return self.respond(201,{'accepted':True,'fragment':z,'coordinate':[7,5,z]})
            if self.path=='/v1/compute':
                if body not in (b'',b'{}'):raise ValueError('unsupported request')
                with LOCK:
                    expire()
                    if len(STATE['received'])!=42:
                        return self.respond(409,{'error':'missing_fragments',
                            'missing':[i for i in range(42) if i not in STATE['received']]})
                    original=b''.join(STATE['received'][i] for i in range(42))
                    if original!=DATA or hashlib.sha256(original).digest()!=DIGEST:
                        raise ValueError('source mismatch')
                    result=bytes(v^90 for v in original)
                    STATE['completed']+=1;STATE['received'].clear()
                    return self.respond(200,{'verified':True,'operation':'xor8','parameter':90,
                        'original_bytes':42000,'fragments':42,'original_sha256':DIGEST.hex(),
                        'result_sha256':hashlib.sha256(result).hexdigest(),
                        'source_popcount':sum(v.bit_count() for v in original),
                        'session_discarded':True,'scope':'synthetic-only remote HTTPS compute'})
            return self.respond(404,{'error':'not_found'})
        except (ValueError,struct.error):
            with LOCK:STATE['rejected']+=1
            return self.respond(422,{'error':'invalid_synthetic_frame'})

if __name__=='__main__':
    port=int(os.environ.get('PORT','10000'))
    server=ThreadingHTTPServer(('0.0.0.0',port),Handler)
    server.daemon_threads=True
    print(f'PDM3D_SYNTHETIC_EDGE_READY port={port}',flush=True)
    server.serve_forever(poll_interval=.25)
