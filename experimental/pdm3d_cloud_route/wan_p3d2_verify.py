#!/usr/bin/env python3
"""HTTPS interoperability proof against synthetic-only public P3D2 receiver."""
import argparse,hashlib,hmac,json,struct,time,zlib
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request,urlopen
from urllib.parse import urlsplit
from uuid import UUID

DATA=bytes((i*71+13)&255 for i in range(42000))
FRAGMENTS=[DATA[i:i+1024] for i in range(0,len(DATA),1024)]
TOKEN='PDM3D_PUBLIC_SYNTHETIC_INTEROP_TOKEN_ONLY_2026'
KEY=hashlib.sha256(b'PDM3D-v2-HMAC-context\x00'+TOKEN.encode()).digest()
UID=UUID('12345678-1234-5678-1234-567812345678')
SHA=hashlib.sha256(DATA).digest()
PREFIX=struct.Struct('!4sBBH16s12sIQH32s')
assert PREFIX.size==82

def packet(i):
    payload=FRAGMENTS[i]
    coord=((7<<60)|(5<<30)|i).to_bytes(12,'big')
    prefix=PREFIX.pack(b'P3D2',2,0,0,UID.bytes,coord,42,42000,len(payload),SHA)
    crc=struct.pack('!I',zlib.crc32(prefix+payload))
    mac=hmac.new(KEY,prefix+crc+payload,hashlib.sha256).digest()[:16]
    return prefix+crc+mac+payload

def invoke(base,method,path,payload=None):
    req=Request(base+path,method=method,data=payload,
        headers={'Content-Type':'application/octet-stream' if payload is not None else 'application/json'})
    for attempt in range(4):
        try:
            with urlopen(req,timeout=25) as r:return r.status,json.loads(r.read())
        except HTTPError as e:
            if e.code in (429,502,503,504) and attempt<3:
                time.sleep(2**attempt);continue
            return e.code,json.loads(e.read())
        except OSError:
            if attempt>=3:raise
            time.sleep(2**attempt)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('url')
    p.add_argument('--report',default='pdm3d-remote-evidence.json')
    args=p.parse_args()
    base=args.url.rstrip('/')
    u=urlsplit(base)
    if u.scheme!='https' or not u.hostname or u.hostname.lower() in ('localhost','127.0.0.1'):
        raise ValueError('an independent public HTTPS endpoint is required')
    start=time.monotonic()
    code,health=invoke(base,'GET','/health')
    assert code==200 and health.get('mode')=='public_synthetic_test_only',health
    sent=0;corrupt=False;dropped=False
    for i in range(42):
        if i==2:dropped=True;continue
        wire=packet(i)
        if i==1:
            bad=wire[:-1]+bytes([wire[-1]^1])
            code,_=invoke(base,'POST','/v1/frame',bad)
            sent+=1
            assert code==422,f'corrupt packet unexpectedly accepted: HTTP {code}'
            corrupt=True
        code,r=invoke(base,'POST','/v1/frame',wire)
        sent+=1
        assert code==201,{'fragment':i,'response':r,'code':code}
    code,f=invoke(base,'GET','/v1/feedback')
    assert code==200 and f['missing']==[2],f
    code,_=invoke(base,'POST','/v1/frame',packet(2))
    sent+=1;assert code==201
    code,receipt=invoke(base,'POST','/v1/compute',b'{}')
    assert code==200 and receipt['verified']
    expected=bytes(v^90 for v in DATA)
    assert receipt['original_sha256']==SHA.hex()
    assert receipt['result_sha256']==hashlib.sha256(expected).hexdigest()
    assert receipt['source_popcount']==sum(v.bit_count() for v in DATA)
    code,f2=invoke(base,'GET','/v1/feedback')
    assert code==200 and f2['received']==0
    report={'status':'REMOTE_SYNTHETIC_P3D2_HTTPS_VERIFIED','scope':'GitHub-hosted runner -> public HTTPS receiver, no sensitive payloads',
        'receiver_url':base,'bytes':42000,'packets':42,'frame_post_attempts':sent,
        'deliberate_loss_recovered':dropped,'deliberate_corruption_rejected':corrupt,
        'payload_sha256':SHA.hex(),'result_sha256':receipt['result_sha256'],
        'elapsed_seconds':round(time.monotonic()-start,3),
        'full_private_cloudbitfabric_v6_deployed':False,'persistent_user_data_stored':False,
        'receiver_feedback':f2}
    Path(args.report).write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    print(json.dumps(report,sort_keys=True,indent=2))

if __name__=='__main__':main()
