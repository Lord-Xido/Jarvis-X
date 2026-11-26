#!/usr/bin/env python3
"""JARVIS X PDF bit VM v1. Requires numpy; PyMuPDF for PDF carrier extraction."""
import argparse, hashlib, json, struct
from pathlib import Path
import numpy as np
MAGIC=b'JXBITVM1'; HEADER=struct.Struct('<8sQI32s')
MAX_BYTES=32*1024*1024; MAX_ROUNDS=100000
LOAD,MAP,XOR,PACK,UNPACK,INV,VERIFY,COMMIT,RECURSE,HALT=1,0x10,0x20,0x21,0x30,0x31,0x60,0x61,0x70,0xff
PROGRAM=bytes([LOAD,MAP,XOR,PACK,UNPACK,INV,VERIFY,COMMIT,RECURSE,HALT])
NAMES=dict(zip(PROGRAM,['LOAD','MAP','XOR','PACK','UNPACK','INV','VERIFY','COMMIT','RECURSE','HALT']))
def digest(b): return hashlib.sha256(b).digest()
def side_for(n):
    s=1
    while s**3<8*n: s*=2
    return s

def coordinate(i):
    return tuple(sum(((i>>(3*k+a))&1)<<k for k in range((i.bit_length()+2)//3)) for a in range(3))

def packet(n,s,h,delta): return HEADER.pack(MAGIC,n,s,h)+np.packbits(delta,bitorder='big').tobytes()
def unpack(p):
    if len(p)<HEADER.size: raise ValueError('truncated header')
    magic,n,s,h=HEADER.unpack_from(p)
    if magic!=MAGIC or not 0<n<=MAX_BYTES or s!=side_for(n) or len(p)!=HEADER.size+n: raise ValueError('invalid packet')
    return n,s,h,np.unpackbits(np.frombuffer(p[HEADER.size:],dtype=np.uint8),bitorder='big')
def inverse(d): return np.bitwise_xor.accumulate(d)
def decode(p):
    n,s,h,d=unpack(p); b=np.packbits(inverse(d),bitorder='big').tobytes()
    if len(b)!=n or digest(b)!=h: raise ValueError('integrity failure')
    return b

def run(source,rounds=1,program=PROGRAM):
    if not 0<len(source)<=MAX_BYTES or not 1<=rounds<=MAX_ROUNDS or program!=PROGRAM: raise ValueError('invalid input or unsupported program')
    committed=source; expected=digest(source); t=0; pc=0; trace=[]; ops=0
    while True:
        op=program[pc]; ops+=1; pc+=1
        if op==LOAD: staged=committed; h=digest(staged)
        elif op==MAP: n=len(staged); s=side_for(n); bits=np.unpackbits(np.frombuffer(staged,dtype=np.uint8),bitorder='big')
        elif op==XOR:
            delta=bits.copy(); delta[1:] ^= bits[:-1]
        elif op==PACK: p=packet(n,s,h,delta)
        elif op==UNPACK: nn,ss,hh,d=unpack(p)
        elif op==INV: candidate=np.packbits(inverse(d),bitorder='big').tobytes()
        elif op==VERIFY:
            if nn!=n or ss!=s or hh!=h or digest(candidate)!=expected or candidate!=staged: raise ValueError('CTR rejected; committed state unchanged')
        elif op==COMMIT:
            committed=candidate; t+=1
            if t in (1,2,10,100,rounds): trace.append({'iteration':t,'bytes':n,'bits':8*n,'side':s,'bit_errors':0,'sha256':expected.hex()})
        elif op==RECURSE:
            if t<rounds: pc=0
        elif op==HALT: return committed,p,{'iterations':t,'opcodes':ops,'bytes':len(source),'bits':8*len(source),'logical_side':s,'logical_cells':s**3,'bit_transitions':8*len(source)*t,'sha256':expected.hex(),'trace':trace}

def extract(carrier,out):
    import fitz
    with fitz.open(carrier) as doc: p=doc.embfile_get('payload.jxb')
    Path(out).write_bytes(decode(p))

def main():
    a=argparse.ArgumentParser(description=__doc__); sub=a.add_subparsers(dest='mode',required=True)
    r=sub.add_parser('run'); r.add_argument('input');r.add_argument('output');r.add_argument('--rounds',type=int,default=1);r.add_argument('--packet');r.add_argument('--trace')
    d=sub.add_parser('decode');d.add_argument('packet');d.add_argument('output')
    e=sub.add_parser('extract');e.add_argument('carrier');e.add_argument('output')
    args=a.parse_args()
    if args.mode=='run':
        b,p,t=run(Path(args.input).read_bytes(),args.rounds); Path(args.output).write_bytes(b)
        if args.packet: Path(args.packet).write_bytes(p)
        if args.trace: Path(args.trace).write_text(json.dumps(t,indent=2)+'\n')
        print(json.dumps(t,indent=2))
    elif args.mode=='decode': Path(args.output).write_bytes(decode(Path(args.packet).read_bytes()))
    else: extract(args.carrier,args.output)
if __name__=='__main__': main()
