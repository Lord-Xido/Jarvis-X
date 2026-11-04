#!/usr/bin/env python3
"""Local-only HTTP control plane for native Jarvis-X 3D C++ execution core.
Zero third-party Python dependencies. Requires a C++20 compiler once.
"""
import argparse
import ctypes as C
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
EXT = '.dylib' if sys.platform == 'darwin' else '.so'
LIBRARY = ROOT / ('libjx3d' + EXT)


def load_native():
    cpp = ROOT/'src'/'jx_core.cpp'
    h = ROOT/'src'/'jx_core.h'
    ann = ROOT/'src'/'jx_ann.hpp'
    if not LIBRARY.exists() or LIBRARY.stat().st_mtime < max(cpp.stat().st_mtime,h.stat().st_mtime,ann.stat().st_mtime):
        compiler=os.environ.get('CXX','g++' if sys.platform!='darwin' else 'clang++')
        cmd=[compiler,'-O2','-std=c++20','-fPIC','-shared',str(cpp),'-o',str(LIBRARY)]
        print('Building native runtime:', ' '.join(cmd), flush=True)
        subprocess.check_call(cmd)
    return C.CDLL(str(LIBRARY))


class Stats(C.Structure):
    _fields_ = [('ticks',C.c_uint64),('dimension',C.c_uint32),('voxels',C.c_uint32),
                ('active_faults',C.c_uint32),('repaired_faults',C.c_uint32),('total_faults',C.c_uint32),
                ('routes_found',C.c_uint32),('routes_failed',C.c_uint32),('last_route_expansions',C.c_uint32),
                ('last_step_ms',C.c_double),('total_step_ms',C.c_double),('last_codec_ratio',C.c_double),('last_max_delta',C.c_double),
                ('last_encoded_bytes',C.c_uint32),('codec_verified',C.c_uint32),
                ('switching_bits_total',C.c_uint64),('last_switching_bits',C.c_uint64),('byte_updates_total',C.c_uint64),
                ('ann_epochs',C.c_uint64),('ann_initial_mse',C.c_double),('ann_last_mse',C.c_double),('last_ann_ratio',C.c_double),
                ('last_ann_encoded_bytes',C.c_uint32)]


def bind(lib):
    p=C.c_void_p; u=C.c_uint32; b=C.POINTER(C.c_uint8)
    signatures = {
        'jx_create':([u,C.c_uint64],p), 'jx_destroy':([p],None),
        'jx_reset':([p,C.c_uint64],C.c_int), 'jx_step':([p,u],C.c_int),
        'jx_fail':([p,u],C.c_int), 'jx_route':([p,u,u,C.POINTER(u),u],C.c_int),
        'jx_copy_field':([p,b,b,u],C.c_int), 'jx_set_field':([p,b,u],C.c_int),
        'jx_stats':([p,C.POINTER(Stats)],C.c_int),
        'jx_codec_bound':([u],u),
        'jx_encode':([b,u,b,u],C.c_int),
        'jx_decode':([b,u,b,u],C.c_int),
        'jx_encode_field':([p,b,u],C.c_int),
        'jx_import_field':([p,b,u],C.c_int),
        'jx_verify_field':([p],C.c_int),
        'jx_bitplane_bound':([u],u),
        'jx_bitplanes':([p,b,u],C.c_int),
        'jx_ann_train':([p,u,C.c_double],C.c_int),
        'jx_ann_preview':([p,b,u],C.c_int),
        'jx_ann_latents':([p,C.POINTER(C.c_float),u],C.c_int),
        'jx_ann_bound':([u],u),
        'jx_ann_export':([p,b,u],C.c_int),
        'jx_ann_import':([p,b,u],C.c_int)
    }
    for name,(arg,rest) in signatures.items():
        fun=getattr(lib,name); fun.argtypes=arg; fun.restype=rest
    return lib


def array_from_bytes(raw):
    return (C.c_uint8*len(raw)).from_buffer_copy(raw)


class Core:
    def __init__(self,d=10,seed=20261104):
        self.lib=bind(load_native())
        self.pointer=self.lib.jx_create(d,seed)
        if not self.pointer: raise RuntimeError('invalid 3D dimension or memory allocation failure')
        self.d,self.n=d,d**3
        self.lock=threading.RLock()
        self.auto=True
        self.path=[]
        self.route_info={}
        self.started=time.monotonic()
        self.rand=random.Random(seed)
        self.last_operation='Native C++ engine initialized'
        self.last_codec_test=None
        self._seen_healed=0
        self.native_error=None
        self.alive=True
        self.thread=threading.Thread(target=self.worker,daemon=True)
        self.thread.start()
        with self.lock:
            self.lib.jx_verify_field(self.pointer)

    def close(self):
        self.alive=False
        self.thread.join(timeout=2)
        with self.lock:
            self.lib.jx_destroy(self.pointer)
            self.pointer=None

    def worker(self):
        while self.alive:
            t=time.monotonic()
            try:
                with self.lock:
                    if self.auto:
                        self.lib.jx_step(self.pointer,1)
                        st=Stats()
                        self.lib.jx_stats(self.pointer,C.byref(st))
                        if st.repaired_faults>self._seen_healed:
                            recovered=st.repaired_faults-self._seen_healed
                            self.last_operation='Native fault recovery: %d voxel(s) restored'%recovered
                            self._seen_healed=st.repaired_faults
            except Exception as exc:
                self.native_error=str(exc)
                self.auto=False
            time.sleep(max(0.01, 0.1-(time.monotonic()-t)))

    def preview(self):
        with self.lock:
            buffer=(C.c_uint8*self.n)()
            got=self.lib.jx_ann_preview(self.pointer,buffer,self.n)
            if got!=self.n:raise RuntimeError('ANN reconstruction preview failed')
            return list(buffer)

    def train(self,epochs,rate):
        if not 1<=epochs<=200:raise ValueError('epochs must be 1..200')
        if not 0<rate<=10:raise ValueError('learning rate must be in (0,10]')
        with self.lock:
            baseline=Stats();self.lib.jx_stats(self.pointer,C.byref(baseline))
            t=time.perf_counter()
            updated=self.lib.jx_ann_train(self.pointer,epochs,rate)
            if updated<0:raise RuntimeError('native ANN training failed')
            final=Stats();self.lib.jx_stats(self.pointer,C.byref(final))
            self.last_operation=f'Native ANN trained {epochs} epochs, accepted {updated}, MSE {final.ann_last_mse:.7f}'
            return {'epochs':epochs,'accepted_updates':updated,
                'mse_before':baseline.ann_last_mse,'mse_after':final.ann_last_mse,
                'train_ms':round((time.perf_counter()-t)*1000,4),
                'total_epochs':final.ann_epochs,'trainable_parameters':76}

    def bitplanes(self):
        with self.lock:
            capacity=self.lib.jx_bitplane_bound(self.n)
            buffer=(C.c_uint8*capacity)()
            got=self.lib.jx_bitplanes(self.pointer,buffer,capacity)
            if got!=capacity:raise RuntimeError('native bitplane extract failed')
            stride=(self.n+7)//8
            planes=[bytes(buffer[k*stride:(k+1)*stride]) for k in range(8)]
            return {'dimension':self.d,'voxels':self.n,
                    'packing':'bitplane LSB first, voxel i at byte i//8 bit i%8',
                    'bits_set':[sum(byte.bit_count() for byte in p) for p in planes],
                    'packed_hex':[plane.hex() for plane in planes]}

    def ann_export(self):
        with self.lock:
            capacity=self.lib.jx_ann_bound(self.n)
            buffer=(C.c_uint8*capacity)()
            got=self.lib.jx_ann_export(self.pointer,buffer,capacity)
            if got<=0:raise RuntimeError('native neural stream export failed: '+str(got))
            self.last_operation=f'JXA1 neural latent / XOR residual exported ({got} B)'
            return bytes(buffer[:got])

    def ann_import(self,encoded):
        if not (len(encoded)>=24+76*4+24 and len(encoded)<=self.lib.jx_ann_bound(self.n)):
            raise ValueError('invalid or oversize JXA1 neural stream')
        if encoded[:4]!=b'JXA1':raise ValueError('not a JXA1 stream')
        with self.lock:
            result=self.lib.jx_ann_import(self.pointer,array_from_bytes(encoded),len(encoded))
            if result:raise ValueError(f'JXA1 verification rejected stream (code {result})')
            self.path=[];self.route_info={}
            self.last_operation=f'JXA1 neural stream decoded and validated ({self.n} voxels)'
            return {'imported':True,'voxels':self.n,'encoded_bytes':len(encoded),'crc32_checked':True}

    def data(self):
        with self.lock:
            stats=Stats()
            if self.lib.jx_stats(self.pointer,C.byref(stats))!=0: raise RuntimeError('native stats failed')
            values=(C.c_uint8*self.n)(); faults=(C.c_uint8*self.n)()
            if self.lib.jx_copy_field(self.pointer,values,faults,self.n)!=self.n:
                raise RuntimeError('native field read failed')
            d={name:getattr(stats,name) for name,_ in Stats._fields_}
            d.update({
                'values':list(values), 'reconstructed':self.preview(),
                'faults':[i for i,flag in enumerate(faults) if flag],
                'bit_ones':[sum((q>>bit)&1 for q in values) for bit in range(8)],
                'auto':self.auto,'path':self.path,'route':self.route_info,
                'uptime_s':round(time.monotonic()-self.started,2),
                'last_operation':self.last_operation,'native_error':self.native_error,
                'mean_signal':round(sum(values)/(255*self.n),5),
                'codec_test':self.last_codec_test,
                'native_backend':'C++20 / CPU (ctypes shared library)',
            })
            return d

    def encode(self,raw):
        src=array_from_bytes(raw)
        cap=self.lib.jx_codec_bound(len(raw))
        output=(C.c_uint8*cap)()
        n=self.lib.jx_encode(src,len(raw),output,cap)
        if n<0:raise ValueError('native codec failed: '+str(n))
        return bytes(output[:n])

    def decode(self,raw):
        if len(raw)<12: raise ValueError('invalid codec header')
        if raw[:4]!=b'JX3D': raise ValueError('not a JX3D file')
        expected=int.from_bytes(raw[4:8],'little')
        if expected>2_000_000: raise ValueError('file exceeds safe payload limit')
        result=(C.c_uint8*expected)()
        src=array_from_bytes(raw)
        n=self.lib.jx_decode(src,len(raw),result,expected)
        if n!=expected: raise ValueError('CRC mismatch / malformed encoded payload (code %d)'%n)
        return bytes(result)

    def route(self, start, end):
        with self.lock:
            if not (0<=start<self.n and 0<=end<self.n):raise ValueError('route indices out of range')
            path=(C.c_uint32*self.n)()
            t=time.perf_counter()
            length=self.lib.jx_route(self.pointer,start,end,path,self.n)
            ms=(time.perf_counter()-t)*1000
            self.path=list(path[:length]) if length>0 else []
            self.route_info={'from':start,'to':end,'found':length>0,
                             'hops':max(0,length-1),'measured_ms':round(ms,4),
                             'reason':None if length>0 else 'blocked endpoints / disconnected lattice'}
            self.last_operation='Native A* route %s → %s: %s'% (start,end,
                str(length-1)+' hops' if length>0 else 'unreachable')
            return self.route_info

    def fail(self, index):
        with self.lock:
            if index is None:
                failed=self.data()['faults']
                choices=[i for i in range(self.n) if i not in set(failed)]
                if not choices: raise ValueError('no healthy voxel available')
                index=self.rand.choice(choices)
            if not(0<=index<self.n):raise ValueError('fault index out of range')
            status=self.lib.jx_fail(self.pointer,index)
            if status:raise ValueError('voxel is already faulted')
            reroute=False
            path_hit=index in self.path and bool(self.route_info)
            if path_hit:
                a,b=self.route_info['from'],self.route_info['to']
                reroute=self.route(a,b)['found']
            self.last_operation='Native fault injected at voxel %s%s'%(index,
                '; A* auto-reroute '+('succeeded' if reroute else 'unreachable') if path_hit else '')
            return {'faulted':index,'healing_ticks':18}

    def reset(self,seed):
        if seed<0 or seed>=2**64:raise ValueError('seed out of range')
        with self.lock:
            self.lib.jx_reset(self.pointer,seed)
            self.lib.jx_verify_field(self.pointer)
            self.path=[];self.route_info={};self._seen_healed=0
            self.last_operation='Deterministic native reset / seed '+str(seed)
            return {'seed':seed,'reset':True}

    def verify(self):
        with self.lock:
            ok=self.lib.jx_verify_field(self.pointer)==0
            st=Stats();self.lib.jx_stats(self.pointer,C.byref(st))
            self.last_operation='Native grid lossless XOR-residual codec %s'%('VERIFIED' if ok else 'FAILED')
            return {'verified':ok,'raw_bytes':self.n,'encoded_bytes':st.last_encoded_bytes,
                    'ratio':round(st.last_codec_ratio,4),'crc32_checked':ok}

    def export(self):
        with self.lock:
            buf=(C.c_uint8*self.lib.jx_codec_bound(self.n))()
            n=self.lib.jx_encode_field(self.pointer,buf,len(buf))
            if n<0:raise RuntimeError('native export failure')
            self.last_operation='Exported CRC-protected native 3D snapshot: %d bytes'%n
            return bytes(buf[:n])

    def import_bytes(self,encoded):
        raw=self.decode(encoded) # Fully verify before mutating state
        if len(raw)!=self.n: raise ValueError('grid dimension does not match loaded engine')
        with self.lock:
            result=self.lib.jx_import_field(self.pointer,array_from_bytes(encoded),len(encoded))
            if result!=0:raise RuntimeError('native state load failed')
            self.path=[];self.route_info={}
            self.lib.jx_verify_field(self.pointer)
            self.last_operation='Imported and CRC-verified 3D field (%d voxels)'%self.n
            return {'imported':True,'voxels':self.n,'crc32_checked':True}

    def roundtrip_text(self,text):
        raw=text.encode('utf-8')
        if len(raw)>100000:raise ValueError('text exceeds 100 KB')
        t=time.perf_counter()
        encoded=self.encode(raw)
        decoded=self.decode(encoded)
        ms=(time.perf_counter()-t)*1000
        r={'verified':raw==decoded,'raw_bytes':len(raw),'encoded_bytes':len(encoded),
           'ratio':round(len(encoded)/max(1,len(raw)),4),'measured_ms':round(ms,4),
           'decoded_text':decoded.decode('utf-8'),
           'encoded_hex_preview':encoded[:64].hex()}
        with self.lock:
            self.last_codec_test=r
            self.last_operation='Native text byte codec %s'%('VERIFIED' if r['verified'] else 'FAILED')
        return r


class Handler(BaseHTTPRequestHandler):
    server_version='JarvisXNative/11.0'
    def log_message(self, fmt, *args):
        if self.server.verbose:super().log_message(fmt,*args)

    def send(self, status, content, mime='application/json', extra=None):
        if isinstance(content,(dict,list)):
            content=json.dumps(content,separators=(',',':'),allow_nan=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type',mime)
        self.send_header('Content-Length',str(len(content)))
        self.send_header('Cache-Control','no-store')
        self.send_header('X-Content-Type-Options','nosniff')
        self.send_header('X-Frame-Options','DENY')
        if extra:
            for k,v in extra.items():self.send_header(k,v)
        self.end_headers();self.wfile.write(content)

    def do_GET(self):
        path=urlparse(self.path).path
        try:
            if path in ('/','/index.html'):
                return self.send(200,(ROOT/'web'/'index.html').read_bytes(),'text/html; charset=utf-8')
            if path=='/app.js':
                return self.send(200,(ROOT/'web'/'app.js').read_bytes(),'text/javascript; charset=utf-8')
            if path=='/style.css':
                return self.send(200,(ROOT/'web'/'style.css').read_bytes(),'text/css; charset=utf-8')
            if path=='/api/state':return self.send(200,self.server.core.data())
            if path=='/api/bitplanes':return self.send(200,self.server.core.bitplanes())
            if path=='/api/ann/preview':return self.send(200,{'reconstructed':self.server.core.preview()})
            if path=='/api/health':return self.send(200,{'ok':self.server.core.native_error is None,'backend':'native C++20','dimension':self.server.core.d,'error':self.server.core.native_error})
            if path=='/api/ann/export':
                return self.send(200,self.server.core.ann_export(),'application/octet-stream',
                    {'Content-Disposition':'attachment; filename="jarvis-x-ann.jxa"'})
            if path=='/api/export':
                return self.send(200,self.server.core.export(),'application/octet-stream',
                    {'Content-Disposition':'attachment; filename="jarvis-x-field.jx3"'})
            return self.send(404,{'error':'unknown endpoint'})
        except Exception as exc:return self.send(500,{'error':str(exc)})

    def do_POST(self):
        length=int(self.headers.get('Content-Length','0'))
        if length<0 or length>2_500_000:return self.send(413,{'error':'request too large'})
        raw=self.rfile.read(length)
        path=urlparse(self.path).path
        try:
            core=self.server.core
            if path=='/api/import':return self.send(200,core.import_bytes(raw))
            if path=='/api/ann/import':return self.send(200,core.ann_import(raw))
            obj=json.loads(raw or b'{}')
            if not isinstance(obj,dict):raise ValueError('JSON object required')
            if path=='/api/auto':
                with core.lock:
                    core.auto=bool(obj.get('enabled',not core.auto))
                    core.last_operation='Autonomous stepping '+('ON' if core.auto else 'OFF')
                    return self.send(200,{'auto':core.auto})
            if path=='/api/step':
                n=int(obj.get('steps',1))
                if not 1<=n<=100:raise ValueError('steps must be 1..100')
                with core.lock:
                    core.lib.jx_step(core.pointer,n)
                    core.last_operation='Native CPU iterations +'+str(n)
                return self.send(200,{'stepped':n})
            if path=='/api/fault':return self.send(200,core.fail(int(obj['index']) if obj.get('index') is not None else None))
            if path=='/api/route':return self.send(200,core.route(int(obj['from']),int(obj['to'])))
            if path=='/api/reset':return self.send(200,core.reset(int(obj.get('seed',20261104))))
            if path=='/api/ann/train':return self.send(200,core.train(int(obj.get('epochs',20)),float(obj.get('learning_rate',1.0))))
            if path=='/api/verify':return self.send(200,core.verify())
            if path=='/api/codec':return self.send(200,core.roundtrip_text(str(obj.get('text',''))))
            return self.send(404,{'error':'unknown endpoint'})
        except (ValueError,KeyError,TypeError,json.JSONDecodeError) as exc:
            return self.send(400,{'error':str(exc)})
        except Exception as exc:return self.send(500,{'error':str(exc)})


def main():
    p=argparse.ArgumentParser(description='Jarvis-X native 3D CPU engine + web control plane')
    p.add_argument('--port',type=int,default=8765)
    p.add_argument('--dimension',type=int,default=10,help='3D grid side length, even numbers 2..32')
    p.add_argument('--seed',type=int,default=20261104)
    p.add_argument('--host',default='127.0.0.1')
    p.add_argument('--verbose',action='store_true')
    args=p.parse_args()
    core=Core(args.dimension,args.seed)
    server=ThreadingHTTPServer((args.host,args.port),Handler)
    server.core=core;server.verbose=args.verbose
    print('Jarvis-X v11 running at http://%s:%d | C++20 native CPU | %d voxels'%
          (args.host,args.port,core.n),flush=True)
    try:server.serve_forever(poll_interval=0.1)
    except KeyboardInterrupt:pass
    finally:
        server.server_close();core.close()

if __name__=='__main__':main()
