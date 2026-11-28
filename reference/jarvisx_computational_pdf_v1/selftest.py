import jarvisx as j, tempfile, pathlib, random, subprocess, json, hashlib, shutil, sys
R=random.Random(42); checks=[]
def check(name,condition):
    if not condition: raise AssertionError(name)
    checks.append(name)
with tempfile.TemporaryDirectory() as d:
    p=pathlib.Path(d); a=p/'in'; b=p/'encoded'; c=p/'decoded'; t=p/'trace'; b2=p/'encoded2';t2=p/'trace2'
    start=j.time.perf_counter()
    for i in range(1000):
        xyz=tuple(R.randrange(1<<30) for _ in range(3)); assert j.unmorton(j.morton(*xyz))==xyz
        raw=R.randbytes(1+i%512); enc,_,m=j.transform(raw); dec,_,m2=j.transform(enc,True);assert dec==raw and m.w==m2.w and m.bias==m2.bias
    check('1000 Morton round trips + 1000 ANN codec round trips and model matches',True)
    sample=(bytes(range(256))*4+bytes([128])*512+R.randbytes(512))*3+bytes([17])*31
    a.write_bytes(sample); metrics=j.encode(a,b,t,1);j.decode(b,c);check('full-file round trip',c.read_bytes()==sample)
    j.encode(a,b2,t2,4);check('1/4 worker deterministic container and trace',b.read_bytes()==b2.read_bytes() and t.read_bytes()==t2.read_bytes())
    normal_trace=[json.loads(x) for x in t.read_text().splitlines()]
    fault=j.encode(a,b2,t2,4,2);j.decode(b2,c);fault_trace=[json.loads(x) for x in t2.read_text().splitlines()]
    check('all injected faults rejected; source reconstructed',c.read_bytes()==sample and all(x['reason']=='integrity' for x in fault_trace[1::2]))
    corrupt=bytearray(b.read_bytes());corrupt[50]^=1;b2.write_bytes(corrupt);c.write_bytes(b'keep')
    try: j.decode(b2,c); raise AssertionError('bad data accepted')
    except (ValueError,j.zlib.error): pass
    check('corrupt container rejected; prior output preserved',c.read_bytes()==b'keep')
    b2.write_bytes(b.read_bytes()[:-1]);
    try: j.decode(b2,c); raise AssertionError('truncated accepted')
    except ValueError: pass
    check('truncation rejected; prior output preserved',c.read_bytes()==b'keep')
    a.write_bytes(b'');j.encode(a,b2);j.decode(b2,c);check('empty file',c.read_bytes()==b'')
    a.write_bytes(sample);j.generate(sample,512,p/'test.pgm','pgm');j.generate(sample,512,p/'test.wav','wav')
    import wave
    with wave.open(str(p/'test.wav')) as f: check('generated WAV valid',f.getnframes()==512 and f.getframerate()==8000)
    check('generated PGM valid',(p/'test.pgm').read_bytes().startswith(b'P5\n8 64\n255\n') and len((p/'test.pgm').read_bytes())==524)
    code=p/'program.bin';code.write_bytes(b''.join(j.struct.pack('>Q',w) for w in j.PROGRAM));exe=p/'native'
    subprocess.run(['g++','-std=c++17','-O2','-Wall','-Wextra','-pedantic',str(pathlib.Path(__file__).with_name('jx_native.cpp')),'-o',str(exe)],check=True)
    for i in range(32):
        raw=R.randbytes(i*16+1);a.write_bytes(raw);subprocess.run([str(exe),str(code),str(a),str(c)],check=True,capture_output=True);assert c.read_bytes()==j.transform(raw)[0]
    check('32 native C++ VM vs Python exact residual comparisons',True)
    code.write_bytes(bytes(48)); result=subprocess.run([str(exe),str(code),str(a),str(c)],capture_output=True);check('native rejects invalid bytecode',result.returncode!=0)
    print(json.dumps({'checks':checks,'count':len(checks),'metrics':metrics,'fault_metrics':fault,'fault_rejections':sum(x['reason']=='integrity' for x in fault_trace),'sample_sha256':hashlib.sha256(sample).hexdigest(),'elapsed_seconds':j.time.perf_counter()-start,'sample_trace':normal_trace[0]},indent=2))
