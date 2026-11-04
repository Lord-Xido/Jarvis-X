#!/usr/bin/env python3
"""Independent AES-256-GCM authenticated envelope test (when OpenSSL is linked)."""
from pathlib import Path
import os
import subprocess
import sys
import tempfile

exe=Path(sys.argv[1] if len(sys.argv)>1 else 'build/moagi3d').resolve()
help_text=subprocess.run([str(exe),'--help'],capture_output=True,text=True,check=True).stdout
if 'moagi3d seal ' not in help_text:
    print('SKIP: OpenSSL AES-GCM commands not compiled')
    sys.exit(0)

with tempfile.TemporaryDirectory() as tmp:
    p=Path(tmp)
    key=p/'k.bin';badkey=p/'bad.bin';raw=p/'raw.bin';arc=p/'raw.m3d'
    sealed=p/'raw.m3ds';opened=p/'open.m3d';corrupt=p/'corrupt.m3ds'
    key.write_bytes(os.urandom(32));badkey.write_bytes(os.urandom(32));raw.write_bytes(os.urandom(512))
    def run(*args,ok=True):
        r=subprocess.run([str(exe),*map(str,args)],capture_output=True,text=True)
        assert (r.returncode==0)==ok, (args,r.returncode,r.stdout,r.stderr)
    run('encode',raw,arc,'--dims','8x8x8','--channels','1','--steps','8')
    run('seal',arc,sealed,key)
    run('unseal',sealed,opened,key)
    assert arc.read_bytes()==opened.read_bytes()
    run('unseal',sealed,p/'must_not_be_created.m3d',badkey,ok=False)
    assert not (p/'must_not_be_created.m3d').exists()
    b=bytearray(sealed.read_bytes());b[-1]^=0x80;corrupt.write_bytes(b)
    run('unseal',corrupt,p/'must_not_exist.m3d',key,ok=False)
    assert not (p/'must_not_exist.m3d').exists()
    b=bytearray(sealed.read_bytes());b[8]^=0x80;corrupt.write_bytes(b)
    run('unseal',corrupt,p/'must_not_exist.m3d',key,ok=False)
print('AES-256-GCM PASS: exact unseal, wrong key rejected, ciphertext and nonce tampering rejected')
