"""Independent Python 3 verifier for JXAR1T12 synthetic 3D reference archives.

No third-party dependencies. CRC32 is accidental-corruption detection, not authentication.
"""
import math
import struct
import sys
import zlib
from pathlib import Path

TASKS = 10**12
SUPER = 1024_000
VOXELS = 16**3
MASK = (1 << 64) - 1


def f32(v):
    return struct.unpack('<f', struct.pack('<f', v))[0]


def splitmix64(n):
    n = (n + 0x9e3779b97f4a7c15) & MASK
    n = ((n ^ (n >> 30)) * 0xbf58476d1ce4e5b9) & MASK
    n = ((n ^ (n >> 27)) * 0x94d049bb133111eb) & MASK
    return n ^ (n >> 31)


def basis_for(k):
    freq = []
    for total in range(20):
        for x in range(total, -1, -1):
            for y in range(total-x, -1, -1):
                z = total-x-y
                if max(x,y,z) < 16:
                    freq.append((x,y,z))
                    if len(freq) == k:
                        break
            if len(freq) >= k:
                break
        if len(freq) >= k:
            break
    assert len(freq) == k
    rows = []
    for xfreq,yfreq,zfreq in freq:
        scale = math.sqrt(2 / 16) if xfreq else 0.25
        scale *= math.sqrt(2 / 16) if yfreq else 0.25
        scale *= math.sqrt(2 / 16) if zfreq else 0.25
        cx = [math.cos(math.pi*(x+0.5)*xfreq/16) for x in range(16)]
        cy = [math.cos(math.pi*(y+0.5)*yfreq/16) for y in range(16)]
        cz = [math.cos(math.pi*(z+0.5)*zfreq/16) for z in range(16)]
        rows.append([f32(scale*cx[x]*cy[y]*cz[z])
                     for z in range(16) for y in range(16) for x in range(16)])
    return rows


def synthetic(task_id, basis):
    if task_id % 5 == 0:
        return [f32(((splitmix64(task_id ^ ((i*0x9e3779b97f4a7c15) & MASK))
                       % 65536)-32768) / 32768.0) for i in range(VOXELS)]
    b0, b1, b2, b3 = basis[:4]
    return [f32(f32(f32(f32(30*b0[i])+f32(17*b1[i])) - f32(13*b2[i]))
                + f32(7*b3[i])) for i in range(VOXELS)]


def verify(control_path,archive_path):
    c = Path(control_path).read_bytes()
    if len(c)!=SUPER or c[:8]!=b'JXSF1T12':
        raise ValueError('bad control frame dimensions/signature')
    if struct.unpack_from('<I', c, SUPER-4)[0]!=zlib.crc32(c[:-4]):
        raise ValueError('corrupt control frame')
    ar = Path(archive_path).read_bytes()
    if len(ar)<36 or ar[:8]!=b'JXAR1T12':
        raise ValueError('bad archive signature')
    if struct.unpack_from('<I', ar, len(ar)-4)[0]!=zlib.crc32(ar[:-4]):
        raise ValueError('archive CRC mismatch')
    ver, domain, count, control_crc, reserved = struct.unpack_from('<IQIII',ar,8)
    if ver!=1 or domain!=TASKS or count>4096 or reserved or control_crc!=zlib.crc32(c):
        raise ValueError('invalid archive header or control binding')
    pos=32
    prev=-1
    basis=basis_for(64)
    stats={'compressed':0,'raw':0,'minimum_r2':1.0,'mean_r2':0.0,'sign_mismatches':0}
    for _ in range(count):
        if len(ar)-pos < 28:
            raise ValueError('truncated record')
        tid,mode,k,flags,q,claimed_signs,decoded_crc,n = struct.unpack_from('<QBBHfIII',ar,pos)
        pos+=28
        if tid<=prev or tid>=TASKS or flags or n>VOXELS*4 or pos+n>len(ar)-4:
            raise ValueError('invalid task or record length')
        prev=tid
        payload=ar[pos:pos+n]
        pos+=n
        if mode==1 and k in (4,16,64) and n==4*k:
            coefficients=struct.unpack('<'+'f'*k, payload)
            decoded=[0.0]*VOXELS
            for j,coeff in enumerate(coefficients):
                row=basis[j]
                for i in range(VOXELS):
                    decoded[i]=f32(decoded[i]+f32(coeff*row[i]))
            stats['compressed']+=1
        elif mode==2 and k==0 and n==VOXELS*4:
            decoded=struct.unpack('<'+'f'*VOXELS,payload)
            stats['raw']+=1
        else:
            raise ValueError('invalid codec mode and length')
        original=synthetic(tid,basis)
        mean=sum(original)/VOXELS
        err=sum((a-b)**2 for a,b in zip(original,decoded))
        variance=sum((a-mean)**2 for a in original)
        actual=1-err/variance if variance else (1.0 if not err else 0.0)
        signs=sum((a<0)!=(b<0) for a,b in zip(original,decoded))
        if actual<0.98 or signs or claimed_signs or abs(q-actual)>0.0001:
            raise ValueError(f'quality invalid at tile {tid}: R2={actual}, sign_errors={signs}')
        stats['minimum_r2']=min(stats['minimum_r2'],actual)
        stats['mean_r2']+=actual
        stats['sign_mismatches']+=signs
    if pos!=len(ar)-4:
        raise ValueError('trailing data')
    stats['mean_r2']/=max(1,count)
    print('INDEPENDENT PYTHON PASS: records=', count, stats)


if __name__=='__main__':
    if len(sys.argv)!=3:
        raise SystemExit('Usage: python3 verify_archive.py control.jxsf tiles.jxar')
    verify(sys.argv[1],sys.argv[2])
