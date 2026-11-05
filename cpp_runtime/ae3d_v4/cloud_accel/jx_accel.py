"""JARVIS-X v4 .jxmodel/.jxvol/.jxpkt compatible batched Torch inference.

Device = 'cpu' or 'cuda'; default float64 targets numerical parity with C++.
This is a reference implementation and deliberately excludes training and arbitrary media.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import struct
import zlib
from typing import Iterator

import numpy as np
import torch

BLOCK = 4
D, H, L = 64, 32, 12
PARAMS = 5004
SIDE = 1_000_000_000_000
MAX_VOXELS = 64 * 1024 * 1024
MAX_FILE = 128 * 1024 * 1024
MODEL_MAGIC = b"JXMOD4\r\n"
VOLUME_MAGIC = b"JXVOL4\r\n"
ARCHIVE_MAGIC = b"JXPKT4\r\n"


def _validate_crc(raw: bytes) -> None:
    if len(raw) < 4 or int.from_bytes(raw[-4:], 'little') != zlib.crc32(raw[:-4]):
        raise ValueError("CRC32 mismatch or truncated input")


def _seal(raw: bytes) -> bytes:
    return raw + struct.pack('<I', zlib.crc32(raw))


def read_bounded(path: str | Path) -> bytes:
    path = Path(path)
    if path.stat().st_size > MAX_FILE:
        raise ValueError("file larger than MAX_FILE")
    return path.read_bytes()


def _bounds(x: int, y: int, z: int) -> int:
    if any(not (4 <= n <= 1024 and n % 4 == 0) for n in (x, y, z)):
        raise ValueError("dimensions must be multiples of 4 in [4,1024]")
    n = x*y*z
    if n > MAX_VOXELS:
        raise ValueError("volume allocation exceeds limit")
    return n


def _origin_ok(dims: tuple[int,int,int], origin: tuple[int,int,int]) -> None:
    if len(origin) != 3 or any(not (0 <= p < SIDE and d <= SIDE-p) for p,d in zip(origin, dims)):
        raise ValueError("origin outside virtual 3D domain")


def model_fingerprint(raw: bytes) -> int:
    h = 14695981039346656037
    for x in raw:
        h = ((h ^ x) * 1099511628211) & ((1<<64)-1)
    return h


@dataclass(frozen=True)
class Volume:
    dims: tuple[int,int,int]  # x,y,z
    voxels: np.ndarray  # native spatial storage shape [z,y,x]

    @classmethod
    def parse(cls, raw: bytes) -> 'Volume':
        if len(raw) < 24 or raw[:8] != VOLUME_MAGIC:
            raise ValueError("bad JXVOL4 magic")
        x,y,z = struct.unpack_from('<III',raw,8)
        n = _bounds(x,y,z)
        if len(raw) != 24+n:
            raise ValueError("incorrect JXVOL4 length")
        _validate_crc(raw)
        return cls((x,y,z), np.frombuffer(raw, dtype=np.uint8, count=n, offset=20).copy().reshape((z,y,x)))

    def serialize(self) -> bytes:
        x,y,z = self.dims
        n = _bounds(x,y,z)
        if self.voxels.shape != (z,y,x) or self.voxels.dtype != np.uint8 or self.voxels.size != n:
            raise ValueError("voxel dimensions/dtype mismatch")
        return _seal(VOLUME_MAGIC + struct.pack('<III', x,y,z) + self.voxels.tobytes(order='C'))

    def blocks(self) -> np.ndarray:
        x,y,z = self.dims
        return np.ascontiguousarray(self.voxels.reshape(z//4,4,y//4,4,x//4,4)
                                    .transpose(0,2,4,1,3,5).reshape(-1,64))

    @classmethod
    def from_blocks(cls, dims: tuple[int,int,int], blocks: np.ndarray) -> 'Volume':
        x,y,z = dims
        n = _bounds(x,y,z)
        if blocks.shape != (n//64, 64) or blocks.dtype != np.uint8:
            raise ValueError("invalid decoded block shape")
        voxels = blocks.reshape(z//4,y//4,x//4,4,4,4).transpose(0,3,1,4,2,5).copy().reshape(z,y,x)
        return cls(dims,voxels)


def _rle_pack(src: bytes) -> bytes:
    out = bytearray()
    i = 0
    n = len(src)
    while i < n:
        k = 1
        while i+k < n and k < 130 and src[i+k] == src[i]:
            k += 1
        if k >= 3:
            out.extend((128+k-3,src[i]))
            i += k
            continue
        start = i
        i += 1
        while i<n and i-start < 128:
            run = 1
            while i+run<n and run<3 and src[i+run] == src[i]:
                run += 1
            if run >= 3:
                break
            i += 1
        out.append(i-start-1)
        out.extend(src[start:i])
    return bytes(out)


def _rle_unpack(src: bytes, expected: int) -> bytes:
    result = bytearray()
    pos = 0
    while pos < len(src):
        hdr = src[pos]
        pos += 1
        count = hdr+1 if hdr<128 else hdr-128+3
        if count > expected-len(result):
            raise ValueError("RLE output overflow")
        if hdr<128:
            if pos+count>len(src):
                raise ValueError("RLE truncated literal")
            result.extend(src[pos:pos+count]);pos += count
        else:
            if pos>=len(src):
                raise ValueError("RLE truncated run")
            result.extend(bytes((src[pos],))*count);pos += 1
    if len(result)!=expected:
        raise ValueError("RLE decoded length mismatch")
    return bytes(result)


@dataclass(frozen=True)
class Archive:
    dims: tuple[int,int,int]
    origin: tuple[int,int,int]
    model_hash: int
    exact: bool
    codec: int
    q: bytes
    residual: bytes

    @classmethod
    def parse(cls, raw: bytes) -> 'Archive':
        if len(raw)<74 or raw[:8]!=ARCHIVE_MAGIC:
            raise ValueError("bad JXPKT4 magic")
        x,y,z,ox,oy,oz,model_hash,exact,codec,ls,rs = struct.unpack_from('<IIIQQQQBBQQ',raw,8)
        n = _bounds(x,y,z)
        _origin_ok((x,y,z),(ox,oy,oz))
        if ls!=n//64*12 or rs>MAX_FILE or len(raw)!=70+ls+rs+4 or exact not in (0,1):
            raise ValueError("archive payload length invalid")
        if (exact and (codec not in (1,2) or rs==0)) or (not exact and (codec!=0 or rs!=0)):
            raise ValueError("invalid residual codec")
        if exact and codec == 1 and rs != n:
            raise ValueError("raw residual length mismatch")
        _validate_crc(raw)
        return cls((x,y,z),(ox,oy,oz),model_hash,bool(exact),codec,raw[70:70+ls],raw[70+ls:-4])

    def serialize(self) -> bytes:
        n = _bounds(*self.dims)
        _origin_ok(self.dims, self.origin)
        if len(self.q)!=n//64*12 or (self.exact and (self.codec not in (1,2) or not self.residual)) or (not self.exact and (self.codec!=0 or self.residual)):
            raise ValueError("invalid archive")
        if self.exact and self.codec == 1 and len(self.residual) != n:
            raise ValueError("raw residual length mismatch")
        header = ARCHIVE_MAGIC + struct.pack('<IIIQQQQBBQQ',*self.dims,*self.origin,self.model_hash,int(self.exact),self.codec,len(self.q),len(self.residual))
        return _seal(header+self.q+self.residual)

    def decoded_residual(self) -> bytes | None:
        if not self.exact:
            return None
        expected=_bounds(*self.dims)
        if self.codec == 1:
            if len(self.residual) != expected:
                raise ValueError("raw residual length mismatch")
            return self.residual
        return _rle_unpack(self.residual,expected)


class NeuralCodec:
    """Batched neural inference with optional CUDA, v4 identical on-disk layout.

    Uses FP64 by default: fp32 is significantly faster on some consumer GPUs but
    must not be used for cross-device exact C++ parity without separate checks.
    """
    def __init__(self, raw_model: bytes, device: str = 'cpu', batch_blocks: int = 4096):
        if len(raw_model)!=8+10+8*PARAMS+4 or raw_model[:8]!=MODEL_MAGIC:
            raise ValueError("invalid JXMOD4 model file")
        _validate_crc(raw_model)
        if struct.unpack_from('<HHHI',raw_model,8)!=(D,H,L,PARAMS):
            raise ValueError("model topology mismatch")
        if device=='cuda' and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but unavailable")
        if device not in ('cpu','cuda'):
            raise ValueError("device must be cpu or cuda")
        if not 1<=batch_blocks<=65536:
            raise ValueError("batch_blocks must be in [1,65536]")
        self.device=device
        self.batch_blocks=batch_blocks
        self.raw_model=raw_model
        self.model_hash=model_fingerprint(raw_model)
        self.sha256=hashlib.sha256(raw_model).hexdigest()
        weights=np.frombuffer(raw_model,dtype='<f8',count=PARAMS,offset=18).copy()
        if not np.isfinite(weights).all():
            raise ValueError("nonfinite model weights")
        self.weights=weights
        cursor=0
        shapes=((32,64),(32,),(12,32),(12,),(32,12),(32,),(64,32),(64,))
        arrays=[]
        for shape in shapes:
            count=int(np.prod(shape))
            arrays.append(torch.as_tensor(weights[cursor:cursor+count].reshape(shape).copy(),dtype=torch.float64,device=device))
            cursor+=count
        if cursor!=PARAMS:
            raise RuntimeError("invalid parameter offsets")
        self.w1,self.b1,self.w2,self.b2,self.w3,self.b3,self.w4,self.b4=arrays

    @torch.inference_mode()
    def _encode_chunk(self, blocks: np.ndarray) -> np.ndarray:
        inp=torch.as_tensor(blocks.astype(np.float64)/255.0,device=self.device,dtype=torch.float64)
        h1=torch.tanh(torch.nn.functional.linear(inp,self.w1,self.b1))
        z=torch.tanh(torch.nn.functional.linear(h1,self.w2,self.b2))
        scale=torch.clamp(z,-1,1)*127
        # C++ std::lround: halfway cases go away from zero, unlike torch.round.
        q=torch.where(scale>=0,torch.floor(scale+0.5),torch.ceil(scale-0.5)).to(torch.int8)
        return q.cpu().numpy()

    @torch.inference_mode()
    def _decode_chunk(self, q: np.ndarray) -> np.ndarray:
        inp=torch.as_tensor(q.astype(np.float64)/127.0,device=self.device,dtype=torch.float64)
        h2=torch.tanh(torch.nn.functional.linear(inp,self.w3,self.b3))
        out=torch.sigmoid(torch.nn.functional.linear(h2,self.w4,self.b4))
        # v4 std::lround uses ties-away-from-zero (nonnegative byte values).
        out=torch.floor(torch.clamp(out,0,1)*255+0.5).to(torch.uint8)
        return out.cpu().numpy()

    def encode_blocks(self, blocks: np.ndarray) -> np.ndarray:
        if blocks.ndim!=2 or blocks.shape[1]!=64 or blocks.dtype!=np.uint8:
            raise ValueError("expected uint8 shape [num_blocks,64]")
        return np.concatenate([self._encode_chunk(blocks[i:i+self.batch_blocks]) for i in range(0,len(blocks),self.batch_blocks)],axis=0)

    def decode_blocks(self, q: np.ndarray) -> np.ndarray:
        if q.ndim!=2 or q.shape[1]!=12 or q.dtype!=np.int8:
            raise ValueError("expected int8 shape [num_blocks,12]")
        return np.concatenate([self._decode_chunk(q[i:i+self.batch_blocks]) for i in range(0,len(q),self.batch_blocks)],axis=0)

    def encode(self, vol: Volume, exact: bool, origin: tuple[int,int,int]=(0,0,0)) -> Archive:
        _origin_ok(vol.dims,origin)
        original=vol.blocks()
        q=self.encode_blocks(original)
        if not exact:
            return Archive(vol.dims,origin,self.model_hash,False,0,q.tobytes(),b'')
        prediction=self.decode_blocks(q)
        residual=np.bitwise_xor(original,prediction).tobytes()
        packed=_rle_pack(residual)
        codec = 2 if len(packed)<len(residual) else 1
        return Archive(vol.dims,origin,self.model_hash,True,codec,q.tobytes(),packed if codec==2 else residual)

    def decode(self, arc: Archive) -> Volume:
        if arc.model_hash!=self.model_hash:
            raise ValueError("model fingerprint mismatch")
        n=_bounds(*arc.dims)
        q=np.frombuffer(arc.q,dtype=np.int8).reshape(n//64,12)
        prediction=self.decode_blocks(q)
        if arc.exact:
            original=np.frombuffer(arc.decoded_residual(),dtype=np.uint8).reshape(-1,64)
            prediction=np.bitwise_xor(prediction,original)
        return Volume.from_blocks(arc.dims,prediction)

    def probe(self, arc: Archive, position: tuple[int,int,int]) -> dict:
        if arc.model_hash!=self.model_hash:
            raise ValueError("model fingerprint mismatch")
        dx,dy,dz=(p-o for p,o in zip(position,arc.origin))
        x,y,z=arc.dims
        if not(0<=dx<x and 0<=dy<y and 0<=dz<z):
            raise ValueError("query coordinate out of bounds")
        block=((dz//4)*(y//4)+dy//4)*(x//4)+dx//4
        local=(dz%4)*16+(dy%4)*4+dx%4
        q=np.frombuffer(arc.q,dtype=np.int8).reshape(-1,12)[block:block+1]
        pred=int(self.decode_blocks(q)[0,local])
        exact=pred
        if arc.exact:
            exact^=arc.decoded_residual()[block*64+local]
        return {'predicted':pred,'decoded':exact,'exact':arc.exact}


def volume_from_file(path: str | Path) -> Volume:
    return Volume.parse(read_bounded(path))


def archive_from_file(path: str | Path) -> Archive:
    return Archive.parse(read_bounded(path))
