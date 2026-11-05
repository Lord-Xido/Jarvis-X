"""JX3D v1: bounded, portable 3D voxel encoder/decoder.

Lossless mode uses 2x2x2 integer block means and XOR residuals.
Preview mode intentionally drops residuals. Neither mode is a trained ANN.
"""
from __future__ import annotations

import hashlib
import math
import struct
import zlib
from dataclasses import dataclass

MAGIC = b"JX3D"
VERSION = 1
MAX_DIM = 64
HEADER = struct.Struct(">4sBBHHHII32s")
MAX_PACKET = 1_500_000

class CodecError(ValueError):
    pass

@dataclass(frozen=True)
class Decoded:
    dim: int
    volume: bytes
    mode: str
    sha256: str
    checksum_verified: bool


def checked_dim(dim: int) -> int:
    if isinstance(dim, bool) or not isinstance(dim, int) or not (16 <= dim <= MAX_DIM) or dim % 2:
        raise CodecError("Dimension must be an even integer between 16 and 64")
    return dim


def hash_volume(volume: bytes) -> str:
    return hashlib.sha256(volume).hexdigest()


def _coarse_grid(volume: bytes, dim: int) -> bytes:
    cd = dim // 2
    coarse = bytearray(cd ** 3)
    for z in range(cd):
        zz = 2 * z
        for y in range(cd):
            yy = 2 * y
            for x in range(cd):
                xx = 2 * x
                s = 0
                for dz in (0, 1):
                    for dy in (0, 1):
                        row = ((zz+dz) * dim + yy+dy) * dim + xx
                        s += volume[row] + volume[row + 1]
                coarse[(z * cd + y) * cd + x] = (s + 4) // 8
    return bytes(coarse)


def _upsample(coarse: bytes, dim: int) -> bytearray:
    cd = dim // 2
    out = bytearray(dim ** 3)
    for z in range(dim):
        cz = (z // 2) * cd * cd
        for y in range(dim):
            cy = (y // 2) * cd
            row = (z * dim + y) * dim
            for x in range(dim):
                out[row+x] = coarse[cz + cy + x//2]
    return out


def encode(volume: bytes, dim: int, mode: str = "lossless") -> bytes:
    checked_dim(dim)
    if not isinstance(volume, bytes) or len(volume) != dim**3:
        raise CodecError("Expected precisely dim³ bytes of uint8 voxel data")
    if mode not in ("lossless", "preview"):
        raise CodecError("Mode must be 'lossless' or 'preview'")
    coarse = _coarse_grid(volume, dim)
    predicted = _upsample(coarse, dim)
    if mode == "lossless":
        residual = bytes(a ^ b for a, b in zip(volume, predicted))
        payload = coarse + residual
        expected_hash = hashlib.sha256(volume).digest()
        flags = 0
    else:
        payload = coarse
        expected_hash = hashlib.sha256(predicted).digest()
        flags = 1
    packed = zlib.compress(payload, 6)
    return HEADER.pack(MAGIC, VERSION, flags, dim, dim, dim,
                       len(coarse), len(packed), expected_hash) + packed


def decode(packet: bytes) -> Decoded:
    if not isinstance(packet, bytes) or not (HEADER.size <= len(packet) <= MAX_PACKET):
        raise CodecError("Invalid JX3D packet size")
    magic, version, flags, dx, dy, dz, clen, zlen, digest = HEADER.unpack_from(packet)
    if magic != MAGIC or version != VERSION or flags not in (0, 1):
        raise CodecError("Unsupported or damaged JX3D header")
    checked_dim(dx)
    if dx != dy or dx != dz or clen != (dx//2)**3 or zlen != len(packet)-HEADER.size:
        raise CodecError("Inconsistent JX3D dimensions or lengths")
    expected = clen + (dx**3 if flags == 0 else 0)
    try:
        inflater = zlib.decompressobj()
        payload = inflater.decompress(packet[HEADER.size:], expected + 1)
        if len(payload) != expected or not inflater.eof or inflater.unused_data or inflater.unconsumed_tail:
            raise CodecError("Compressed payload is corrupt or oversized")
    except zlib.error as exc:
        raise CodecError("Invalid compressed JX3D body") from exc
    reconstruction = _upsample(payload[:clen], dx)
    if flags == 0:
        for i, residual in enumerate(payload[clen:]):
            reconstruction[i] ^= residual
    actual = hashlib.sha256(reconstruction).digest()
    if actual != digest:
        raise CodecError("SHA-256 reconstruction checksum mismatch")
    return Decoded(dx, bytes(reconstruction), "preview" if flags else "lossless", actual.hex(), True)


def generate(dim: int, geometry: str = "mandelbulb", seed: int = 7) -> bytes:
    checked_dim(dim)
    if geometry not in ("mandelbulb", "torus", "sphere", "waves"):
        raise CodecError("Unknown geometry")
    data = bytearray(dim**3)
    phase = seed * 0.017
    for z in range(dim):
        pz = (2*z/(dim-1) - 1)
        for y in range(dim):
            py = (2*y/(dim-1) - 1)
            for x in range(dim):
                px = (2*x/(dim-1) - 1)
                if geometry == "sphere":
                    distance = math.sqrt(px*px + py*py + pz*pz) - 0.70
                    v = 255 / (1 + math.exp(max(-40, min(40, distance*42))))
                elif geometry == "torus":
                    radial = math.hypot(px, py)
                    distance = math.hypot(radial - 0.56, pz) - 0.225
                    v = 255 / (1 + math.exp(max(-40, min(40, distance*48))))
                elif geometry == "waves":
                    w = (math.sin(9*px + phase)*math.cos(8*py - phase) +
                         math.sin(10*pz + 3*px)*0.65)
                    envelope = math.exp(-1.2*(px*px+py*py+pz*pz))
                    v = 255 * max(0.0, min(1.0, 0.55+0.4*w*envelope))
                else:
                    # Mandelbulb escape-time field, power 8, bounded 7 iterations.
                    a, b, c = px*1.28, py*1.28, pz*1.28
                    zx, zy, zz = a, b, c
                    count = 0
                    for _ in range(7):
                        radius = math.sqrt(zx*zx + zy*zy + zz*zz)
                        if radius > 2.0:
                            break
                        count += 1
                        theta = math.acos(max(-1.0, min(1.0, zz/max(radius,1e-10))))
                        phi = math.atan2(zy, zx)
                        power = radius**8
                        theta *= 8
                        phi *= 8
                        zx = power*math.sin(theta)*math.cos(phi) + a
                        zy = power*math.sin(theta)*math.sin(phi) + b
                        zz = power*math.cos(theta) + c
                    v = 255 * min(1.0, (count / 7.0)**2.0)
                data[(z*dim+y)*dim+x] = max(0, min(255, round(v)))
    return bytes(data)


def evolve(volume: bytes, dim: int, alpha: float) -> bytes:
    """3D 6-neighbor discrete heat-kernel relaxation, reflective boundaries."""
    checked_dim(dim)
    if not (0 <= alpha <= 0.5) or not math.isfinite(alpha):
        raise CodecError("Relaxation alpha must be finite, between 0 and 0.5")
    out = bytearray(len(volume))
    for z in range(dim):
        for y in range(dim):
            for x in range(dim):
                i = (z*dim+y)*dim+x
                mean = (
                    volume[(z*dim+y)*dim + max(0,x-1)] +
                    volume[(z*dim+y)*dim + min(dim-1,x+1)] +
                    volume[(z*dim+max(0,y-1))*dim+x] +
                    volume[(z*dim+min(dim-1,y+1))*dim+x] +
                    volume[(max(0,z-1)*dim+y)*dim+x] +
                    volume[(min(dim-1,z+1)*dim+y)*dim+x]
                )/6.0
                out[i] = round((1-alpha)*volume[i] + alpha*mean)
    return bytes(out)


def reconstruction_error(a: bytes, b: bytes) -> tuple[float, float]:
    if len(a) != len(b) or not a:
        raise CodecError("Mismatched volumes")
    absolute, squared = 0, 0
    for x, y in zip(a,b):
        d = x-y
        absolute += abs(d)
        squared += d*d
    return absolute/len(a), squared/len(a)
