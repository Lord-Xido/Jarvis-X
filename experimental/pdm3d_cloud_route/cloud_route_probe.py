#!/usr/bin/env python3
"""Public-safe P3D2 protocol cloud probe. Synthetic data only; not the full v5 runtime.

Performs exact 102-byte header / 90-bit address interoperability, loopback TCP
retransmission, SHA-256 reconstruction and optional public HTTPS echo feedback.
"""
import argparse
import base64
import hashlib
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from http.client import HTTPConnection
import json
import os
import secrets
import ssl
import struct
import threading
import time
import urllib.request
from urllib.error import URLError
from uuid import UUID
import zlib

PREFIX = struct.Struct('!4sBBH16s12sIQH32s')
HEADER = struct.Struct('!4sBBH16s12sIQH32sI16s')
AXIS = 1 << 30
PAYLOAD = 1024
assert PREFIX.size == 82 and HEADER.size == 102


def encode90(x, y, z):
    if any(type(v) is not int or not 0 <= v < AXIS for v in (x, y, z)):
        raise ValueError('invalid 30-bit coordinate')
    return ((x << 60) | (y << 30) | z).to_bytes(12, 'big')


def decode90(b):
    if len(b) != 12 or b[0] & 0xfc:
        raise ValueError('invalid 90-bit address')
    a = int.from_bytes(b, 'big')
    return ((a >> 60) & (AXIS-1), (a >> 30) & (AXIS-1), a & (AXIS-1))


def p3d2_key(token):
    if len(token) < 32:
        raise ValueError('minimum token length 32')
    return hashlib.sha256(b'PDM3D-v2-HMAC-context\x00' + token.encode()).digest()


def encode_frame(payload, i, n, whole, digest, transfer, key):
    prefix = PREFIX.pack(b'P3D2', 2, 0, 0, transfer.bytes, encode90(7, 5, i),
                         n, whole, len(payload), digest)
    crc = struct.pack('!I', zlib.crc32(prefix + payload))
    mac = hmac.new(key, prefix + crc + payload, hashlib.sha256).digest()[:16]
    return prefix + crc + mac + payload


def decode_frame(frame, key):
    if not 102 <= len(frame) <= 1126:
        raise ValueError('frame size')
    magic, version, flags, reserved, uid, coord, n, total, ln, digest, crc, mac = HEADER.unpack_from(frame)
    body = frame[102:]
    if (magic, version, flags, reserved) != (b'P3D2', 2, 0, 0):
        raise ValueError('frame version/flags')
    if ln != len(body) or total > 8 * 1024 * 1024 or n != max(1, (total + 1023)//1024):
        raise ValueError('frame lengths')
    if not hmac.compare_digest(mac, hmac.new(key, frame[:86] + body, hashlib.sha256).digest()[:16]):
        raise ValueError('frame auth')
    if crc != zlib.crc32(frame[:82] + body):
        raise ValueError('frame crc')
    x, y, z = decode90(coord)
    if z >= n or len(body) != max(0, min(1024, total-z*1024)):
        raise ValueError('fragment index')
    return UUID(bytes=uid), (x, y, z), n, total, digest, body


def local_roundtrip():
    source = bytes((i * 71 + 13) & 255 for i in range(42000))
    sha = hashlib.sha256(source).digest()
    token = secrets.token_urlsafe(40)
    key = p3d2_key(token)
    transfer = UUID('12345678-1234-5678-1234-567812345678')
    chunks = [source[i:i+1024] for i in range(0, len(source), 1024)]
    wires = [encode_frame(c, i, len(chunks), len(source), sha, transfer, key)
             for i, c in enumerate(chunks)]
    received = {}
    attempts = {}
    events = {'deliberate_drop': 0, 'deliberate_corruption': 0, 'rejected': 0}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            if self.path != '/frame':
                self.send_error(404)
                return
            length = int(self.headers.get('Content-Length', '0'))
            if not 102 <= length <= 1126:
                self.send_error(413)
                return
            raw = self.rfile.read(length)
            try:
                uid, xyz, count, total, digest, body = decode_frame(raw, key)
                if uid != transfer or count != len(chunks) or total != len(source) or digest != sha or xyz[:2] != (7,5):
                    raise ValueError('metadata mismatch')
                received[xyz[2]] = body
            except ValueError:
                events['rejected'] += 1
                self.send_error(422)
                return
            self.send_response(201)
            self.send_header('Content-Length', '0')
            self.end_headers()

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    th = threading.Thread(target=server.serve_forever, daemon=True)
    th.start()
    start = time.monotonic()
    try:
        for _ in range(3):
            for i, wire in enumerate(wires):
                if i in received:
                    continue
                attempts[i] = attempts.get(i, 0) + 1
                if i == 2 and attempts[i] == 1:
                    events['deliberate_drop'] += 1
                    continue
                if i == 1 and attempts[i] == 1:
                    wire = wire[:-1] + bytes([wire[-1] ^ 1])
                    events['deliberate_corruption'] += 1
                connection = HTTPConnection('127.0.0.1', server.server_port, timeout=3)
                try:
                    connection.request('POST', '/frame', wire, {'Content-Type':'application/octet-stream'})
                    res = connection.getresponse()
                    res.read()
                    if res.status not in (201, 422):
                        raise RuntimeError(f'HTTP {res.status}')
                finally:
                    connection.close()
            if len(received) == len(chunks):
                break
        restored = b''.join(received[i] for i in range(len(chunks)))
        assert restored == source and hashlib.sha256(restored).digest() == sha
        assert events == {'deliberate_drop': 1, 'deliberate_corruption': 1, 'rejected': 1}
        assert encode90(7,1,2).hex() == '000000007000000040000002'
        assert decode90(encode90(7,1,2)) == (7,1,2)
        return {'passed':True,'scope':'real TCP localhost on current execution host',
                'source_bytes':len(source),'fragments':len(chunks),
                'frame_header_bytes':102,'total_serialized_bytes':sum(map(len,wires)),
                'http_attempts':sum(attempts.values())-events['deliberate_drop'],
                'retry_fragments':sum(v-1 for v in attempts.values()),
                'recovery':events,'elapsed_ms':round((time.monotonic()-start)*1000,2),
                'sha256':sha.hex()}
    finally:
        server.shutdown()
        server.server_close()
        th.join(timeout=5)


def wan_probe():
    # Only synthetic bytes: no application data, secrets or raw tenant packets.
    payload = base64.b64encode(b'PDM3D synthetic protocol WAN probe').decode('ascii')
    message = json.dumps({'pdm3d_test':payload,'data_type':'synthetic','version':'P3D2'}).encode()
    result = {'status':'not_verified', 'scheme':'https', 'bytes_sent':len(message)}
    for endpoint in ('https://httpbingo.org/post', 'https://postman-echo.com/post'):
        req = urllib.request.Request(endpoint, data=message, headers={'Content-Type':'application/json'}, method='POST')
        try:
            with urllib.request.urlopen(req, timeout=8, context=ssl.create_default_context()) as response:
                if response.status != 200:
                    continue
                raw = response.read(100000)
            echoed = json.loads(raw)
            candidate = echoed.get('json')
            if candidate is None and isinstance(echoed.get('data'), str):
                candidate = json.loads(echoed['data'])
            if isinstance(candidate, dict) and candidate.get('pdm3d_test') == payload:
                return {'status':'verified_https_echo','scheme':'https','endpoint':endpoint,
                        'bytes_sent':len(message),'echo_integrity':True}
        except (OSError, ValueError, URLError):
            continue
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wan-probe', action='store_true')
    parser.add_argument('--report', default='pdm3d-cloud-feedback.json')
    args = parser.parse_args()
    report = {'name':'PDM3D synthetic cloud route compatibility probe',
              'full_cloudbitfabric_v5_executed':False,
              'public_receiver_deployed':False,
              'runner':os.getenv('RUNNER_NAME','local'),
              'github_sha':os.getenv('GITHUB_SHA',''),
              'loopback':local_roundtrip(),
              'wan':wan_probe() if args.wan_probe else {'status':'not_attempted'}}
    with open(args.report,'w',encoding='utf-8') as fh:
        json.dump(report,fh,indent=2,sort_keys=True)
        fh.write('\n')
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
