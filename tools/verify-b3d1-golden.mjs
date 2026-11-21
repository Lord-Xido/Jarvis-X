// Independent Node.js B3D1 oracle: no Python, native addons or network required.
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { resolve, dirname } from 'node:path';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const fixture = JSON.parse(readFileSync(resolve(root, 'native/bitcube3d/tests/fixtures/b3d1-v1.json'), 'utf8'));
const SIDE = 1_000_000_000;
const BRICK_BYTES = 512;

function pass(brick, axis, stride) {
  const delta = [1, 16, 256][axis] * stride;
  for (let z = 0; z < 16; z++) {
    for (let y = 0; y < 16; y++) {
      for (let x = 0; x < 16; x++) {
        if (([x, y, z][axis] & stride) !== 0) continue;
        const lo = x + 16 * y + 256 * z;
        if ((brick[lo >>> 3] & (1 << (lo & 7))) !== 0) {
          const hi = lo + delta;
          brick[hi >>> 3] ^= 1 << (hi & 7);
        }
      }
    }
  }
}

function encode(points, rounds) {
  if (!Number.isInteger(rounds) || rounds < 0 || rounds > 32) throw new Error('invalid rounds');
  const bricks = new Map();
  for (const point of points) {
    if (!Array.isArray(point) || point.length !== 3 ||
        point.some(v => !Number.isInteger(v) || v < 0 || v >= SIDE)) {
      throw new Error('invalid coordinate');
    }
    const key = point.map(v => Math.floor(v / 16));
    const id = key.join(',');
    if (!bricks.has(id)) bricks.set(id, {key, bytes: Buffer.alloc(BRICK_BYTES)});
    const b = bricks.get(id).bytes;
    const bit = point[0] % 16 + 16 * (point[1] % 16) + 256 * (point[2] % 16);
    b[bit >>> 3] |= 1 << (bit & 7);
  }
  for (const {bytes} of bricks.values()) {
    for (let n = 0; n < rounds; n++) {
      for (let axis = 0; axis < 3; axis++) {
        for (const stride of [1, 2, 4, 8]) pass(bytes, axis, stride);
      }
    }
  }
  const blocks = [...bricks.values()].sort((a, b) => {
    for (let i = 0; i < 3; i++) if (a.key[i] !== b.key[i]) return a.key[i] - b.key[i];
    return 0;
  });
  const out = Buffer.alloc(20 + blocks.length * (12 + BRICK_BYTES));
  out.write('B3D1', 0, 'ascii');
  out.writeUInt32LE(SIDE, 4);
  out.writeUInt32LE(rounds, 8);
  out.writeBigUInt64LE(BigInt(blocks.length), 12);
  let offset = 20;
  for (const {key, bytes} of blocks) {
    for (let j = 0; j < 3; j++) out.writeUInt32LE(key[j], offset + 4 * j);
    bytes.copy(out, offset + 12);
    offset += 12 + BRICK_BYTES;
  }
  return out;
}

function validate(payload) {
  if (payload.length < 20 || payload.toString('ascii', 0, 4) !== 'B3D1' ||
      payload.readUInt32LE(4) !== SIDE) throw new Error('invalid header');
  const count = payload.readBigUInt64LE(12);
  if (count > 100_000n || payload.length !== 20 + Number(count) * 524) {
    throw new Error('invalid payload length or brick budget');
  }
  let prev = null;
  for (let i = 0; i < Number(count); i++) {
    const at = 20 + i * 524;
    const coord = [payload.readUInt32LE(at), payload.readUInt32LE(at + 4),
      payload.readUInt32LE(at + 8)];
    if (coord.some(v => v >= SIDE / 16)) throw new Error('out-of-range brick');
    if (prev && (coord[0] < prev[0] ||
        (coord[0] === prev[0] && coord[1] < prev[1]) ||
        (coord[0] === prev[0] && coord[1] === prev[1] && coord[2] <= prev[2]))) {
      throw new Error('noncanonical brick ordering');
    }
    if (!payload.subarray(at + 12, at + 524).some(x => x !== 0)) throw new Error('empty brick');
    prev = coord;
  }
}

for (const vector of fixture.vectors) {
  const bytes = encode(vector.bits, vector.rounds);
  validate(bytes);
  const digest = createHash('sha256').update(bytes).digest('hex');
  if (bytes.length !== vector.expected_bytes || digest !== vector.sha256 ||
      bytes.readBigUInt64LE(12) !== BigInt(vector.expected_bricks)) {
    throw new Error('B3D1 golden mismatch: ' + vector.name + ' actual=' + digest);
  }
  console.log('B3D1 JS golden ' + vector.name + ': PASS ' + digest);
}
if (process.argv.length >= 4 && process.argv[2] === '--verify-file') {
  const actual = readFileSync(process.argv[3]);
  const which = fixture.vectors.find(v => v.name === process.argv[4]);
  if (!which) throw new Error('unknown fixture');
  validate(actual);
  const actualHash = createHash('sha256').update(actual).digest('hex');
  if (actualHash !== which.sha256) throw new Error('external B3D1 file mismatch: ' + actualHash);
}
