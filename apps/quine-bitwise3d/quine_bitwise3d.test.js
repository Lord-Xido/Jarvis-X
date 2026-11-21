'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const {
  indexToVoxel, voxelToIndex, encodeBytes3D, decodeBytes3D,
  exactBitResidual, fixedPointCycle, runBoundedLoop,
  classicalQuineSource, verifyBundledClassicQuine,
} = require('./quine_bitwise3d');

test('geometry is a bijection over declared virtual byte indices', () => {
  for (const [width, height] of [[1, 1], [4, 8], [16, 16], [64, 64]]) {
    for (let i = 0; i < 10000; i++) {
      const c = indexToVoxel(i, width, height);
      assert.equal(voxelToIndex(c.x, c.y, c.z, width, height), i);
    }
  }
});
test('exact roundtrip: empty, all zero, bit patterns, unicode and deterministic random data', () => {
  const cases = [Buffer.alloc(0), Buffer.alloc(128), Buffer.from([0, 1, 2, 254, 255]), Buffer.from('Moagi Ψ* — 3D 🚀', 'utf8'), Buffer.from({ length: 4096 }, (_, i) => (i * 73 + 29) & 255)];
  for (const source of cases) {
    const archive = encodeBytes3D(source, { width: 13, height: 7 });
    const serializable = JSON.parse(JSON.stringify(archive));
    assert.deepEqual(decodeBytes3D(serializable), source);
    assert.deepEqual(exactBitResidual(source, decodeBytes3D(serializable)), {
      equal: true, differingBits: 0, lengthMismatch: false, leftBytes: source.length, rightBytes: source.length,
    });
  }
});
test('one flipped byte bit produces popcount exactly 1', () => {
  assert.equal(exactBitResidual(Buffer.from([3, 4]), Buffer.from([3, 5])).differingBits, 1);
});
test('unequal byte lengths cannot qualify as an exact fixed point', () => {
  assert.deepEqual(exactBitResidual(Buffer.from('a'), Buffer.from('ab')).lengthMismatch, true);
});
test('corrupted payload fails closed by SHA-256, never silently reconstructs', () => {
  const archive = encodeBytes3D(Buffer.from('hello world'));
  archive.voxels[0][3] ^= 1;
  assert.throws(() => decodeBytes3D(archive), /SHA-256/);
});
test('duplicate voxel and out-of-bounds voxel cannot overwrite another byte', () => {
  const archive = encodeBytes3D(Buffer.from([97, 0, 99]));
  archive.voxels.push([...archive.voxels[0]]);
  assert.throws(() => decodeBytes3D(archive), /duplicate/);
  const outOfBounds = encodeBytes3D(Buffer.from('abc'));
  outOfBounds.voxels[0][2] = 1;
  assert.throws(() => decodeBytes3D(outOfBounds), /Out-of-range/);
});
test('integer and bounded allocation validation rejects malformed records', () => {
  assert.throws(() => indexToVoxel(-1), /safe integer/);
  assert.throws(() => voxelToIndex(16, 0, 0), /safe integer/);
  assert.throws(() => encodeBytes3D(Buffer.alloc(32), { maxBytes: 31 }), /ceiling/);
  assert.throws(() => decodeBytes3D({ schema: 'other' }), /schema/);
  assert.throws(() => encodeBytes3D('string source'), /Buffer/);
});
test('bounded cycle commits identical candidate and rolls back bit-changing candidate', () => {
  const source = Buffer.from('Dr Moagi');
  const good = fixedPointCycle(source, Buffer.from(source));
  assert.equal(good.receipt.committed, true);
  assert.equal(good.receipt.fixedPoint.differingBits, 0);
  const changed = Buffer.from(source);
  changed[0] ^= 1;
  const bad = fixedPointCycle(source, changed);
  assert.equal(bad.receipt.committed, false);
  assert.equal(bad.receipt.fixedPoint.differingBits, 1);
  assert.deepEqual(bad.source, source);
  assert.deepEqual(source, Buffer.from('Dr Moagi'));
});
test('bounded inward loop retains source under alternating mutation attempts', () => {
  const source = Buffer.from('genesis');
  const loop = runBoundedLoop(source, { cycles: 10, propose: (s, i) => { if (i % 2 === 1) s[0] ^= 1; return s; } });
  assert.equal(loop.committed, 5);
  assert.equal(loop.rejected, 5);
  assert.deepEqual(loop.source, source);
});
test('classic quine is exact across 1,000 source reproduction iterations', () => {
  const check = verifyBundledClassicQuine(1000);
  assert.equal(check.source, classicalQuineSource());
  assert.equal(check.bytes, 161);
  assert.match(check.sha256, /^[a-f0-9]{64}$/);
});
test('Float32 modulo folding cannot be treated as a reversible source codec', () => {
  const encode = (s) => {
    const z = new Float32Array(512);
    for (let i = 0; i < s.length; i++) z[i % 512] += s.charCodeAt(i) * 0.0007;
    return Buffer.from(z.buffer);
  };
  const a = 'A' + '\0'.repeat(511) + 'B';
  const b = 'B' + '\0'.repeat(511) + 'A';
  assert.notEqual(a, b);
  assert.deepEqual(encode(a), encode(b));
});
