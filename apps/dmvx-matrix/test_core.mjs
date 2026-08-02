import test from 'node:test';
import assert from 'node:assert/strict';
import {
  DMVXMatrixRuntime,
  base64ToBytes,
  bytesToBase64,
  crc32,
  fromQ16,
  mulberry32,
  toQ16,
} from './core.mjs';

test('Q16.16 conversion is bounded and reversible within one quantum', () => {
  for (const value of [-1, -0.25, 0, 0.5, 1, 12.75]) {
    assert.ok(Math.abs(fromQ16(toQ16(value)) - value) <= 1 / 65536);
  }
});

test('seeded random stream is deterministic', () => {
  const left = mulberry32(1234);
  const right = mulberry32(1234);
  assert.deepEqual(
    Array.from({ length: 8 }, () => left()),
    Array.from({ length: 8 }, () => right()),
  );
});

test('transaction commits a finite bounded reconstruction', () => {
  const runtime = new DMVXMatrixRuntime({ activeCells: 512, maxActiveCells: 1024 });
  const receipt = runtime.tick({ timeSeconds: 1.25, drive: 0.6 });
  assert.equal(receipt.committed, true);
  assert.equal(receipt.version, 1);
  assert.ok(receipt.reconstructionDistance <= runtime.reconstructionTolerance);
  for (let index = 0; index < runtime.activeCells; index += 1) {
    assert.ok(runtime.committed[index] >= 0 && runtime.committed[index] <= 1);
  }
});

test('invalid candidate rolls back committed state', () => {
  const runtime = new DMVXMatrixRuntime({
    activeCells: 256,
    maxActiveCells: 512,
    reconstructionTolerance: 0,
  });
  const before = runtime.committed.slice();
  const receipt = runtime.tick({ timeSeconds: 4, drive: 1 });
  assert.equal(receipt.committed, false);
  assert.equal(receipt.version, 0);
  assert.deepEqual(runtime.committed, before);
});

test('inward turn reduces the active support without crossing latent dimension', () => {
  const runtime = new DMVXMatrixRuntime({ activeCells: 1024, maxActiveCells: 1024, latentDim: 64 });
  assert.deepEqual(runtime.turnInward(4), { changed: true, activeCells: 256 });
  assert.deepEqual(runtime.turnInward(10), { changed: true, activeCells: 64 });
  assert.deepEqual(runtime.turnInward(10), { changed: false, activeCells: 64 });
});

test('ROM round trip restores committed state and rejects corruption', () => {
  const source = new DMVXMatrixRuntime({ activeCells: 384, maxActiveCells: 1024 });
  source.tick({ timeSeconds: 2, drive: 0.75 });
  const encoded = source.encodeROM();
  const encodedBase64 = bytesToBase64(encoded);

  const target = new DMVXMatrixRuntime({ activeCells: 128, maxActiveCells: 1024 });
  const result = target.decodeROM(base64ToBytes(encodedBase64));
  assert.equal(result.activeCells, source.activeCells);
  assert.equal(result.version, source.version);
  assert.deepEqual(
    Array.from(target.committed.slice(0, source.activeCells)),
    Array.from(source.committed.slice(0, source.activeCells)).map((value) => fromQ16(toQ16(value))),
  );

  const corrupted = encoded.slice();
  corrupted[corrupted.length - 1] ^= 0xff;
  assert.throws(() => target.decodeROM(corrupted), /checksum mismatch/);
  assert.notEqual(crc32(corrupted.subarray(36)), crc32(encoded.subarray(36)));
});
