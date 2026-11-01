import test from 'node:test';
import assert from 'node:assert/strict';
import { encodeRGBA, decodeRGBA, equalBytes, bitErrorCount, medianMs, throughputMetrics } from './codec.mjs';

function sample(n) { return Uint8Array.from({length:n}, (_, i) => (i * 73 + 9) & 255); }

test('empty input uses one zero-padded row', () => {
  const e = encodeRGBA(new Uint8Array(0));
  assert.deepEqual([e.width, e.height, e.data.length], [256, 1, 1024]);
  assert.equal(decodeRGBA(e.data, e.width, e.height, 0).length, 0);
});

test('1-byte and partial-row payloads are reconstructed exactly', () => {
  for (const n of [1, 3, 4, 5, 1023, 1024, 1025, 8197]) {
    const bytes = sample(n);
    const e = encodeRGBA(bytes);
    assert.equal(e.data.length % 1024, 0);
    assert.equal(equalBytes(bytes, decodeRGBA(e.data, e.width, e.height, n)), true);
    assert.equal(bitErrorCount(bytes, decodeRGBA(e.data, e.width, e.height, n)), 0);
  }
});

test('vertical GPU readback orientation can be corrected', () => {
  const original = sample(2053);
  const e = encodeRGBA(original);
  const reversed = new Uint8Array(e.data.length);
  for (let y = 0; y < e.height; y++) {
    reversed.set(e.data.subarray(y * 1024, (y + 1) * 1024), (e.height - y - 1) * 1024);
  }
  assert.equal(equalBytes(original, decodeRGBA(reversed, 256, e.height, original.length)), false);
  assert.equal(equalBytes(original, decodeRGBA(reversed, 256, e.height, original.length, true)), true);
});

test('reference works for arbitrary valid texture widths', () => {
  const bytes = sample(94);
  const e = encodeRGBA(bytes, 7);
  assert.equal(e.height, 4);
  assert.equal(equalBytes(bytes, decodeRGBA(e.data, 7, 4, 94)), true);
});

test('byte equality and differing-bit counts fail closed', () => {
  const a = Uint8Array.of(0, 255, 0xaa);
  const b = Uint8Array.of(1, 254, 0x55);
  assert.equal(equalBytes(a, b), false);
  assert.equal(bitErrorCount(a, b), 10);
  assert.equal(equalBytes(a, Uint8Array.of(0)), false);
  assert.throws(() => bitErrorCount(a, new Uint8Array(0)), RangeError);
});

test('median timing excludes outlier influence', () => {
  assert.equal(medianMs([10, 2, 3, 100, 4]), 4);
  assert.equal(medianMs([4, 2]), 3);
  assert.throws(() => medianMs([]), RangeError);
  assert.throws(() => medianMs([0]), RangeError);
});

test('throughput units distinguish byte/bit rates and GPU fragments', () => {
  const m = throughputMetrics(1048576, 256, 1000, 1000);
  assert.equal(m.height, 1024);
  assert.equal(m.inputGBps, 1.048576);
  assert.equal(m.inputGbps, 8.388608);
  assert.equal(m.roundTripsPerSecond, 1000);
  assert.equal(m.fragmentSamplesPerSecond, 262144000);
  assert.equal(m.uploadReadbackGBpsLowerBound, 2.097152);
  assert.equal(m.visualPointAttributeBytesIfAllBytesRendered, 25165824);
});

test('padded payload accounts for physical texture traffic', () => {
  const m = throughputMetrics(1, 256, 1, 1000);
  assert.equal(m.allocatedBytes, 1024);
  assert.equal(m.uploadReadbackGBpsLowerBound, 2048 / 1e9);
  assert.equal(m.inputGBps, 1e-9);
});

test('invalid dimensions and intervals are rejected', () => {
  assert.throws(() => encodeRGBA([1, 2, 3]), TypeError);
  assert.throws(() => encodeRGBA(sample(2), 0), RangeError);
  assert.throws(() => decodeRGBA(new Uint8Array(2), 1, 1, 2), RangeError);
  assert.throws(() => decodeRGBA(new Uint8Array(4), 1, 1, 5), RangeError);
  assert.throws(() => throughputMetrics(1024, 256, 0, 1), RangeError);
  assert.throws(() => throughputMetrics(1024, 256, 1, 0), RangeError);
});
