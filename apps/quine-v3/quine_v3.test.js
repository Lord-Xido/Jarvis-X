'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const { createHash } = require('node:crypto');
const { quineSource, verifyQuine, field, march3D } = require('./quine_v3_verified');

test('quine is byte-for-byte identical after 1,000 executions', () => {
  const result = verifyQuine(quineSource, 1000);
  assert.equal(result.fixedPoint, true);
  assert.equal(result.iterations, 1000);
  assert.equal(result.source, quineSource);
  assert.equal(result.utf8Bytes, 55);
  assert.equal(result.sha256, 'd60ef0606a6e33dc604fd84fb8c73f79dfdd697ae62114ee57f047e8ce3d1619');
  assert.equal(result.sha256, createHash('sha256').update(result.source, 'utf8').digest('hex'));
});

test('quine iteration budget must be a positive safe integer', () => {
  for (const iterations of [-1, 0, 0.5, Number.MAX_SAFE_INTEGER + 1]) {
    assert.throws(() => verifyQuine(quineSource, iterations), assert.AssertionError);
  }
});

test('field preserves known axial values', () => {
  assert.equal(field(0, 0, 0), -0.9);
  assert.equal(field(0, 0, 3), 2.1);
  assert.ok(Math.abs(field(0, 0, 0.9)) < 1e-12);
});

test('inward ray hits expected axial surface without origin mutation', () => {
  const ro = Object.freeze({ x: 0, y: 0, z: 3 });
  const rd = Object.freeze({ x: 0, y: 0, z: -1 });
  const result = march3D(ro, rd);
  assert.equal(result.hit, true);
  assert.ok(Math.abs(result.t - 2.1) <= 0.001);
  assert.ok(result.fieldValue <= 0.001 && result.fieldValue >= 0);
  assert.equal(result.position.z, 3 - result.t);
  assert.deepEqual(ro, { x: 0, y: 0, z: 3 });
  assert.deepEqual(rd, { x: 0, y: 0, z: -1 });
});

test('outward and offset rays miss within bounded distance', () => {
  const outward = march3D({ x: 0, y: 0, z: 3 }, { x: 0, y: 0, z: 1 });
  assert.equal(outward.hit, false);
  assert.equal(outward.reason, 'maxDistance');
  const offset = march3D({ x: 3, y: 0, z: 3 }, { x: 0, y: 0, z: -1 });
  assert.equal(offset.hit, false);
});

test('iteration budget is bounded and reported', () => {
  const result = march3D({ x: 0, y: 0, z: 3 }, { x: 0, y: 0, z: -1 }, { maxSteps: 1 });
  assert.equal(result.hit, false);
  assert.equal(result.reason, 'maxSteps');
  assert.equal(result.steps, 1);
});

test('non-finite inputs, non-unit rays, and invalid budgets are rejected', () => {
  const ro = { x: 0, y: 0, z: 3 };
  const rd = { x: 0, y: 0, z: -1 };
  assert.throws(() => march3D({ ...ro, x: Infinity }, rd), TypeError);
  assert.throws(() => march3D(ro, { x: 0, y: 0, z: -2 }), RangeError);
  assert.throws(() => march3D(ro, rd, { maxSteps: 0 }), RangeError);
  assert.throws(() => march3D(ro, rd, { maxDistance: 0 }), RangeError);
  assert.throws(() => march3D(ro, rd, { epsilon: -1 }), RangeError);
});

test('original V3 transformation is not a fixed-point quine', () => {
  const E = s => s.replace(/\s+/g, ' ').trim();
  const D = c => c.split('').map(x => x.charCodeAt(0) ^ 42);
  const R = d => String.fromCharCode(...d.map(n => n ^ 42));
  const sample = 'function f() { return 1; }';
  const transformed = E(R(D(sample))) + '\n// appended function';
  assert.notEqual(transformed, sample);
  assert.equal(R(D(sample)), sample);
});

test('quine verifier rejects untrusted executable source', () => {
  assert.throws(() => verifyQuine('process.exit(1)', 1), TypeError);
});
