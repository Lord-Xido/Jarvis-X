#!/usr/bin/env node
'use strict';

// QUINE V3 — verified fixed-point reproduction and independent 3D field march.
// Requirements: Node.js 18+ (built-in node:vm and node:crypto only).
const vm = require('node:vm');
const { createHash } = require('node:crypto');
const assert = require('node:assert/strict');

// This string is a self-reproducing JavaScript expression.
// The expression evaluates to its own exact source bytes.
const quineSource = '(function quine(){return "("+quine.toString()+")()"})()';

function verifyQuine(source, iterations = 1000) {
  if (source !== quineSource) throw new TypeError('Only the bundled quine source is supported');
  assert(Number.isSafeInteger(iterations) && iterations >= 1);
  let current = source;
  for (let i = 0; i < iterations; i++) {
    // Run only our statically defined quine, never untrusted input.
    const next = vm.runInNewContext(current, Object.create(null), { timeout: 1000 });
    assert.equal(next, current, `self-reproduction failed at iteration ${i + 1}`);
    current = next;
  }
  return {
    iterations,
    fixedPoint: true,
    utf8Bytes: Buffer.byteLength(current, 'utf8'),
    sha256: createHash('sha256').update(current, 'utf8').digest('hex'),
    source: current,
  };
}

// F(x,y,z) is an implicit surface, not a mathematically exact SDF.
// Lipschitz bound: ||grad F|| <= 1 + 0.6 = 1.6 where differentiable.
// The bound is suitable for conservative sphere-tracing steps outside the surface.
function field(x, y, z) {
  return Math.hypot(x, y, z) - 0.9 +
    0.2 * Math.sin(3 * x) * Math.sin(3 * y);
}

function march3D(ro, rd, { maxSteps = 128, maxDistance = 10, epsilon = 0.001 } = {}) {
  const values = [ro.x, ro.y, ro.z, rd.x, rd.y, rd.z, maxDistance, epsilon];
  if (!values.every(Number.isFinite)) throw new TypeError('All coordinates and parameters must be finite');
  const norm = Math.hypot(rd.x, rd.y, rd.z);
  if (Math.abs(norm - 1) > 1e-6) throw new RangeError('Ray direction must have unit length');
  if (!Number.isSafeInteger(maxSteps) || maxSteps < 1 || maxDistance <= 0 || epsilon <= 0) {
    throw new RangeError('Invalid march parameters');
  }

  let t = 0;
  for (let i = 0; i < maxSteps; i++) {
    const x = ro.x + rd.x * t;
    const y = ro.y + rd.y * t;
    const z = ro.z + rd.z * t;
    const d = field(x, y, z);
    if (d <= epsilon) return { hit: true, t, steps: i + 1, position: { x, y, z }, fieldValue: d };
    t += d / 1.6;
    if (t > maxDistance) return { hit: false, t: maxDistance, steps: i + 1, reason: 'maxDistance' };
  }
  return { hit: false, t, steps: maxSteps, reason: 'maxSteps' };
}

function main() {
  const quine = verifyQuine(quineSource, 1000);
  const origin = Object.freeze({ x: 0, y: 0, z: 3 });
  const toward = march3D(origin, { x: 0, y: 0, z: -1 });
  const away = march3D(origin, { x: 0, y: 0, z: 1 });
  assert(toward.hit && Math.abs(toward.t - 2.1) < 0.001, 'Expected axial sphere hit near t=2.1');
  assert(!away.hit, 'Expected outward ray to miss');
  assert.deepEqual(origin, { x: 0, y: 0, z: 3 }, 'Origin must not be mutated');
  const result = {
    quine: { iterations: quine.iterations, fixedPoint: quine.fixedPoint, utf8Bytes: quine.utf8Bytes, sha256: quine.sha256 },
    rayMarch: { toward, away, originUnchanged: true },
    note: '1000 successful self-reproductions are not a 1000x performance speedup.',
  };
  console.log(JSON.stringify(result, null, 2));
}

if (require.main === module) main();
module.exports = { quineSource, verifyQuine, field, march3D };
