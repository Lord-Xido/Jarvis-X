#!/usr/bin/env node
'use strict';
/*
 * Dr Moagi React Artifact (10): verifiable sparse 3D byte/bit reference bridge.
 * No learned ANN, compression claim, autonomous source changes or sandboxing.
 * Sparse virtual coordinates are bounded: this module uses only input-proportional RAM.
 * Node.js 18+; no third-party dependencies.
 */
const crypto = require('node:crypto');
const vm = require('node:vm');

const SCHEMA = 'dr-moagi/quine-bitwise3d/v1';
const MAX_BYTES = 1024 * 1024;
const CLASSIC_TEMPLATE = 'const quine = () => { const q = %Q%; return q.replace(\'%Q%\', JSON.stringify(q)) }';

function requireSafeInt(n, name, min = 0, max = Number.MAX_SAFE_INTEGER) {
  if (!Number.isSafeInteger(n) || n < min || n > max) {
    throw new RangeError(`${name} must be a safe integer in [${min}, ${max}]`);
  }
}
function asBytes(value, name = 'bytes') {
  if (!Buffer.isBuffer(value) && !(value instanceof Uint8Array)) {
    throw new TypeError(`${name} must be a Buffer or Uint8Array (not a string)`);
  }
  return Buffer.from(value);
}
function sha256(bytes) {
  return crypto.createHash('sha256').update(bytes).digest('hex');
}

function indexToVoxel(index, width = 16, height = 16) {
  requireSafeInt(index, 'index');
  requireSafeInt(width, 'width', 1, 65536);
  requireSafeInt(height, 'height', 1, 65536);
  return {
    x: index % width,
    y: Math.floor(index / width) % height,
    z: Math.floor(index / (width * height)),
  };
}
function voxelToIndex(x, y, z, width = 16, height = 16) {
  requireSafeInt(width, 'width', 1, 65536);
  requireSafeInt(height, 'height', 1, 65536);
  requireSafeInt(x, 'x', 0, width - 1);
  requireSafeInt(y, 'y', 0, height - 1);
  requireSafeInt(z, 'z', 0);
  const index = x + width * (y + height * z);
  requireSafeInt(index, 'index');
  return index;
}
function encodeBytes3D(value, { width = 16, height = 16, maxBytes = MAX_BYTES } = {}) {
  const bytes = asBytes(value);
  requireSafeInt(maxBytes, 'maxBytes', 0);
  requireSafeInt(width, 'width', 1, 65536);
  requireSafeInt(height, 'height', 1, 65536);
  if (bytes.length > maxBytes) throw new RangeError('Source exceeds bounded byte ceiling');
  // Omit zeros; the byteLength and explicit spatial index retain their positions.
  const voxels = [];
  for (let i = 0; i < bytes.length; i++) {
    if (bytes[i] !== 0) {
      const { x, y, z } = indexToVoxel(i, width, height);
      voxels.push([x, y, z, bytes[i]]);
    }
  }
  return { schema: SCHEMA, width, height, byteLength: bytes.length, sha256: sha256(bytes), voxels };
}
function decodeBytes3D(archive, { maxBytes = MAX_BYTES } = {}) {
  requireSafeInt(maxBytes, 'maxBytes', 0);
  if (!archive || typeof archive !== 'object' || archive.schema !== SCHEMA) {
    throw new TypeError('Unrecognized 3D archive schema');
  }
  const { width, height, byteLength, voxels, sha256: digest } = archive;
  requireSafeInt(width, 'width', 1, 65536);
  requireSafeInt(height, 'height', 1, 65536);
  requireSafeInt(byteLength, 'byteLength', 0, maxBytes);
  if (!Array.isArray(voxels) || voxels.length > byteLength) throw new TypeError('Invalid voxel table');
  if (typeof digest !== 'string' || !/^[0-9a-f]{64}$/.test(digest)) throw new TypeError('Invalid digest');
  const result = Buffer.alloc(byteLength);
  const seen = new Set();
  for (const item of voxels) {
    if (!Array.isArray(item) || item.length !== 4) throw new TypeError('Invalid voxel record');
    const [x, y, z, bitByte] = item;
    requireSafeInt(bitByte, 'voxel byte', 1, 255);
    const index = voxelToIndex(x, y, z, width, height);
    if (index >= byteLength || seen.has(index)) throw new RangeError('Out-of-range or duplicate voxel');
    seen.add(index);
    result[index] = bitByte;
  }
  if (sha256(result) !== digest) throw new Error('SHA-256 integrity verification failed');
  return result;
}

const POPCOUNT8 = Uint8Array.from({ length: 256 }, (_, n) => {
  let count = 0;
  for (let x = n; x; x &= x - 1) count++;
  return count;
});
function exactBitResidual(a, b) {
  const left = asBytes(a, 'left');
  const right = asBytes(b, 'right');
  if (left.length !== right.length) {
    return { equal: false, differingBits: null, lengthMismatch: true, leftBytes: left.length, rightBytes: right.length };
  }
  let differingBits = 0;
  for (let i = 0; i < left.length; i++) differingBits += POPCOUNT8[left[i] ^ right[i]];
  return { equal: differingBits === 0, differingBits, lengthMismatch: false, leftBytes: left.length, rightBytes: right.length };
}

function fixedPointCycle(source, candidate, options) {
  const authoritative = asBytes(source, 'source');
  const proposed = asBytes(candidate, 'candidate');
  // No state mutation before both integrity and exact source-equivalence gates.
  const archive = encodeBytes3D(proposed, options);
  const restored = decodeBytes3D(archive, options);
  const reconstruction = exactBitResidual(proposed, restored);
  const fixedPoint = exactBitResidual(authoritative, restored);
  const committed = reconstruction.equal && fixedPoint.equal;
  return {
    source: committed ? restored : Buffer.from(authoritative),
    receipt: {
      committed,
      reconstruction,
      fixedPoint,
      baselineSha256: sha256(authoritative),
      candidateSha256: archive.sha256,
      resultSha256: sha256(committed ? restored : authoritative),
    },
  };
}
function runBoundedLoop(source, { cycles = 1, propose = (value) => value, ...codecOptions } = {}) {
  requireSafeInt(cycles, 'cycles', 1, 1000);
  if (typeof propose !== 'function') throw new TypeError('propose must be a function');
  let current = asBytes(source);
  const receipts = [];
  for (let i = 0; i < cycles; i++) {
    const candidate = propose(Buffer.from(current), i);
    const { source: next, receipt } = fixedPointCycle(current, candidate, codecOptions);
    current = next;
    receipts.push({ cycle: i + 1, ...receipt });
  }
  return { source: current, receipts, committed: receipts.filter(r => r.committed).length, rejected: receipts.filter(r => !r.committed).length };
}

function classicalQuineSource() {
  return CLASSIC_TEMPLATE.replace('%Q%', JSON.stringify(CLASSIC_TEMPLATE));
}
function verifyBundledClassicQuine(iterations = 1000) {
  requireSafeInt(iterations, 'iterations', 1, 1000);
  const original = classicalQuineSource();
  let current = original;
  for (let i = 0; i < iterations; i++) {
    // Execute only the bundled, statically defined self-reproducer.
    // node:vm is NOT a security sandbox for untrusted code.
    const output = vm.runInNewContext(`${current}; quine()`, Object.create(null), { timeout: 1000 });
    if (output !== original) throw new Error(`Quine source mismatch at cycle ${i + 1}`);
    current = output;
  }
  return { source: current, iterations, bytes: Buffer.byteLength(current, 'utf8'), sha256: sha256(Buffer.from(current)) };
}

module.exports = {
  SCHEMA, MAX_BYTES, indexToVoxel, voxelToIndex, encodeBytes3D, decodeBytes3D,
  exactBitResidual, fixedPointCycle, runBoundedLoop,
  classicalQuineSource, verifyBundledClassicQuine, sha256,
};

if (require.main === module) {
  const summary = verifyBundledClassicQuine(1000);
  const loop = runBoundedLoop(Buffer.from(summary.source, 'utf8'), { cycles: 3 });
  console.log(JSON.stringify({ quine: { bytes: summary.bytes, sha256: summary.sha256, cycles: summary.iterations }, loop: { committed: loop.committed, rejected: loop.rejected, sourceMatches: loop.source.toString('utf8') === summary.source } }, null, 2));
}
