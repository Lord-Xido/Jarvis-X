#!/usr/bin/env node
'use strict';
/**
 * Dr Moagi sparse 1000^3 inward octree (research reference).
 * Virtual logical cells: 1_000_000_000, physically materialized: occupied paths only.
 * Each dyadic fold is a SHA-256 Merkle summary, NOT a reversible compression by itself.
 * Original byte leaves remain authoritative for exact inverse reconstruction.
 * Inward candidate calculations are shadow overlays; rejected candidates never mutate state.
 * JavaScript/Node 18+, built-in crypto. Not a trained ANN or 1e9-fold speedup.
 */
const crypto = require('node:crypto');
const assert = require('node:assert/strict');
const { exactBitResidual, classicalQuineSource, verifyBundledClassicQuine } = require('./quine_bitwise3d');

const SIDE = 1000;
const VIRTUAL_CELLS = SIDE ** 3;
const SOURCE_LIMIT = 1024 * 1024;
const DIMS = Object.freeze((() => {
  const d = [SIDE];
  while (d[d.length - 1] > 1) d.push(Math.ceil(d[d.length - 1] / 2));
  return d;
})());
const EMPTY = null;
const hash = parts => {
  const h = crypto.createHash('sha256');
  for (const p of parts) h.update(p);
  return h.digest('hex');
};
const positive = (v, name, max) => {
  if (!Number.isSafeInteger(v) || v < 0 || v > max) throw new RangeError(`${name} must be an integer in [0, ${max}]`);
  return v;
};
function coord(i, d = SIDE) {
  positive(i, 'index', d ** 3 - 1);
  return [i % d, Math.floor(i / d) % d, Math.floor(i / (d * d))];
}
function linear(x, y, z, d = SIDE) {
  positive(x, 'x', d - 1); positive(y, 'y', d - 1); positive(z, 'z', d - 1);
  return x + d * (y + d * z);
}
function parentKey(childIndex, childSide, parentSide) {
  const [x, y, z] = coord(childIndex, childSide);
  return linear(Math.floor(x / 2), Math.floor(y / 2), Math.floor(z / 2), parentSide);
}
function leafHash(index, byte) {
  positive(byte, 'byte', 255);
  if (!byte) return EMPTY;
  const h = crypto.createHash('sha256');
  h.update('DM3D-LEAF-v1:');
  const binary = Buffer.allocUnsafe(5);
  binary.writeUInt32LE(index, 0);
  binary[4] = byte;
  h.update(binary);
  return h.digest('hex');
}
function nodeHash(level, index, childHashes) {
  if (childHashes.every(x => x === EMPTY)) return EMPTY;
  const h = crypto.createHash('sha256');
  h.update(`DM3D-NODE-v1:${level}:${index}:`);
  for (let slot = 0; slot < 8; slot++) {
    h.update(Buffer.from([slot]));
    h.update(childHashes[slot] ? Buffer.from(childHashes[slot], 'hex') : Buffer.alloc(32));
  }
  return h.digest('hex');
}
function rootSeal(rootHash, byteLength) {
  const payload = Buffer.alloc(4);
  payload.writeUInt32LE(byteLength);
  return hash(['DM3D-ROOT-v1:', payload, rootHash ? Buffer.from(rootHash, 'hex') : Buffer.alloc(32)]);
}

class SparseInward1000 {
  constructor(source, { maxBytes = SOURCE_LIMIT } = {}) {
    if (!Buffer.isBuffer(source) && !(source instanceof Uint8Array)) throw new TypeError('source must be bytes');
    this.source = Buffer.from(source);
    this.maxBytes = positive(maxBytes, 'maxBytes', VIRTUAL_CELLS);
    if (this.source.length > this.maxBytes || this.source.length > VIRTUAL_CELLS) throw new RangeError('source exceeds allocated byte ceiling');
    this.levels = DIMS.map(() => new Map());
    this.leaves = new Map();
    for (let i = 0; i < this.source.length; i++) {
      if (this.source[i] !== 0) {
        this.leaves.set(i, this.source[i]);
        this.levels[0].set(i, leafHash(i, this.source[i]));
      }
    }
    // Bulk-build shared inward parents; no dense 1e9-cell allocation.
    for (let lev = 1; lev < DIMS.length; lev++) {
      const parents = new Map();
      for (const [childIndex, digest] of this.levels[lev - 1]) {
        const [x, y, z] = coord(childIndex, DIMS[lev - 1]);
        const p = linear(x >> 1, y >> 1, z >> 1, DIMS[lev]);
        let slots = parents.get(p);
        if (!slots) { slots = Array(8).fill(EMPTY); parents.set(p, slots); }
        slots[(x & 1) | ((y & 1) << 1) | ((z & 1) << 2)] = digest;
      }
      const layer = this.levels[lev];
      for (const [key, slots] of parents) layer.set(key, nodeHash(lev, key, slots));
    }
    this.root = rootSeal(this.levels[this.levels.length - 1].get(0) || EMPTY, this.source.length);
    this.originalSha256 = crypto.createHash('sha256').update(this.source).digest('hex');
  }
  metadata() {
    return {
      virtualSide: SIDE,
      virtualCells: VIRTUAL_CELLS,
      logicalBitsAtOneBitPerCell: VIRTUAL_CELLS,
      inwardSides: [...DIMS],
      levels: DIMS.length,
      inwardFolds: DIMS.length - 1,
      sourceBytes: this.source.length,
      allocatedHashesByLevel: this.levels.map(m => m.size),
      nonzeroLeaves: this.levels[0].size,
      rootSha256: this.root,
      sourceSha256: this.originalSha256,
    };
  }
  reconstruct() {
    const result = Buffer.alloc(this.source.length);
    for (const [i, value] of this.leaves) {
      // Reconstruct from the stored 3D leaf payload, not from the authoritative buffer.
      result[i] = value;
    }
    return result;
  }
  // Complete independent root and byte revalidation. O(K*levels + N).
  deepAudit() {
    const recovered = this.reconstruct();
    const bitwise = exactBitResidual(this.source, recovered);
    const rebuilt = new SparseInward1000(recovered, { maxBytes: this.maxBytes });
    const rootEqual = rebuilt.root === this.root;
    const originalDigestMatches = rebuilt.originalSha256 === this.originalSha256;
    const hashesEqual = rebuilt.levels.every((layer, l) => {
      const current = this.levels[l];
      if (layer.size !== current.size) return false;
      for (const [key, digest] of layer) if (current.get(key) !== digest) return false;
      return true;
    });
    return { pass: bitwise.equal && rootEqual && originalDigestMatches && hashesEqual,
      differingBits: bitwise.differingBits, rootEqual, originalDigestMatches, hashesEqual };
  }
  // Shadow candidate. Only affected octree ancestors get recomputed.
  shadow(patches = []) {
    if (!Array.isArray(patches) || patches.length > 1024) throw new RangeError('patches must be an array of <=1024 edits');
    const unique = new Map();
    for (const p of patches) {
      if (!Array.isArray(p) || p.length !== 2) throw new TypeError('patch must be [byteIndex, byteValue]');
      const [index, value] = p;
      positive(index, 'byteIndex', this.source.length - 1);
      positive(value, 'byteValue', 255);
      if (unique.has(index)) throw new RangeError('duplicate patch index');
      unique.set(index, value);
    }
    let differingBits = 0;
    const delta = DIMS.map(() => new Map());
    let dirt = new Set();
    for (const [index, value] of unique) {
      const old = this.source[index];
      const v = old ^ value;
      for (let n = v; n; n &= n - 1) differingBits++;
      const newHash = leafHash(index, value);
      delta[0].set(index, newHash);
      dirt.add(index);
    }
    for (let lev = 1; lev < DIMS.length; lev++) {
      const parentDirt = new Set();
      for (const childIndex of dirt) {
        parentDirt.add(parentKey(childIndex, DIMS[lev - 1], DIMS[lev]));
      }
      for (const key of parentDirt) {
        const [px, py, pz] = coord(key, DIMS[lev]);
        const slots = Array(8).fill(EMPTY);
        for (let slot = 0; slot < 8; slot++) {
          const cx = 2 * px + (slot & 1);
          const cy = 2 * py + ((slot >> 1) & 1);
          const cz = 2 * pz + ((slot >> 2) & 1);
          if (cx >= DIMS[lev - 1] || cy >= DIMS[lev - 1] || cz >= DIMS[lev - 1]) continue;
          const childKey = linear(cx, cy, cz, DIMS[lev - 1]);
          slots[slot] = delta[lev - 1].has(childKey)
            ? delta[lev - 1].get(childKey)
            : (this.levels[lev - 1].get(childKey) || EMPTY);
        }
        delta[lev].set(key, nodeHash(lev, key, slots));
      }
      dirt = parentDirt;
    }
    const candidateRoot = rootSeal(delta.at(-1).has(0) ? delta.at(-1).get(0) : (this.levels.at(-1).get(0) || EMPTY), this.source.length);
    return { differingBits, candidateRoot, touchedNodes: delta.reduce((sum, m) => sum + m.size, 0),
      accepted: differingBits === 0 && candidateRoot === this.root };
  }
  // Candidate-first, no mutation: exact quine's source bits must remain identical.
  loop(cycles = 1000, { mutationAt = -1, mutationIndex = 0, auditEvery = 0 } = {}) {
    positive(cycles, 'cycles', 1000);
    if (mutationAt >= 0) {
      positive(mutationAt, 'mutationAt', cycles - 1);
      positive(mutationIndex, 'mutationIndex', this.source.length - 1);
    }
    positive(auditEvery, 'auditEvery', 1000);
    const metrics = { logicalCycles: cycles, accepted: 0, rolledBack: 0, shadowNodeRehashes: 0, deepAudits: 0, deepAuditsPassed: 0 };
    for (let t = 0; t < cycles; t++) {
      const edits = t === mutationAt ? [[mutationIndex, this.source[mutationIndex] ^ 1]] : [];
      const result = this.shadow(edits);
      metrics.shadowNodeRehashes += result.touchedNodes;
      if (result.accepted) metrics.accepted++;
      else metrics.rolledBack++;
      if (auditEvery && (t + 1) % auditEvery === 0) {
        metrics.deepAudits++;
        if (this.deepAudit().pass) metrics.deepAuditsPassed++;
        else throw new Error(`deep integrity audit failed at cycle ${t + 1}`);
      }
    }
    metrics.finalRoot = this.root;
    return metrics;
  }
}

function demonstration(source = Buffer.from(classicalQuineSource(), 'utf8')) {
  verifyBundledClassicQuine(1000);
  const t0 = process.hrtime.bigint();
  const engine = new SparseInward1000(source);
  const builtMs = Number(process.hrtime.bigint() - t0) / 1e6;
  const audit = engine.deepAudit();
  assert(audit.pass, 'initial deep audit must pass');
  const t1 = process.hrtime.bigint();
  const loop = engine.loop(1000, { mutationAt: 499, auditEvery: 250 });
  const loopMs = Number(process.hrtime.bigint() - t1) / 1e6;
  return { model: 'sparse 1000^3 virtual inward Merkle octree, not 1e9 physical operations',
    ...engine.metadata(), constructionMs: builtMs, loopMs, fullReconstruction: audit, loop };
}

module.exports = { SIDE, VIRTUAL_CELLS, DIMS, coord, linear, SparseInward1000, demonstration };
if (require.main === module) console.log(JSON.stringify(demonstration(), null, 2));
