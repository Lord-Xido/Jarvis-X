'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const { SIDE, DIMS, VIRTUAL_CELLS, coord, linear, SparseInward1000, demonstration } = require('./inward1000');

test('logical extent is 1000^3 with ceil-saturating inward dimensions', () => {
  assert.equal(SIDE, 1000);
  assert.equal(VIRTUAL_CELLS, 1_000_000_000);
  assert.deepEqual(DIMS, [1000,500,250,125,63,32,16,8,4,2,1]);
});
test('3D bijective indexing includes last logical voxel without allocating cube', () => {
  for (const i of [0,1,999,1000,1000000,999_999_999]) {
    assert.equal(linear(...coord(i)), i);
  }
  assert.deepEqual(coord(999_999_999), [999,999,999]);
});
test('source reconstruction reads stored leaf payload independently', () => {
  for (const buffer of [Buffer.alloc(0), Buffer.alloc(512), Buffer.from([0,1,255,2,0,3]), crypto.randomBytes(9000)]) {
    const e = new SparseInward1000(buffer);
    assert.deepEqual(e.reconstruct(), buffer);
    assert.equal(e.deepAudit().pass, true);
  }
});
test('small source only allocates sparse ancestor paths, not 1e9 cells', () => {
  const e = new SparseInward1000(Buffer.from('quine'));
  assert.equal(e.metadata().nonzeroLeaves, 5);
  assert.ok(e.metadata().allocatedHashesByLevel.reduce((a,b)=>a+b,0) < 100);
  assert.equal(e.metadata().inwardFolds, 10);
});
test('one-bit mutation candidate changes Merkle root and is refused', () => {
  const e = new SparseInward1000(Buffer.from('quine'));
  const original = e.root;
  const attempt = e.shadow([[0, 'q'.charCodeAt(0) ^ 1]]);
  assert.equal(attempt.differingBits, 1);
  assert.equal(attempt.accepted, false);
  assert.notEqual(attempt.candidateRoot, original);
  assert.equal(e.root, original);
  assert.deepEqual(e.reconstruct(), Buffer.from('quine'));
});
test('Merkle shadow candidate equals independently rebuilt full 3D tree', () => {
  const source = crypto.randomBytes(2560);
  const e = new SparseInward1000(source);
  const patches = [[0, source[0]^1],[1000,0],[2000,255],[2559,source[2559]^4]];
  const expected = Buffer.from(source);
  for (const [i,b] of patches) expected[i]=b;
  const a=e.shadow(patches);
  const b=new SparseInward1000(expected);
  assert.equal(a.candidateRoot,b.root);
  assert.equal(a.accepted,false);
  assert.ok(a.touchedNodes <= patches.length*DIMS.length);
  assert.equal(e.deepAudit().pass,true);
});
test('identical candidate patch passes and retains exact root', () => {
  const bytes = Buffer.from('MAXIDO');
  const e = new SparseInward1000(bytes);
  const a=e.shadow([[0,bytes[0]],[4,bytes[4]]]);
  assert.equal(a.accepted,true);
  assert.equal(a.differingBits,0);
  assert.equal(a.candidateRoot,e.root);
});
test('mutation and no-op mix: 1 rollback in 1000 cycles and four deep audits', () => {
  const e = new SparseInward1000(Buffer.from('quine'));
  const data=e.loop(1000,{mutationAt:499,auditEvery:250});
  assert.equal(data.accepted,999);
  assert.equal(data.rolledBack,1);
  assert.equal(data.deepAudits,4);
  assert.equal(data.deepAuditsPassed,4);
  assert.equal(data.finalRoot,e.root);
});
test('tampered stored leaf payload fails independent deep integrity validation', () => {
  const e = new SparseInward1000(Buffer.from('quine'));
  e.leaves.set(0, e.leaves.get(0)^1);
  assert.equal(e.deepAudit().pass,false);
});
test('tampered cached parent hash fails deep integrity validation', () => {
  const e = new SparseInward1000(Buffer.from('quine'));
  e.levels[3].set(0,'0'.repeat(64));
  assert.equal(e.deepAudit().pass,false);
});
test('input ceilings, malformed edit paths and duplicate patches fail closed', () => {
  assert.throws(()=>new SparseInward1000('source'),/bytes/);
  assert.throws(()=>new SparseInward1000(Buffer.alloc(3),{maxBytes:2}),/ceiling/);
  const e = new SparseInward1000(Buffer.from('123'));
  assert.throws(()=>e.shadow([[3,0]]),/byteIndex/);
  assert.throws(()=>e.shadow([[1,256]]),/byteValue/);
  assert.throws(()=>e.shadow([[1,1],[1,2]]),/duplicate/);
  assert.throws(()=>e.loop(1001),/cycles/);
});
test('complete standard demonstration preserves quine source', () => {
  const a=demonstration();
  assert.equal(a.fullReconstruction.pass,true);
  assert.equal(a.sourceBytes,161);
  assert.equal(a.loop.rolledBack,1);
});
