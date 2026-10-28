import test from 'node:test';
import assert from 'node:assert/strict';
import {G,BLOCK,CELLS,generateCoordinates,encodeCoordinates,decodeCoordinates,verifyCodec,
  initialState,step,verify,optimizeParameters,spectralRadius,simulate,digest} from './core.mjs';

test('virtual domain uses sparse support and 977 folded cells per axis',()=>{
  assert.equal(G,1_000_000);assert.equal(BLOCK,1024);assert.equal(CELLS,977);
  assert.equal(generateCoordinates(12,42).length,36);
});
test('RS3D bit packing reconstructs all boundary probes exactly',()=>{
  const probes=new Uint32Array([0,0,0,G-1,G-1,G-1,1023,1024,1025,524287,524288,524289]);
  const packet=encodeCoordinates(probes);
  assert.equal(packet.length,24+8*4);
  assert.deepEqual(decodeCoordinates(packet),probes);
  assert.equal(verifyCodec(probes).mismatches,0);
});
test('checksum detects accidental corruption and reserved bits reject malformed data',()=>{
  const original=encodeCoordinates(new Uint32Array([1,2,3]));
  const bad=original.slice();bad[26]^=2;
  assert.throws(()=>decodeCoordinates(bad),/checksum/);
  const reserved=original.slice();reserved[31]|=0x80;
  const view=new DataView(reserved.buffer);
  // recompute checksum to isolate reserved-padding validation
  let h=2166136261>>>0;
  for(const b of reserved.subarray(24))h=Math.imul(h^b,16777619)>>>0;
  view.setUint32(20,h,true);
  assert.throws(()=>decodeCoordinates(reserved),/reserved/);
});
test('invalid packets, parameters and out-of-range coordinates fail closed',()=>{
  assert.throws(()=>generateCoordinates(200001,42),/count/);
  assert.throws(()=>encodeCoordinates(new Uint32Array([G,0,0])),/outside/);
  assert.throws(()=>decodeCoordinates(new Uint8Array(5)),/truncated/);
  assert.throws(()=>optimizeParameters([0,1,1]),/weights/);
  const s=initialState(new Uint32Array([20,40,60]));
  assert.throws(()=>step(s,[1,1,1],{q:4,beta:0}),/unstable/);
  assert.throws(()=>simulate(new Uint32Array([0,1,2]),{iterations:10001}),/iterations/);
});
test('tuner produces bounded stable contraction with deterministic parameters',()=>{
  const tuning=optimizeParameters([1,.8,1.25]);
  assert.ok(tuning.rho<1);
  assert.equal(tuning.rho,spectralRadius(tuning.q,tuning.beta,[1,.8,1.25]));
  assert.equal(tuning.candidates,5576);
});
test('1000 logical passes converge, prune, and round-trip 20000 coordinates',()=>{
  const input=generateCoordinates(20_000,42);
  const {state,report}=simulate(input);
  assert.equal(report.iterations,1000);
  assert.equal(report.particleCount,20_000);
  assert.equal(report.codec.mismatches,0);
  assert.equal(report.geometry.nonFinite,0);
  assert.equal(report.geometry.wrongCell,0);
  assert.equal(report.geometry.exactFixedPoint,true);
  assert.equal(report.passed,true);
  assert.equal(report.stateDigest,'3DC62705');
  assert.equal(digest(state),report.stateDigest);
  assert.ok(state.updates < 20_000*1000);
});
test('same seed and config reproduce identical final states',()=>{
  const x=generateCoordinates(256,3);
  const a=simulate(x,{iterations:1000});
  const b=simulate(x,{iterations:1000});
  assert.equal(a.report.stateDigest,b.report.stateDigest);
  assert.deepEqual(a.state.positions,b.state.positions);
});
test('partial terminal block remains bounded and in its original cell',()=>{
  const s=simulate(new Uint32Array([999999,999424,999935]),{iterations:1000});
  assert.equal(s.report.geometry.pass,true);
  assert.equal(verify(s.state).wrongCell,0);
  for (const x of s.state.positions)assert.ok(x>=0&&x<G);
});
