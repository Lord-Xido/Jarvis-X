import test from 'node:test';
import assert from 'node:assert/strict';
import { randomBytes } from 'node:crypto';
import { BYTECODE,PROGRAM_SHA256,sha,encodeCube,decodeCube,voxelize,seal,
 openPacket,run,setEnabled } from '../engine.mjs';
test('bitcode fixed 14x32 bit instructions',()=>{
 assert.equal(BYTECODE.length,56);assert.equal(sha(BYTECODE),PROGRAM_SHA256);
});
test('3D transform reverses every bit for variable sizes',()=>{
 for(const n of [0,1,2,7,8,9,16,64,125,511,512,999]){
  const b=randomBytes(n);const e=encodeCube(b);
  assert.deepEqual(decodeCube(e),b);
 }
});
test('deterministic 512 cell 3D stencil',()=>{
 const a=voxelize('abc'),b=voxelize('abc');
 assert.equal(a.length,512);assert.deepEqual(a,b);
});
test('AES-GCM bit tamper and wrong AAD rejected',()=>{
 const root=randomBytes(32),rid='case';
 const p=seal({hello:'test'},0,'GENESIS',rid,root);
 assert.deepEqual(openPacket(p,0,'GENESIS',rid,root),{hello:'test'});
 assert.throws(()=>openPacket(p,1,'GENESIS',rid,root));
 assert.throws(()=>openPacket(p,0,'forged',rid,root));
 const q={...p,data:Buffer.from(p.data,'base64').map((v,i)=>i===0?v^1:v).toString('base64')};
 assert.throws(()=>openPacket(q,0,'GENESIS',rid,root));
 assert.throws(()=>openPacket({...p,length:p.length-1},0,'GENESIS',rid,root));
 assert.throws(()=>openPacket({...p,side:p.side+1},0,'GENESIS',rid,root));
});
test('3 authenticated clinical stages, no treatment orders',()=>{
 const r=run({text:'jaw swelling and trouble swallowing',flags:[],language:'en'});
 assert.equal(r.classification,'SIMULATION_RED_FLAG');
 assert.equal(r.stageProofs.length,3);assert.equal(r.orders.length,0);
 assert.equal(r.trace.length,13);assert.equal(r.cube.voxels.length,512);
});
test('unrecognised case never cleared',()=>{
 assert.equal(run({text:'other symptom',flags:[],language:'zu'}).classification,'UNASSESSED_NOT_CLEARED');
});
test('modified bytecode rejected',()=>{
 const b=Buffer.from(BYTECODE);b[4]^=1;
 assert.throws(()=>run({text:'none',flags:[],language:'en'},{bytecode:b}),/integrity/);
});
test('admin kill switch blocks VM',()=>{
 setEnabled(false);
 assert.throws(()=>run({text:'none',flags:[],language:'en'}),/disabled/);
 setEnabled(true);
});
test('malformed inputs rejected',()=>{
 assert.throws(()=>run({text:4,flags:[],language:'en'}),/Invalid text/);
 assert.throws(()=>run({text:'x',flags:[],language:'zz'}),/Invalid language/);
});
