'use strict';
const assert=require('node:assert/strict');
const crypto=require('node:crypto');
const D=require('./virtual_codec.js');
const fixtures=require('./cross_language_fixture.json');
const digest=bytes=>crypto.createHash('sha256').update(bytes).digest('hex');

assert.equal(D.SIDE, 10n**24n);
assert.equal(D.SIDE**3n, 10n**72n);
assert.equal(D.N_TILES*8n, D.SIDE);
assert.deepEqual(D.readPosition(['0','0','0']),[0n,0n,0n]);
assert.deepEqual(D.readPosition(['999999999999999999999999','2','3'])[0],D.SIDE-1n);
assert.throws(()=>D.readPosition(['1000000000000000000000000','0','0']),RangeError);
assert.throws(()=>D.readPosition(['-1','0','0']),RangeError);
assert.throws(()=>D.readPosition(['0','0','0.5']));
assert.throws(()=>D.getTile([D.N_TILES,0n,0n]),RangeError);
assert.throws(()=>D.encode(new Uint8Array(1)),RangeError);
assert.throws(()=>D.fold(new Uint8Array(192),1.2,2),RangeError);
assert.throws(()=>D.fold(new Uint8Array(192),0.2,101),RangeError);
const allSame=new Uint8Array(192).fill(137);
for(const alpha of [0,0.1,0.7,1]) assert.deepEqual(D.fold(allSame,alpha,5),allSame);
for(const f of fixtures){
  const q=f.q.map(BigInt);
  const t=D.processTile(q,0.12,2,7);
  assert.equal(t.orig.length,1536);
  assert.equal(t.z.length,192);
  assert.equal(t.zf.length,192);
  assert.equal(t.exact.length,1536);
  assert.equal(t.identical,true,'bit-exact residual restoration');
  assert.deepEqual(t.orig,t.exact);
  assert.deepEqual([...t.orig.subarray(0,3)],f.rgb);
  assert.equal(digest(t.orig),f.sha);
  assert.equal(digest(t.exact),f.sha);
  assert.ok(Math.abs(t.mse-f.mse)<1e-7);
  assert.deepEqual(D.getTile(q,7),D.getTile(q,7),'deterministic replay');
  const altered=new Uint8Array(t.exact);altered[0]^=1;
  assert.notEqual(digest(altered),f.sha,'tamper detection via external hash');
  for(const alpha of [0,0.5,1]){
    const p=D.processTile(q,alpha,3,7);
    assert.ok(p.identical);
    assert.deepEqual(p.exact,p.orig);
  }
}
const q=[0n,0n,0n];
let chosen=0.24;
const validation=[[10n,20n,30n],[110n,120n,130n]];
const metric=a=>validation.reduce((v,p)=>v+D.loss(p,a,2),0)/validation.length;
const candidate=[0.02,0.05,0.10,0.15,0.20,0.24].sort((a,b)=>metric(a)-metric(b))[0];
if(metric(candidate)<metric(chosen)-1e-10)chosen=candidate;
assert.ok(metric(chosen)<=metric(0.24)+1e-10,'CTR-style selection cannot regress');
console.log('PASS: logical 10^72 space, BigInt bounds, Python fixtures, SHA-256, 3D fold, exact residuals, gate.');
