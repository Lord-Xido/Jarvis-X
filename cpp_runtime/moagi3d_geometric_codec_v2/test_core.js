const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const c = require('./moagi_core.js');
assert.equal(c.N,784); assert.equal(c.K,16); assert.equal(c.basisInt.length,12544);
let tested=0;
for(const mode of ['torus','sphere','helix','wave']) {
  for(const seed of [0,17,42,2026,98765]) {
    const original=c.generate(mode,seed);
    const z=c.encode(original), approx=c.decodeBytes(z), packed=c.pack(original,z);
    assert.deepEqual(c.unpack(packed.bytes).voxels,original);
    assert.equal(packed.lossless,true);
    assert.equal(packed.bytes.length,16+64+784);
    assert.ok(c.mse(original,approx)>=0);
    assert.deepEqual(c.pack(original,c.encode(original)).bytes,packed.bytes);
    tested++;
  }
}
for(const rho of [0.1,0.55,0.95]){
  const target=c.encode(c.generate('sphere',321));let z=new Float64Array(c.K),prev=c.l2(z,target);
  for(let t=0;t<28;t++){z=c.iterate(z,target,rho);const next=c.l2(z,target);if(prev>1e-10)assert.ok(next<=prev*(rho+1e-8));prev=next;}
}
const sample=c.generate(),packet=c.pack(sample,c.encode(sample)).bytes;
for(const pos of [0,4,9,11,12,30,packet.length-1]){
  const tampered=packet.slice();tampered[pos]^=7;assert.throws(()=>c.unpack(tampered));
}
fs.writeFileSync(path.join(__dirname,'test_torus.dm3d'),packet);
fs.writeFileSync(path.join(__dirname,'test_input.raw'),sample);
console.log('PASS '+tested+' shape/seed roundtrips; 3 contraction tests; 7 corruption rejection cases');
