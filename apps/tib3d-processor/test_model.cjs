'use strict';
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const vm=require('node:vm');
const {test}=require('node:test');

const html=fs.readFileSync(path.join(__dirname,'index.html'),'utf8');
const m=html.match(/<script id="tib3d-model">([\s\S]*?)<\/script>/);
assert.ok(m,'model script must exist');
const Model=vm.runInNewContext(m[1]+'\nTiB3DModel;');

test('inline scripts parse',()=>{
 const scripts=[...html.matchAll(/<script[^>]*>([\s\S]*?)<\/script>/g)];
 assert.equal(scripts.length,2);
 scripts.forEach(s=>new vm.Script(s[1]));
});

test('exact TiB geometry and target arithmetic',()=>{
 const x=new Model();
 assert.equal(x.planeBytes,67108864);
 assert.equal(x.totalBytes,1099511627776);
 assert.equal(x.totalBytes,2**40);
 assert.equal(x.targetGBps,1000);
 assert.ok(Math.abs(x.targetSweepSeconds-1.099511627776)<1e-15);
});

test('3D address mapping round trips exactly',()=>{
 const x=new Model();
 const probes=[[0,0,0],[8191,0,0],[0,8191,0],[0,0,16383],[8191,8191,16383],[17,4096,8192]];
 for(const p of probes){
  const a=x.address(...p),q=x.coord(a);
  assert.deepEqual([q.x,q.y,q.z],p);
 }
 assert.equal(x.address(8191,8191,16383),x.totalBytes-1);
 assert.throws(()=>x.address(8192,0,0),{name:'RangeError'});
 assert.throws(()=>x.coord(x.totalBytes),{name:'RangeError'});
});

test('pipeline ordering and residual commit are deterministic',()=>{
 const x=new Model();
 assert.equal(x.stages.length,8);
 for(let i=0;i<5;i++)x.step();
 assert.equal(x.stages[x.stage],'RESIDUAL');
 x.step();
 assert.equal(x.stages[x.stage],'VERIFY');
 assert.equal(x.decision,'COMMIT');
 const before=x.residual;
 x.step();
 assert.equal(x.stages[x.stage],'RECUR');
 assert.ok(x.residual<before);
 assert.equal(x.iteration,1);
});

test('advance is bounded and recurrence can halt',()=>{
 const x=new Model();
 assert.throws(()=>x.advance(-1),{name:'RangeError'});
 assert.throws(()=>x.advance(1,0),{name:'RangeError'});
 x.setRecurrence(false);
 for(let i=0;i<100&&x.running;i++)x.advance(.5,1);
 assert.equal(x.running,false);
 assert.ok(x.iteration>=1);
});

test('residual stays positive and nonincreasing under commits',()=>{
 const x=new Model();let prev=x.residual;
 for(let i=0;i<80;i++){
  x.step();
  if(x.stages[x.stage]==='RECUR'){
   assert.ok(x.residual>0);
   assert.ok(x.residual<=prev);
   prev=x.residual;
  }
 }
});
