'use strict';
const assert = require('node:assert/strict');
const {Runtime}=require('../dm3d_core.js');
let n=0;
function check(v,msg){assert.ok(v,msg);n++;}
for(const side of [8,16]){
 const r=new Runtime(side,12345),v=r.volume('torus',.15);
 for(let k=0;k<side*2;k++){
  const a=r.jump(v.noisy,k),b=r.repeated(v.noisy,k);
  check(a.every((x,i)=>x===b[i]),'jump matches repeated '+side+' k='+k);
 }
 for(const k of [side/2,1_000_000,1_000_003]){
  const a=r.jump(v.noisy,k),b=r.repeated(v.noisy,k%(side/2));
  check(a.every((x,i)=>x===b[i]),'periodicity '+side+' k='+k);
 }
 assert.throws(()=>r.jump(v.noisy,-1),RangeError);
 assert.throws(()=>r.jump(v.noisy,1.3),RangeError);
 assert.throws(()=>r.jump(new Uint8Array(5),1),TypeError);
}
const r=new Runtime(16,333),train=r.createDataset(16),validation=r.createDataset(8),audit=r.createDataset(8);
const before=r.evaluate(validation);
r.train(train,6000,48,.12);
const after=r.evaluate(validation),out=r.evaluate(audit);
check(after.mse<before.mse*.5,'supervised validation MSE improves substantially');
check(after.byTask.restore.accuracy>.88,'restoration validation accuracy');
check(after.byTask.edge.accuracy>.73,'edge validation accuracy');
check(out.byTask.restore.iou>=0 && out.byTask.restore.iou<=1,'IoU audit range');
check(out.byTask.edge.balancedAccuracy>=0 && out.byTask.edge.balancedAccuracy<=1,'balanced audit range');
const assisted=out.byTask.xor.accuracy;
const counter=r.auditOracleFreeXor(audit);
check(counter.count===8,'independent audit has expected number of XOR samples');
check(counter.accuracy<=assisted,'oracle-free counterfactual no better than oracle-assisted on seeded data');
for(const task of ['restore','edge','xor']){
 const v=r.volume('sphere',.13);
 const om=r.omega;
 const fold=r.fold(v,task,{tol:1e-8,maxIter:100,recordMemory:false});
 check(fold.certifiedBound<=1e-8 && fold.contractionLip<1,'certified contraction '+task);
 check(r.omega===om,'read-only fold does not mutate memory '+task);
 const q=.5*r.λ*(1+r.γ),zs=r.smooth(fold.z),eh=r.smooth(fold.head);
 let inf=0;
 for(let i=0;i<r.N;i++){
  const f=(1-r.λ)*fold.b[i]+.5*r.λ*(1-r.γ)*eh[i]+q*zs[i];
  inf=Math.max(inf,Math.abs(f-fold.z[i]));
 }
 check(inf<=1e-8,'fixed-point residual '+task);
}
console.log(JSON.stringify({status:'pass',assertions:n,validationBefore:before.mse,validationAfter:after.mse,auditLearnedAccuracy:(out.byTask.restore.accuracy+out.byTask.edge.accuracy)/2,oracleAssistedXorAccuracy:assisted,oracleFreeXorAccuracy:counter.accuracy,byTask:out.byTask},null,2));