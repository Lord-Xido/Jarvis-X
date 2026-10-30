#!/usr/bin/env node
'use strict';
const crypto = require('node:crypto');
const { SparseInward1000 } = require('./inward1000');
// Equivalent output: candidate Merkle root from a one-byte edit.
// Not a benchmark of full source-integrity scans or a claim of 1e9 actual operations.
const N = 16_384;
const R = 8;
const source = Buffer.allocUnsafe(N);
for (let i=0;i<N;i++) source[i]=(i*23+19)&255;
const baseline = new SparseInward1000(source);
const patches = Array.from({length:R},(_,j)=>[1000+j*1127,source[1000+j*1127]^1]);
const measure = fn => {
  const samples=[];
  for (let j=0;j<R;j++) {
    const t0=process.hrtime.bigint();
    fn(j);
    samples.push(Number(process.hrtime.bigint()-t0)/1e6);
  }
  samples.sort((a,b)=>a-b);
  return { medianMs:(samples[3]+samples[4])/2, minMs:samples[0], maxMs:samples.at(-1) };
};
let expectedRoots=[];
const full = measure(j=>{
  const candidate=Buffer.from(source);
  candidate[patches[j][0]]=patches[j][1];
  expectedRoots[j]=new SparseInward1000(candidate).root;
});
const incremental = measure(j=>{
  const receipt=baseline.shadow([patches[j]]);
  if (receipt.candidateRoot!==expectedRoots[j]) throw new Error('Merkle root differs between shadow and full rebuild');
});
const report={virtualCells:1_000_000_000, physicalSourceBytes:N, candidateUpdates:R,
  fullRebuild:full, pathOnlyShadow:incremental,
  observedRatioFullRebuildToShadow:full.medianMs/incremental.medianMs,
  interpretation:'Time to compute one modified Merkle root only; NOT a general-purpose 1e9x speedup; no source reread or full audit in incremental mode.'};
console.log(JSON.stringify(report,null,2));
