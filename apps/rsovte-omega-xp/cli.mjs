#!/usr/bin/env node
import {readFileSync,writeFileSync} from 'node:fs';
import {generateCoordinates,decodeCoordinates,encodeCoordinates,simulate} from './core.mjs';
function arg(flag,defaultValue) {
  const ix=process.argv.indexOf('--'+flag);
  if(ix===-1)return defaultValue;
  if(ix+1>=process.argv.length)throw new Error('--'+flag+' requires a value');
  return process.argv[ix+1];
}
try {
  const input=arg('input',null),count=Number(arg('count','20000')),
    seed=Number(arg('seed','42')),iterations=Number(arg('iterations','1000')),
    weights=String(arg('weights','1,0.8,1.25')).split(',').map(Number);
  const coords=input?decodeCoordinates(new Uint8Array(readFileSync(input))):generateCoordinates(count,seed);
  const {report}=simulate(coords,{iterations,weights});
  const output=arg('out',null),packet=arg('packet',null);
  if(output)writeFileSync(output,JSON.stringify(report,null,2)+'\n');
  if(packet)writeFileSync(packet,encodeCoordinates(coords));
  console.log(JSON.stringify({passed:report.passed,iterations:report.iterations,
    particles:report.particleCount,actualPointUpdates:report.actualPointUpdates,
    codecMismatches:report.codec.mismatches,finalResidual:report.geometry.maxError,
    digest:report.stateDigest},null,2));
  if(!report.passed)process.exitCode=2;
}catch(e){console.error(e.message);process.exitCode=1;}
