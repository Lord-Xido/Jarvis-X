import assert from 'node:assert/strict';
import {startServer} from '../server.mjs';
const s=await startServer(0);
try{
 const addr=s.address(),base='http://127.0.0.1:'+addr.port;
 const health=await fetch(base+'/health');assert.equal(health.status,200);
 const r=await fetch(base+'/api/simulate',{method:'POST',headers:{'content-type':'application/json'},
 body:JSON.stringify({text:'jaw swelling and trouble swallowing',language:'en',flags:[]})});
 assert.equal(r.status,200);const j=await r.json();
 assert.equal(j.classification,'SIMULATION_RED_FLAG');
 assert.equal(j.stageProofs.length,3);
 const html=await (await fetch(base+'/')).text();
 assert.match(html,/Clinic3D/);assert.match(html,/Not a medical device/);
 console.log('HTTP smoke PASS');
}finally{await new Promise(resolve=>s.close(resolve));}
