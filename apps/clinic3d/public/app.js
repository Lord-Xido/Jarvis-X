'use strict';
const $=id=>document.getElementById(id);
let csrf=null,voxels=new Array(512).fill(0),angle=0;
function draw(){
 const c=$('volume'),ctx=c.getContext('2d');ctx.clearRect(0,0,c.width,c.height);
 ctx.save();ctx.translate(250,170);angle+=.004;
 for(let z=0;z<8;z++)for(let y=0;y<8;y++)for(let x=0;x<8;x++){
  const index=x+8*y+64*z,v=voxels[index];if(!v)continue;
  const xx=(x-3.5)*24,yy=(y-3.5)*24,zz=(z-3.5)*24;
  const rx=xx*Math.cos(angle)-zz*Math.sin(angle);
  const rz=xx*Math.sin(angle)+zz*Math.cos(angle);
  const scale=310/(380+rz),px=rx*scale,py=(yy*.9-rz*.25)*scale;
  ctx.fillStyle='rgba(106,221,239,'+(0.15+0.85*v/255)+')';
  ctx.fillRect(px,py,Math.max(1,3*scale),Math.max(1,3*scale));
 }
 ctx.restore();requestAnimationFrame(draw);
}
draw();
$('form').addEventListener('submit',async e=>{
 e.preventDefault();$('status').textContent='Simulating…';$('sources').replaceChildren();$('trace').textContent='';
 try{
  const r=await fetch('/api/simulate',{method:'POST',headers:{'content-type':'application/json'},
   body:JSON.stringify({text:$('text').value,language:$('lang').value,flags:[]})});
  const d=await r.json();if(!r.ok)throw new Error(d.error||'Simulation failed');
  voxels=d.cube.voxels;
  $('status').textContent=d.classification+' — '+d.notice;
  for(const item of d.evidence){const p=document.createElement('p'),a=document.createElement('a');
    a.href=item.url;a.target='_blank';a.rel='noopener noreferrer';a.textContent=item.title+' (source index)';
    p.append(a);$('sources').append(p);
  }
  $('trace').textContent=JSON.stringify({bytecode:d.bytecode,stageProofs:d.stageProofs,trace:d.trace},null,2);
 }catch(err){$('status').textContent=String(err.message)}
});
$('login').addEventListener('click',async()=>{
 try{const r=await fetch('/api/admin/login',{method:'POST',headers:{'content-type':'application/json'},
 body:JSON.stringify({password:$('pw').value})});const d=await r.json();$('pw').value='';
 if(!r.ok)throw Error(d.error);csrf=d.csrf;$('adminResult').textContent='Signed in for 15 minutes';
 $('disable').hidden=false;$('enable').hidden=false;}catch(e){$('adminResult').textContent=String(e.message)}
});
for(const [id,enabled] of [['disable',false],['enable',true]]){
 $(id).addEventListener('click',async()=>{
  const r=await fetch('/api/admin/toggle',{method:'POST',
  headers:{'content-type':'application/json','x-csrf-token':csrf},
  body:JSON.stringify({enabled})});const d=await r.json();
  $('adminResult').textContent=r.ok?'Processing enabled: '+d.enabled:(d.error||'Denied');
 });
}
