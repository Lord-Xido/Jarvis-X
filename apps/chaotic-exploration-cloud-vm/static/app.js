'use strict';
const statusNode=document.querySelector('#status');
const output=document.querySelector('#output');
const runButton=document.querySelector('#run');
const metrics=document.querySelector('#metrics');
const fieldCanvas=document.querySelector('#field');
const traceCanvas=document.querySelector('#residual');
const fmt=(v,d=6)=>Number(v).toPrecision(d);
const finite=x=>Number.isFinite(x);
function makeField(side,type){
  const vals=[];
  for(let z=0;z<side;z++)for(let y=0;y<side;y++)for(let x=0;x<side;x++){
    const X=(x+.5)/side-.5,Y=(y+.5)/side-.5,Z=(z+.5)/side-.5;
    const r2=X*X+Y*Y+Z*Z;
    vals.push(type==='sphere'? Math.exp(-15*r2)-.2 :
      Math.sin(2*Math.PI*X)*Math.cos(2*Math.PI*Y)*Math.sin(2*Math.PI*Z));
  }
  return vals;
}
function paintField(result){
  const ctx=fieldCanvas.getContext('2d'),w=fieldCanvas.width,h=fieldCanvas.height;
  ctx.clearRect(0,0,w,h); ctx.fillStyle='#061320';ctx.fillRect(0,0,w,h);
  const n=result.side;
  const full=Array.isArray(result.field_zyx), data=full?result.field_zyx:result.xy_slice;
  const s=Math.min(w/(n*1.35),h/(n*1.5))*0.8;
  const pts=[];
  for(let z=0;z<(full?n:1);z++)for(let y=0;y<n;y++)for(let x=0;x<n;x++){
    const realZ=full?z:result.slice_z;
    const i=full?(z*n+y)*n+x:y*n+x;
    const val=data[i];if(!finite(val)||Math.abs(val)<.012)continue;
    const px=w/2+(x-y)*s*.70;
    const py=h*.78+(x+y-n)*s*.25-realZ*s*.68;
    pts.push({depth:x+y+realZ,px,py,val});
  }
  pts.sort((a,b)=>a.depth-b.depth);
  for(const p of pts){
    const strength=Math.min(1,Math.abs(p.val)*4);
    const radius=Math.max(1.25, Math.min(6,s*.14)*(.35+strength));
    ctx.beginPath();ctx.arc(p.px,p.py,radius,0,2*Math.PI);
    ctx.fillStyle=p.val>=0?`rgba(78,233,219,${.18+.78*strength})`:
                             `rgba(238,150,90,${.18+.78*strength})`;
    ctx.fill();
  }
  ctx.fillStyle='#92b7c5';ctx.font='14px ui-monospace,monospace';
  ctx.fillText(`X/Y/Z lattice: ${n}³`,15,23);
  document.querySelector('#fieldnote').textContent=full?
    `Rendered ${pts.length} active voxels from the actual native field (ZYX indexing)`:
    `Rendered central XY slice at z=${result.slice_z}; enable side ≤ 16 for full 3D field.`;
}
function paintTrace(rows){
  const ctx=traceCanvas.getContext('2d'),w=traceCanvas.width,h=traceCanvas.height;
  ctx.clearRect(0,0,w,h); ctx.fillStyle='#061320';ctx.fillRect(0,0,w,h);
  if(!rows?.length)return;
  const left=65,right=w-20,top=25,bottom=h-42;
  const ys=rows.map(r=>r.residual_rms);
  const hi=Math.max(...ys,1e-10)*1.05;
  ctx.strokeStyle='#284655';ctx.lineWidth=1;
  for(let j=0;j<=4;j++){
    const y=top+(bottom-top)*j/4;
    ctx.beginPath();ctx.moveTo(left,y);ctx.lineTo(right,y);ctx.stroke();
    ctx.fillStyle='#9fbbc8';ctx.font='12px ui-monospace,monospace';
    ctx.fillText(fmt(hi*(1-j/4),3),4,y+4);
  }
  ctx.strokeStyle='#55ddcd';ctx.lineWidth=3;ctx.beginPath();
  rows.forEach((r,i)=>{
    const x=rows.length===1?(left+right)/2:left+(right-left)*i/(rows.length-1);
    const y=bottom-(bottom-top)*r.residual_rms/hi;
    if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y);
  });ctx.stroke();ctx.fillStyle='#bbd7e0';ctx.fillText(`Iteration 1 — ${rows.length}`,left,bottom+26);
}
function updateMetrics(a){
  const fields=[['Voxels',a.voxels.toLocaleString()],['Executed steps',a.iterations],
    ['Residual RMS',fmt(a.residual_rms)],['L∞ residual',fmt(a.residual_linf)],
    ['Field energy',fmt(a.field_energy)],['Stable spatial margin',fmt(a.stability_margin,4)],
    ['Chosen path',a.path_id],['Native CPU time',`${fmt(a.runtime_ms,4)} ms`],
    ['3D boundedness',a.spatial_invariants_verified?'PASS':'FAIL'],
    ['Stationarity observed',a.stationary_observed?'Yes':'No']];
  metrics.replaceChildren(...fields.map(([label,value])=>{
    const d=document.createElement('div');d.className='metric';
    const h=document.createElement('span');h.textContent=label;
    const v=document.createElement('strong');v.textContent=String(value);d.append(h,v);return d;
  }));
}
fetch('/api/status').then(r=>r.json()).then(d=>{
  statusNode.textContent=d.ok?`Native CPU available · ${d.cloud_deployed?'Cloud Run':'local/container'}`:'Native binary unavailable';
}).catch(()=>statusNode.textContent='Gateway unreachable');
document.querySelector('#form').addEventListener('submit',async ev=>{
  ev.preventDefault();runButton.disabled=true;statusNode.textContent='Executing native 3D recurrence…';
  const intKeys=new Set(['side','steps','seed','horizon']);
  const payload={trace:true};
  for(const [key,value] of new FormData(ev.target)){
    if(key==='source')continue;
    payload[key]=intKeys.has(key)?Number(value):Number(value);
  }
  const source=ev.target.elements.source.value;
  if(source!=='seeded'&&payload.side>16){
    statusNode.textContent='Explicit geometric voxels require side ≤ 16';runButton.disabled=false;return;
  }
  if(source!=='seeded')payload.values=makeField(payload.side,source);
  payload.emit_field=payload.side<=16;
  try{
    const response=await fetch('/api/execute',{
      method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
    const result=await response.json();output.textContent=JSON.stringify(result,null,2);
    if(!response.ok){statusNode.textContent=`Request failed (${response.status}) · ${result.error}`;return;}
    updateMetrics(result);paintField(result);paintTrace(result.trace);
    statusNode.textContent='Completed · native 3D invariants checked';
  }catch(err){statusNode.textContent=`Execution failed: ${String(err)}`;}
  finally{runButton.disabled=false;}
});
