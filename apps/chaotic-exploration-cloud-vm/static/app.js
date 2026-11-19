const status=document.querySelector('#status'), output=document.querySelector('#output');
const btn=document.querySelector('#run');
fetch('/api/status').then(r=>r.json()).then(d=>{
  status.textContent=d.ok ? `Native executable present · ${d.cloud_deployed?'Cloud Run':'local/container'} runtime` : 'Native executable unavailable';
}).catch(()=>{status.textContent='API unreachable';});
document.querySelector('#form').addEventListener('submit',async e=>{
  e.preventDefault(); btn.disabled=true; status.textContent='Executing…';
  const ints=new Set(['side','steps','seed','horizon']), data={};
  for(const [k,v] of new FormData(e.target))data[k]=ints.has(k)?Number.parseInt(v,10):Number.parseFloat(v);
  try{
    const r=await fetch('/api/execute',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
    const result=await r.json();output.textContent=JSON.stringify(result,null,2);
    status.textContent=r.ok?'Execution completed and numeric state verified':`Request rejected (${r.status})`;
  }catch(err){output.textContent=String(err);status.textContent='Execution failed';}
  finally{btn.disabled=false;}
});
