'use strict';
(() => {
  const $ = id => document.getElementById(id);
  const canvas=$('field'),ctx=canvas.getContext('2d',{alpha:false});
  let state=null,points=[],route=[],width=400,height=400,ratio=1;
  let yaw=0.64,pitch=-0.36,zoom=1,drag=false,prev=null,lastPos=[],selected=null,lastSeen='';
  let cameraTurn=true,loading=false;
  const log=(message,kind='')=>{
    const el=document.createElement('div');el.className='logitem '+kind;
    const time=document.createElement('time');time.textContent='['+new Date().toLocaleTimeString()+'] ';
    el.append(time,document.createTextNode(message));
    $('log').prepend(el);
    while($('log').children.length>28)$('log').lastElementChild.remove();
  };
  async function get(path){const r=await fetch(path,{cache:'no-store'});const data=await r.json();if(!r.ok)throw Error(data.error||'HTTP '+r.status);return data;}
  async function post(path,value={},binary=false){
    const r=await fetch(path,{method:'POST',headers:binary?{}:{'Content-Type':'application/json'},body:binary?value:JSON.stringify(value)});
    const data=await r.json();if(!r.ok)throw Error(data.error||'HTTP '+r.status);return data;
  }
  async function invoke(callback){try{await callback();await refresh();}catch(e){log(e.message,'error');$('api-info').textContent='ERROR: '+e.message;}}
  const write=(id,x)=>{$(id).textContent=x};
  const prettyNum=n=>Number(n).toLocaleString('en-US');
  function updateStats(s){
    write('grid',s.dimension+' × '+s.dimension+' × '+s.dimension);
    write('voxels',prettyNum(s.voxels)+' ACTUAL VOXELS');
    write('ticks',prettyNum(s.ticks));
    write('failures',s.active_faults+' / '+s.repaired_faults);
    write('codec-ratio',s.last_codec_ratio?Math.round(s.last_codec_ratio*100)+'%':'—');
    write('codec-state',s.codec_verified?'CRC32 / BIT EXACT':'NOT VERIFIED');
    write('step-ms',Number(s.last_step_ms).toFixed(3)+' ms');
    write('delta',Number(s.last_max_delta).toFixed(6));
    write('ann-loss',Number(s.ann_last_mse).toFixed(6));
    write('ann-epochs',prettyNum(s.ann_epochs)+' TRAINING EPOCHS');
    write('bitflips',prettyNum(s.switching_bits_total));
    write('last-flips',prettyNum(s.last_switching_bits));
    write('byte-updates',prettyNum(s.byte_updates_total));
    write('ann-ratio',s.last_ann_ratio?Number(s.last_ann_ratio).toFixed(2)+'×':'—');
    write('found',s.routes_found+' / '+s.routes_failed+' failed');
    write('expansions',s.last_route_expansions);
    write('uptime',Math.round(s.uptime_s)+' s');
    write('api-info','NATIVE C++20 API · '+(s.auto?'AUTO':'PAUSED'));
    write('status',s.native_error?'BACKEND ERROR':s.auto?'AUTONOMOUS':'PAUSED');
    $('pulse').classList.toggle('off',!s.auto||s.native_error);
    $('btn-auto').textContent=s.auto?'PAUSE CPU EVOLUTION':'RUN CPU EVOLUTION';
    $('btn-auto').classList.toggle('primary',s.auto);
    if(s.last_operation&&s.last_operation!==lastSeen){lastSeen=s.last_operation;log(lastSeen,s.last_operation.includes('VERIFIED')?'success':'');}
    if(s.native_error)log(s.native_error,'error');
    const info=s.route||{};
    if(info.from!==undefined)write('route-result',info.found?
      `PATH ${info.from}→${info.to}: ${info.hops} HOPS / ${info.measured_ms} ms (measured)`:
      `PATH ${info.from}→${info.to}: UNREACHABLE`);
    points=s.values;route=s.path||[];
  }
  async function refresh(){
    if(loading)return;loading=true;
    try{
      const s=await get('/api/state');
      state=s;updateStats(s);
    } catch(e){write('status','DISCONNECTED');$('pulse').classList.add('off');write('api-info',e.message);}
    finally{loading=false;}
  }
  function measure(){const rect=canvas.getBoundingClientRect();width=Math.max(1,rect.width);height=Math.max(1,rect.height);
    ratio=Math.min(window.devicePixelRatio||1,2);canvas.width=Math.round(width*ratio);canvas.height=Math.round(height*ratio);ctx.setTransform(ratio,0,0,ratio,0,0);
  }
  window.addEventListener('resize',measure);if(window.ResizeObserver)new ResizeObserver(measure).observe(canvas);measure();
  function project(id, d) {
    const x=id%d-(d-1)/2,y=Math.floor(id/d)%d-(d-1)/2,z=Math.floor(id/(d*d))-(d-1)/2;
    const cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch);
    const xx=x*cy+z*sy, zz=-x*sy+z*cy, yy=y*cp-zz*sp, z2=y*sp+zz*cp;
    const factor=(Math.min(width,height)*0.84*zoom/d)*(2.65*d)/(2.65*d-z2);
    return {x:width/2+xx*factor,y:height/2-yy*factor,z:z2,scale:factor/(Math.min(width,height)*0.84/d)};
  }
  function projectCoord(x,y,z,d){return project(x+d*y+d*d*z,d);}
  function drawLine(a,b,color,lineWidth=1){ctx.strokeStyle=color;ctx.lineWidth=lineWidth;ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.stroke();}
  function render(now){
    requestAnimationFrame(render);
    if(!state)return;
    if(cameraTurn && state.auto && !drag)yaw+=0.0008;
    ctx.fillStyle='#030916';ctx.fillRect(0,0,width,height);
    const d=state.dimension, corners=[];
    const val=d-1;
    for(let k=0;k<8;k++)corners.push(projectCoord((k&1)?val:0,(k&2)?val:0,(k&4)?val:0,d));
    const edges=[[0,1],[0,2],[0,4],[1,3],[1,5],[2,3],[2,6],[3,7],[4,5],[4,6],[5,7],[6,7]];
    for(const [a,b] of edges)drawLine(corners[a],corners[b],'rgba(72,200,240,.18)',1);
    ctx.fillStyle='#5788ad';ctx.font='10px ui-monospace,monospace';
    ctx.fillText('X',corners[1].x+6,corners[1].y);
    ctx.fillText('Y',corners[2].x+6,corners[2].y);
    ctx.fillText('Z',corners[4].x+6,corners[4].y);
    const failedSet=new Set(state.faults||[]);
    const pts=[];lastPos=new Array(state.voxels);
    for(let i=0;i<state.voxels;i++){
      const p=project(i,d);lastPos[i]=p;
      const mode=$('view-mode').value;
      const display=mode==='ann'?state.reconstructed[i]:
        mode.startsWith('bit')?(((points[i]>>>Number(mode.slice(3)))&1)*255):points[i];
      pts.push({id:i,...p,value:display,bad:failedSet.has(i)});
    }
    pts.sort((a,b)=>a.z-b.z);
    for(const p of pts){
      const r=(p.bad?4.2:1.75+2.8*(p.value/255))*Math.max(.6,p.scale*.85);
      if(p.bad){ctx.fillStyle='#ff437d';ctx.shadowColor='#ff437d';ctx.shadowBlur=10;}
      else{
        const e=p.value/255;
        ctx.fillStyle=`hsla(${195-90*e},${75+e*18}%,${27+e*45}%,${.45+e*.50})`;
        ctx.shadowColor=e>.65?'#43ffc8':'#37bdea';ctx.shadowBlur=e>.75?7:0;
      }
      ctx.beginPath();ctx.arc(p.x,p.y,r,0,Math.PI*2);ctx.fill();
    }
    ctx.shadowBlur=0;
    if(route.length){
      ctx.beginPath();route.forEach((id,j)=>{const p=lastPos[id];if(j)ctx.lineTo(p.x,p.y);else ctx.moveTo(p.x,p.y)});
      ctx.strokeStyle='rgba(67,255,177,.8)';ctx.lineWidth=3;ctx.shadowColor='#27e3a9';ctx.shadowBlur=13;ctx.stroke();ctx.shadowBlur=0;
      for(const id of [route[0],route[route.length-1]]){const p=lastPos[id];ctx.beginPath();ctx.fillStyle='#d0ffea';ctx.arc(p.x,p.y,6,0,2*Math.PI);ctx.fill();}
    }
    if(selected!==null && lastPos[selected]){
      const p=lastPos[selected];ctx.beginPath();ctx.strokeStyle='#f7e18c';ctx.lineWidth=2;ctx.arc(p.x,p.y,10,0,2*Math.PI);ctx.stroke();
    }
    if(state.active_faults>0){ctx.fillStyle='#ff5b88';ctx.font='10px monospace';ctx.fillText(`${state.active_faults} NATIVE FAULT(S)`,18,height-45);}
  }
  requestAnimationFrame(render);
  canvas.addEventListener('pointerdown',e=>{drag=true;prev={x:e.clientX,y:e.clientY,downX:e.clientX,downY:e.clientY};canvas.setPointerCapture(e.pointerId);});
  canvas.addEventListener('pointermove',e=>{
    if(!drag||!prev)return;
    const dx=e.clientX-prev.x,dy=e.clientY-prev.y;
    if(Math.abs(dx)>0||Math.abs(dy)>0){yaw+=dx*.008;pitch=Math.max(-1.4,Math.min(1.4,pitch+dy*.008));cameraTurn=false;}
    prev.x=e.clientX;prev.y=e.clientY;
  });
  canvas.addEventListener('pointerup',e=>{
    if(!prev)return;
    const tapped=Math.hypot(e.clientX-prev.downX,e.clientY-prev.downY)<6;
    drag=false;prev=null;
    if(tapped&&state){
      const rect=canvas.getBoundingClientRect();let best=null,dist=Infinity;
      for(let i=0;i<lastPos.length;i++){let p=lastPos[i];let delta=Math.hypot(p.x-(e.clientX-rect.left),p.y-(e.clientY-rect.top));if(delta<dist){dist=delta;best=i;}}
      if(best!==null&&dist<22){selected=best;const d=state.dimension;
        write('selected',`NODE ${best} · XYZ (${best%d},${Math.floor(best/d)%d},${Math.floor(best/(d*d))}) · Q8=${points[best]} · ANN=${state.reconstructed[best]} · ${state.faults.includes(best)?'FAILED':'HEALTHY'}`);}
    }
  });
  canvas.addEventListener('wheel',e=>{e.preventDefault();zoom=Math.max(.35,Math.min(2.7,zoom*Math.exp(-e.deltaY*.001)));},{passive:false});
  canvas.addEventListener('dblclick',()=>{yaw=.64;pitch=-.36;zoom=1;selected=null;cameraTurn=true;});
  $('view-mode').onchange=()=>{
    const mode=$('view-mode').value;
    write('viewcaption',mode==='ann'?'QUANTIZED 8→4→8 NEURAL RECONSTRUCTION':
      mode.startsWith('bit')?`TRUE Q8 BITPLANE ${mode.slice(3)} / x,y,z COORDINATES`:'PROJECTED NATIVE 3D FIELD · DRAG TO ORBIT');
  };
  $('btn-train').onclick=()=>invoke(async()=>{
    const epochs=Number($('ann-training-epochs').value), learning_rate=Number($('ann-rate').value);
    const s=await post('/api/ann/train',{epochs,learning_rate});
    write('train-result',`BACKPROP ${s.epochs} epochs / ${s.accepted_updates} accepted · MSE ${s.mse_before.toFixed(6)} → ${s.mse_after.toFixed(6)} · ${s.train_ms.toFixed(3)} ms measured`);
    log(`ANN 8→4→8 training: MSE ${s.mse_before.toFixed(6)} → ${s.mse_after.toFixed(6)}`,'success');
  });
  $('btn-ann-export').onclick=()=>invoke(async()=>{
    const r=await fetch('/api/ann/export');if(!r.ok)throw Error('neural export error');
    const blob=await r.blob(),url=URL.createObjectURL(blob),a=document.createElement('a');
    a.href=url;a.download='jarvis-x-neural-3d.jxa';a.click();setTimeout(()=>URL.revokeObjectURL(url),5000);
    log(`JXA1 exported: ${blob.size} B (latent + residual + shared model)`,'success');
  });
  $('ann-import-file').onchange=e=>invoke(async()=>{
    const file=e.target.files[0];if(!file)return;
    const result=await post('/api/ann/import',await file.arrayBuffer(),true);
    log(`JXA1 validated: ${result.voxels} Q8 voxels restored`,'success');e.target.value='';
  });
  $('btn-auto').onclick=()=>invoke(async()=>{const r=await post('/api/auto',{enabled:!state?.auto});log('Autonomous control '+(r.auto?'enabled':'paused'));});
  $('btn-step').onclick=()=>invoke(()=>post('/api/step',{steps:1}));
  $('btn-step10').onclick=()=>invoke(()=>post('/api/step',{steps:10}));
  $('btn-reset').onclick=()=>invoke(async()=>{const r=await post('/api/reset',{seed:20261104});log('Native state reset: '+r.seed,'success');});
  $('btn-fault').onclick=()=>invoke(async()=>{
    const value=$('fault-id').value;const r=await post('/api/fault',{index:value===''?null:Number(value)});
    log('Fault injected at index '+r.faulted+'; recovery after '+r.healing_ticks+' ticks','error');
  });
  $('btn-route').onclick=()=>invoke(()=>post('/api/route',{from:Number($('route-from').value),to:Number($('route-to').value)}));
  $('btn-random-route').onclick=()=>invoke(async()=>{
    const n=state?.voxels||1000,from=Math.floor(Math.random()*n),to=Math.floor(Math.random()*n);
    $('route-from').value=from;$('route-to').value=to;
    await post('/api/route',{from,to});
  });
  $('btn-verify').onclick=()=>invoke(async()=>{const s=await post('/api/verify');log(`Codec ${s.verified?'PASS':'FAIL'}: ${s.raw_bytes} B → ${s.encoded_bytes} B, ratio ${s.ratio}, CRC32 ${s.crc32_checked}`,'success');});
  $('btn-codec').onclick=()=>invoke(async()=>{
    const r=await post('/api/codec',{text:$('codec-input').value});
    write('codec-result',`BIT-EXACT: ${r.verified} • RAW ${r.raw_bytes} B → STREAM ${r.encoded_bytes} B • ${r.measured_ms} ms • HEX ${r.encoded_hex_preview}`);
  });
  $('btn-export').onclick=()=>invoke(async()=>{
    const r=await fetch('/api/export');if(!r.ok)throw Error('export error');
    const blob=await r.blob(),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='jarvis-x-field.jx3';a.click();setTimeout(()=>URL.revokeObjectURL(url),5000);
    log(`Native snapshot exported: ${blob.size} bytes`,'success');
  });
  $('import-file').onchange=e=>invoke(async()=>{
    const file=e.target.files[0];if(!file)return;
    const result=await post('/api/import',await file.arrayBuffer(),true);
    log('Verified and imported '+result.voxels+' field bytes','success');
    e.target.value='';
  });
  refresh();setInterval(refresh,300);
})();
