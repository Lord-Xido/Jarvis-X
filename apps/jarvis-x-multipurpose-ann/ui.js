(function(){
 'use strict';
 const $=id=>document.getElementById(id);
 const model=new DM3D.Runtime(16,333);
 const training=model.createDataset(16), validation=model.createDataset(8), audit=model.createDataset(8);
 let volume=model.volume('torus',.14),mode='restore',view='fold',last=null;
 let yaw=.7,pitch=.38,zoom=1.0,dragging=false,lastMouse=null,orbit=true;
 const main=$('fieldCanvas'),ctx=main.getContext('2d');
 function fmt(x,d=4){return Number.isFinite(x)?x.toFixed(d):'—';}
 function setText(id,v){$(id).textContent=v;}
 function refresh(){
  mode=$('task').value;view=$('view').value;
  model.λ=+$('lam').value;model.γ=+$('gam').value;
  setText('lamVal',fmt(model.λ,2));setText('gamVal',fmt(model.γ,2));
  last=model.fold(volume,mode,{tol:1e-7,maxIter:100,recordMemory:false});
  setText('iterations',last.iterations+'');setText('bound',last.certifiedBound.toExponential(2));
  setText('lip',fmt(last.contractionLip,3));setText('mse',fmt(last.metrics.mse,5));
  setText('acc',fmt(last.metrics.accuracy*100,2)+'%');setText('omega',fmt(last.omega,5));
  const held=model.evaluate(validation);setText('held',fmt(held.mse,5));setText('heldacc',fmt(held.accuracy*100,1)+'%');
  const k=Math.max(0,Math.min(1e6,Math.trunc(+$('jump').value||0)));
  $('jump').value=k;const effective=k%(model.L/2),passes=effective.toString(2).split('1').length-1;
  setText('jumplabel',`K^${k.toLocaleString()} → K^${effective} · ${passes} volume pass${passes===1?'':'es'}`);
  setText('nameShape',volume.shape.toUpperCase());setText('nameTask',mode.toUpperCase());
  drawConvergence();render();
 }
 function newVolume(){volume=model.volume($('shape').value,+$('noise').value);setText('noiseVal',fmt(volume.noise,2));refresh();model.omega=.85*model.omega+.15*last.metrics.mse;setText('omega',fmt(model.omega,5));}
 function drawConvergence(){const c=$('lossCanvas'),cc=c.getContext('2d');const w=c.width=600,h=c.height=110;
  cc.clearRect(0,0,w,h);cc.strokeStyle='rgba(91,149,185,.3)';cc.lineWidth=1;
  for(let y=0;y<=4;y++){cc.beginPath();cc.moveTo(0,10+y*23);cc.lineTo(w,10+y*23);cc.stroke();}
  const arr=last?.history||[];if(!arr.length)return;
  let minL=-9,maxL=0;cc.strokeStyle='#32daef';cc.lineWidth=2;cc.beginPath();
  arr.forEach((v,i)=>{let y=10+(maxL-Math.max(minL,Math.log10(Math.max(v.certifiedBound,1e-12))))/(maxL-minL)*92;
   let x=(i/Math.max(1,arr.length-1))*w;if(i===0)cc.moveTo(x,y);else cc.lineTo(x,y);});cc.stroke();
 }
 function selection(){if(!last)return Float64Array.from(volume.noisy);if(view==='input')return Float64Array.from(volume.noisy);if(view==='truth')return Float64Array.from(last.target);
  if(view==='ANN')return last.head;if(view==='latent')return last.z;if(view==='bit')return Float64Array.from(model.jump(volume.noisy,Number($('jump').value)));return last.predicted;}
 function render(){
  const width=main.width,height=main.height,w=width/2,h=height;
  const columns=[Float64Array.from(volume.noisy),selection()];
  ctx.fillStyle='#050e19';ctx.fillRect(0,0,width,height);
  for(let t=0;t<2;t++){
   const cx=w*(t+.5),cy=h*.51,rad=Math.min(w,h)*.32*zoom;
   ctx.strokeStyle='rgba(67,171,215,.10)';ctx.lineWidth=1;
   for(let q=-3;q<=3;q++){
    ctx.beginPath();ctx.moveTo(cx+q*rad/3,cy-rad*.98);ctx.lineTo(cx+q*rad/3,cy+rad*.98);ctx.stroke();
   }
   let points=[];const v=columns[t];let count=0,energy=0;
   for(let i=0;i<model.N;i++){
    const value=v[i];if(value<.53)continue;
    let [x,y,z]=model.paths[i];x=(x/15-.5)*2;y=(y/15-.5)*2;z=(z/15-.5)*2;
    const xx=x*Math.cos(yaw)-z*Math.sin(yaw),zz=x*Math.sin(yaw)+z*Math.cos(yaw);
    const yy=y*Math.cos(pitch)-zz*Math.sin(pitch),depth=y*Math.sin(pitch)+zz*Math.cos(pitch);
    const perspective=1/(1+.16*depth),px=cx+xx*rad*perspective,py=cy-yy*rad*perspective;
    points.push({px,py,z:depth,v:value});count++;energy+=value;
   }
   points.sort((a,b)=>b.z-a.z);
   points.forEach(p=>{
    const opacity=.34+.64*Math.max(0,Math.min(1,p.v));let radius=Math.max(.7,(1.4-p.z*.18)*zoom)*(devicePixelRatio||1);
    ctx.fillStyle=t===0?`rgba(230,188,97,${opacity})`:`rgba(${view==='truth'?'125,255,163':'63,220,245'},${opacity})`;
    ctx.beginPath();ctx.arc(p.px,p.py,radius,0,Math.PI*2);ctx.fill();
   });
   ctx.font='bold 14px ui-monospace,monospace';ctx.textAlign='center';ctx.fillStyle=t===0?'#e6bd61':'#42d9f4';
   ctx.fillText(t===0?'INPUT · BIT VOXELS':view.toUpperCase()+' · 3D FIELD',cx,30);
   ctx.font='11px ui-monospace,monospace';ctx.fillStyle='#89a7c3';ctx.fillText(`${count.toLocaleString()} active voxels`,cx,h-20);
  }
  ctx.strokeStyle='rgba(78,169,200,.22)';ctx.beginPath();ctx.moveTo(width/2,45);ctx.lineTo(width/2,height-36);ctx.stroke();
  ctx.textAlign='center';ctx.fillStyle='#e6bd61';ctx.font='20px ui-monospace,monospace';ctx.fillText('→',width/2,height/2);
 }
 function updateAudit(){const a=model.evaluate(audit),p=a.byTask;
  setText('audit',fmt((p.restore.accuracy+p.edge.accuracy)/2*100,2)+'%');
  const d=model.auditOracleFreeXor(audit);
  setText('oracleFree',fmt(d.accuracy*100,2)+'%');
 }
 let trainingBusy=false;
 function gateTraining(steps){if(trainingBusy)return;trainingBusy=true;
  $('train').disabled=true;$('trainLong').disabled=true;setText('gate','TRAINING');
  const before=model.evaluate(validation),snap={w1:new Float64Array(model.w1),b1:new Float64Array(model.b1),w2:new Float64Array(model.w2),b2:model.b2};
  const n=Math.ceil(steps/6);let total=0;
  function batch(){const chunk=Math.min(n,steps-total);if(chunk>0){model.train(training,chunk,48,.12);total+=chunk;setText('gate',`TRAIN ${Math.round(100*total/steps)}%`);setTimeout(batch,0);return;}
   const after=model.evaluate(validation);
   if(after.mse<=before.mse){setText('gate',`ACCEPTED · Δ${(before.mse-after.mse).toFixed(4)}`);model.bestValidation=after.mse;}
   else{model.w1.set(snap.w1);model.b1.set(snap.b1);model.w2.set(snap.w2);model.b2=snap.b2;setText('gate','REJECTED · ROLLED BACK');}
   trainingBusy=false;$('train').disabled=false;$('trainLong').disabled=false;
   refresh();updateAudit();
  }setTimeout(batch,0);
 }
 $('train').addEventListener('click',()=>gateTraining(1200));$('trainLong').addEventListener('click',()=>gateTraining(4200));
 $('new').addEventListener('click',newVolume);
 ['task','view','lam','gam','jump'].forEach(id=>$(id).addEventListener(id==='task'||id==='view'?'change':'input',refresh));
 $('noise').addEventListener('input',()=>setText('noiseVal',fmt(+$('noise').value,2)));
 $('noise').addEventListener('change',newVolume);$('shape').addEventListener('change',newVolume);
 $('orbit').addEventListener('click',()=>{orbit=!orbit;setText('orbit',orbit?'Orbit: ON':'Orbit: OFF');});
 main.addEventListener('pointerdown',ev=>{dragging=true;orbit=false;setText('orbit','Orbit: OFF');lastMouse=[ev.clientX,ev.clientY];main.setPointerCapture(ev.pointerId);});
 main.addEventListener('pointerup',()=>{dragging=false;});
 main.addEventListener('pointermove',ev=>{if(!dragging||!lastMouse)return;yaw+=(ev.clientX-lastMouse[0])*.012;pitch=Math.max(-1.4,Math.min(1.4,pitch+(ev.clientY-lastMouse[1])*.008));lastMouse=[ev.clientX,ev.clientY];render();});
 function resize(){const pixelRatio=Math.min(devicePixelRatio||1,2),r=main.getBoundingClientRect();main.width=Math.max(1,Math.floor(r.width*pixelRatio));main.height=Math.max(1,Math.floor(r.height*pixelRatio));if(last)render();}
 window.addEventListener('resize',resize);
 let then=0;function frame(time){if(orbit&&time-then>40){yaw+=.005;render();then=time;}requestAnimationFrame(frame);}
 resize();refresh();updateAudit();requestAnimationFrame(frame);
})();
