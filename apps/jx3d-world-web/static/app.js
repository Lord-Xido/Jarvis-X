"use strict";
const $ = id => document.getElementById(id);
let current = null;
let bytes = null;
let dim = 32;
let drawnPoints = 0;
let yaw = 0.54, pitch = -0.28, distance = 4.1, orbit = true;
let moving = false, lastX=0, lastY=0;
let positions = new Float32Array(0);
let webgl = null;
let lastFrame = performance.now();

function notify(message, error = false) {
  $('notice').textContent = message;
  $('notice').className = 'notice ' + (error ? 'error' : 'ok');
}
function bytesString(n) {
  if (n < 1024) return `${n} B`;
  if (n < 1048576) return `${(n/1024).toFixed(1)} KiB`;
  return `${(n/1048576).toFixed(2)} MiB`;
}
function decodeBase64(s) {
  const binary = atob(s), b = new Uint8Array(binary.length);
  for (let i=0; i<binary.length; ++i) b[i]=binary.charCodeAt(i);
  return b;
}
function resizeCanvas(canvas) {
  const ratio = Math.min(window.devicePixelRatio || 1,2);
  const w = Math.max(1, Math.round(canvas.clientWidth*ratio));
  const h = Math.max(1, Math.round(canvas.clientHeight*ratio));
  if (canvas.width !== w || canvas.height !== h) { canvas.width=w; canvas.height=h; }
}

function surfacePoints(volume, size, threshold) {
  const points = [];
  const scale = 2.15/(size-1);
  for (let z=0; z<size; ++z) for(let y=0; y<size; ++y) for(let x=0; x<size; ++x) {
    const i=(z*size+y)*size+x, v=volume[i];
    if (v < threshold) continue;
    const edge = x===0 || y===0 || z===0 || x===size-1 || y===size-1 || z===size-1;
    if (!edge && volume[i-1]>=threshold && volume[i+1]>=threshold &&
       volume[i-size]>=threshold && volume[i+size]>=threshold &&
       volume[i-size*size]>=threshold && volume[i+size*size]>=threshold) continue;
    points.push(x*scale-1.075,y*scale-1.075,z*scale-1.075,v/255);
  }
  return new Float32Array(points);
}

function initWebGL() {
  const canvas=$('viewport');
  const gl = canvas.getContext('webgl2', {antialias:true, alpha:false});
  if (!gl) {
    $('render-state').textContent='CANVAS 2D FALLBACK';
    return null;
  }
  const vs=`#version 300 es
precision highp float;
in vec3 aPos; in float aValue;
uniform float uYaw, uPitch, uDist, uAspect, uRadius;
out float vValue;
void main(){
 float cy=cos(uYaw),sy=sin(uYaw),cp=cos(uPitch),sp=sin(uPitch);
 float xx=cy*aPos.x+sy*aPos.z;
 float zz=-sy*aPos.x+cy*aPos.z;
 float yy=cp*aPos.y-sp*zz;
 float depth=sp*aPos.y+cp*zz+uDist;
 float nearPlane=.1, farPlane=40.0;
 gl_Position=vec4(xx*1.8/uAspect, yy*1.8,depth*(farPlane+nearPlane)/(farPlane-nearPlane)-(2.0*farPlane*nearPlane)/(farPlane-nearPlane),depth);
 gl_PointSize=clamp(uRadius/depth,1.5,11.0);
 vValue=aValue;
}`;
  const fs=`#version 300 es
precision highp float;
in float vValue; uniform int uLine; out vec4 frag;
void main(){
 if(uLine==1){frag=vec4(.25,.60,.82,.34);return;}
 vec2 uv=gl_PointCoord*2.0-1.0;
 float r=dot(uv,uv);
 if(r>1.0)discard;
 vec3 a=vec3(.21,.46,.99),b=vec3(.32,.97,.82);
 vec3 color=mix(a,b,clamp(vValue,0.,1.));
 color*=1.12-.37*r;
 frag=vec4(color,.96);
}`;
  function shader(type, source) {
    const sh=gl.createShader(type);gl.shaderSource(sh,source);gl.compileShader(sh);
    if(!gl.getShaderParameter(sh,gl.COMPILE_STATUS))throw new Error(gl.getShaderInfoLog(sh));
    return sh;
  }
  const prog=gl.createProgram();
  gl.attachShader(prog,shader(gl.VERTEX_SHADER,vs));
  gl.attachShader(prog,shader(gl.FRAGMENT_SHADER,fs));
  gl.linkProgram(prog);
  if(!gl.getProgramParameter(prog,gl.LINK_STATUS))throw new Error(gl.getProgramInfoLog(prog));
  const pointBuffer=gl.createBuffer(), lineBuffer=gl.createBuffer();
  const corners=[];
  const verts=[];
  for(let z of [-1.08,1.08])for(let y of [-1.08,1.08])for(let x of [-1.08,1.08]) corners.push([x,y,z]);
  for(let a=0;a<8;a++)for(let bit of [1,2,4]) {
    const b=a^bit; if(a<b) verts.push(...corners[a],0,...corners[b],0);
  }
  gl.bindBuffer(gl.ARRAY_BUFFER,lineBuffer);
  gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(verts),gl.STATIC_DRAW);
  return {gl, prog, pointBuffer, lineBuffer, lineCount:verts.length/4,
    aPos:gl.getAttribLocation(prog,'aPos'),aValue:gl.getAttribLocation(prog,'aValue'),
    yaw:gl.getUniformLocation(prog,'uYaw'),pitch:gl.getUniformLocation(prog,'uPitch'),
    dist:gl.getUniformLocation(prog,'uDist'),aspect:gl.getUniformLocation(prog,'uAspect'),
    radius:gl.getUniformLocation(prog,'uRadius'),line:gl.getUniformLocation(prog,'uLine')};
}
function setGeometry() {
  if(!bytes) return;
  positions=surfacePoints(bytes,dim,Number($('threshold').value));
  drawnPoints=positions.length/4;
  $('point-count').textContent=drawnPoints.toLocaleString()+' surface samples';
  if(webgl) {
    const gl=webgl.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER,webgl.pointBuffer);
    gl.bufferData(gl.ARRAY_BUFFER,positions,gl.STATIC_DRAW);
  }
}
function bindAttributes(buf) {
  const {gl,aPos,aValue}=webgl;
  gl.bindBuffer(gl.ARRAY_BUFFER,buf);
  gl.enableVertexAttribArray(aPos);
  gl.vertexAttribPointer(aPos,3,gl.FLOAT,false,16,0);
  gl.enableVertexAttribArray(aValue);
  gl.vertexAttribPointer(aValue,1,gl.FLOAT,false,16,12);
}
function renderWebGL() {
  const {gl,prog}=webgl;
  const canvas=$('viewport');resizeCanvas(canvas);
  gl.viewport(0,0,canvas.width,canvas.height);
  gl.clearColor(.02,.065,.12,1);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
  gl.enable(gl.DEPTH_TEST);gl.enable(gl.BLEND);gl.blendFunc(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA);
  gl.useProgram(prog);
  gl.uniform1f(webgl.yaw,yaw);gl.uniform1f(webgl.pitch,pitch);gl.uniform1f(webgl.dist,distance);
  gl.uniform1f(webgl.aspect,canvas.width/canvas.height);gl.uniform1f(webgl.radius,19);
  bindAttributes(webgl.lineBuffer);
  gl.uniform1i(webgl.line,1);gl.drawArrays(gl.LINES,0,webgl.lineCount);
  if(drawnPoints>0) {
    bindAttributes(webgl.pointBuffer);
    gl.uniform1i(webgl.line,0);gl.drawArrays(gl.POINTS,0,drawnPoints);
  }
}
function fallbackCanvas() {
  const canvas=$('viewport'), ctx=canvas.getContext('2d');
  if (!ctx) return;
  resizeCanvas(canvas);
  const w=canvas.width,h=canvas.height;
  ctx.fillStyle='#081829';ctx.fillRect(0,0,w,h);
  const cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch);
  const points=[];
  for(let i=0;i<positions.length;i+=4){
    const x=positions[i],y=positions[i+1],z=positions[i+2];
    const xx=cy*x+sy*z,zz=-sy*x+cy*z;
    const yy=cp*y-sp*zz,d=sp*y+cp*zz+distance;
    points.push({x:w/2+xx*1.8/d*w/2,y:h/2-yy*1.8/d*h/2,d,v:positions[i+3]});
  }
  points.sort((a,b)=>b.d-a.d);
  for(const p of points){ctx.beginPath();ctx.fillStyle=`rgba(${Math.round(45+44*p.v)},${Math.round(115+130*p.v)},${Math.round(240-27*p.v)},.83)`;ctx.arc(p.x,p.y,Math.max(1,Math.min(4,8/p.d)),0,2*Math.PI);ctx.fill();}
}
function tick(t) {
  const dt=Math.min(100,t-lastFrame);lastFrame=t;
  if(orbit&&!moving)yaw += dt*.00019;
  if(webgl)renderWebGL();else fallbackCanvas();
  requestAnimationFrame(tick);
}
function configureOrbit() {
  const c=$('viewport');
  c.addEventListener('pointerdown',e=>{moving=true;lastX=e.clientX;lastY=e.clientY;c.setPointerCapture(e.pointerId)});
  c.addEventListener('pointermove',e=>{if(!moving)return;yaw+=(e.clientX-lastX)*.009;pitch=Math.max(-1.4,Math.min(1.4,pitch+(e.clientY-lastY)*.008));lastX=e.clientX;lastY=e.clientY});
  for(const event of ['pointerup','pointercancel','lostpointercapture'])c.addEventListener(event,()=>moving=false);
  c.addEventListener('wheel',e=>{e.preventDefault();distance=Math.max(2.2,Math.min(9,distance+e.deltaY*.004))},{passive:false});
  $('spin').onclick=()=>{orbit=!orbit;$('spin').textContent=orbit?'⏸ Pause orbit':'▶ Resume orbit'};
  $('reset-view').onclick=()=>{yaw=.54;pitch=-.28;distance=4.1};
}
function drawSlice() {
  if(!bytes) return;
  const canvas=$('slice'),ctx=canvas.getContext('2d');
  const z=Number($('slice-pos').value);$('slice-label').textContent=z;
  const img=ctx.createImageData(dim,dim);
  for(let y=0;y<dim;y++)for(let x=0;x<dim;x++){
    const v=bytes[(z*dim+y)*dim+x]/255,off=(y*dim+x)*4;
    img.data[off]=Math.floor(16+55*v);
    img.data[off+1]=Math.floor(29+211*v);
    img.data[off+2]=Math.floor(52+168*v);
    img.data[off+3]=255;
  }
  const temp=document.createElement('canvas');temp.width=dim;temp.height=dim;
  temp.getContext('2d').putImageData(img,0,0);
  ctx.imageSmoothingEnabled=false;
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.drawImage(temp,0,0,canvas.width,canvas.height);
}
function drawChart(rounds) {
  const canvas=$('chart'),ctx=canvas.getContext('2d');
  resizeCanvas(canvas);const w=canvas.width,h=canvas.height;
  ctx.clearRect(0,0,w,h);
  ctx.fillStyle='#091a2e';ctx.fillRect(0,0,w,h);
  ctx.strokeStyle='#264663';ctx.lineWidth=1;
  for(let k=1;k<=3;k++){const y=k*h/4;ctx.beginPath();ctx.moveTo(15,y);ctx.lineTo(w-12,y);ctx.stroke();}
  if(!rounds.length){$('chart-label').textContent='Awaiting data';return;}
  $('chart-label').textContent=`${rounds.length} completed cycles`;
  const min=Math.min(...rounds.map(p=>p.ratio)),max=Math.max(...rounds.map(p=>p.ratio));
  const span=Math.max(.15,max-min);
  const p=i=>({x:22+i*(w-44)/Math.max(1,rounds.length-1),y:h-20-(rounds[i].ratio-min+span*.05)/(span*1.1)*(h-36)});
  ctx.strokeStyle='#49e3d6';ctx.lineWidth=3;ctx.beginPath();
  rounds.forEach((r,i)=>{const q=p(i);if(i===0)ctx.moveTo(q.x,q.y);else ctx.lineTo(q.x,q.y)});ctx.stroke();
  ctx.font=`${Math.max(10,Math.floor(w/65))}px monospace`;
  for(let i=0;i<rounds.length;i++){
    const q=p(i);ctx.fillStyle='#a8f6ed';ctx.beginPath();ctx.arc(q.x,q.y,4,0,Math.PI*2);ctx.fill();
    ctx.fillStyle='#91aec6';ctx.fillText(`${i+1}`,q.x-3,h-5);
  }
}
function show(result, message) {
  if(!result || !result.volume_b64)throw new Error('Missing decoded field');
  current=result;bytes=decodeBase64(result.volume_b64);dim=result.dimension;
  if(bytes.length !== dim*dim*dim)throw new Error('Malformed field length');
  $('error').textContent=result.mse===0?'0.0000':Number(result.mse).toFixed(3);
  $('integrity').textContent=result.mse===0?'Exact reconstruction':`MAE ${result.mae} / 255`;
  $('compressed').textContent=bytesString(result.encoded_bytes);
  $('raw').textContent='Raw memory: '+bytesString(result.raw_bytes);
  $('ratio').textContent=result.compression_ratio.toFixed(2)+'×';
  $('passes').textContent=result.rounds?.length||1;
  $('verified').textContent=result.checksum_verified?'SHA-256 verified':'Verification failed';
  $('digest').textContent=result.sha256;
  $('download').disabled=false;
  $('slice-pos').max=dim-1;$('slice-pos').value=Math.floor(dim/2);
  $('threshold-label').textContent=$('threshold').value;
  setGeometry();drawSlice();drawChart(result.rounds||[]);
  const rows=$('ledger');rows.replaceChildren();
  const history=result.rounds?.length?result.rounds:[{cycle:1,bytes:result.encoded_bytes,ratio:result.compression_ratio,mse:result.mse,duration_ms:'—',sha256:result.sha256,verified:result.checksum_verified}];
  for(const r of history){
    const tr=document.createElement('tr');
    for(const text of [r.cycle,bytesString(r.bytes),r.ratio+'×',r.mse,r.duration_ms==='—'?'—':r.duration_ms+' ms',r.sha256.slice(0,12)+'…',r.verified?'✓ PASS':'✗ FAIL']) {
      const td=document.createElement('td');td.textContent=text;tr.append(td);
    }
    rows.append(tr);
  }
  notify(message || 'Decoded voxel field passed its SHA-256 integrity verification.');
}
async function jsonRequest(path,options) {
  const response=await fetch(path,options);
  const data=await response.json().catch(()=>({error:'Invalid JSON response'}));
  if(!response.ok)throw new Error(data.error||`HTTP ${response.status}`);
  return data;
}
async function run() {
  const button=$('run'),label=$('run-label');
  button.disabled=true;document.body.classList.add('busy');label.textContent='Executing real 3D cycles…';
  try {
    const result=await jsonRequest('/api/simulate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({
      dimension:Number($('dimension').value),geometry:$('geometry').value,
      mode:$('mode').value,seed:Number($('seed').value),cycles:Number($('cycles').value),
      alpha:Number($('alpha').value)
    })});
    show(result,`${result.rounds.length} recursive encoding/decoding cycles completed. Reconstructed state verified.`);
  }catch(e){notify(e.message,true);}finally{button.disabled=false;document.body.classList.remove('busy');label.textContent='▶ Execute 3D pipeline';}
}
async function loadFile(file) {
  const raw=new Uint8Array(await file.arrayBuffer());
  const isPacket=file.name.toLowerCase().endsWith('.jx3d');
  const path=isPacket?'/api/decode':'/api/encode';
  const headers={'Content-Type':'application/octet-stream'};
  if(!isPacket){headers['X-JX3D-Dimension']=$('dimension').value;headers['X-JX3D-Mode']=$('mode').value;}
  const result=await jsonRequest(path,{method:'POST',headers,body:raw});
  show(result,(isPacket?'Decoded and verified packet: ':'Encoded raw voxels: ')+file.name);
}
async function connect() {
  try {const r=await jsonRequest('/api/health');$('pulse').classList.add('online');$('connection').textContent=`${r.engine} · API CONNECTED`;
  } catch(e) {$('connection').textContent='RUNTIME UNAVAILABLE';notify('Cannot reach server: '+e.message,true);}
}
async function init() {
  $('cycles').oninput=()=>$('cycles-label').textContent=$('cycles').value;
  $('alpha').oninput=()=>$('alpha-label').textContent=Number($('alpha').value).toFixed(2);
  $('threshold').oninput=()=>{$('threshold-label').textContent=$('threshold').value;setGeometry()};
  $('slice-pos').oninput=drawSlice;
  $('run').onclick=run;
  $('download').onclick=()=>{if(current)window.location.href='/api/download/'+encodeURIComponent(current.encoded_id)};
  $('file').onchange=async e=>{
    const f=e.target.files?.[0];if(!f)return;
    try {notify('Reading '+f.name+'…');await loadFile(f);}catch(err){notify(err.message,true);}
    e.target.value='';
  };
  configureOrbit();
  try{webgl=initWebGL()}catch(e){console.warn('WebGL2 initialization failed:',e);webgl=null;$('render-state').textContent='CANVAS 2D FALLBACK'}
  requestAnimationFrame(tick);
  await connect();
  await run();
}
window.addEventListener('DOMContentLoaded',init);
