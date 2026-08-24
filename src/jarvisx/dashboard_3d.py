"""Dependency-free WebGL dashboard for the operational 3D runtime."""

DASHBOARD_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Dr Moagi 3D Auto-Codec Runtime</title>
<style>
:root { color-scheme: dark; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
* { box-sizing:border-box; }
body { margin:0; background:#070a11; color:#e8eefc; }
main { max-width:1320px; margin:auto; padding:24px; }
.grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:10px; }
.layout { display:grid; grid-template-columns:minmax(0,1.7fr) minmax(300px,.8fr); gap:12px; margin-top:12px; }
.card { background:#101623; border:1px solid #27344f; border-radius:12px; padding:14px; overflow:hidden; }
label { display:block; font-size:11px; color:#91a0bc; margin-bottom:5px; }
input, textarea, button { width:100%; border-radius:8px; border:1px solid #344362; background:#09101f; color:#eef4ff; padding:9px; }
textarea { min-height:145px; resize:vertical; }
button { cursor:pointer; background:#162e58; font-weight:700; }
button:hover { background:#1d3d74; }
button:disabled { opacity:.55; cursor:default; }
.metric { font-size:20px; font-weight:700; overflow-wrap:anywhere; }
.muted { color:#91a0bc; font-size:11px; line-height:1.5; }
.status { display:inline-block; padding:4px 8px; border-radius:999px; border:1px solid #344362; }
canvas { display:block; width:100%; height:520px; border-radius:9px; background:#050811; touch-action:none; }
pre { white-space:pre-wrap; overflow-wrap:anywhere; max-height:330px; overflow:auto; font-size:11px; }
.controls { display:grid; grid-template-columns:100px 1fr 80px; gap:8px; align-items:center; margin-top:8px; }
#frame { width:100%; }
@media(max-width:900px){ .layout{grid-template-columns:1fr;} canvas{height:420px;} }
</style>
</head>
<body>
<main>
  <h1>Dr Moagi 3D Auto-Encoding / Decoding Runtime</h1>
  <p class="muted">Authoritative path: 3D sparse field → Morton spatial latent → decode → residual → 3D field operator → verify → commit/rollback → Omega journal → persisted run → next frame.</p>

  <div class="grid">
    <div class="card"><label>Quantization step</label><input id="step" type="number" step="0.01" value="0.10"></div>
    <div class="card"><label>Alpha · reconstruction closure</label><input id="alpha" type="number" step="0.1" value="1.0"></div>
    <div class="card"><label>Lambda · residual Laplacian</label><input id="lambda" type="number" step="0.05" value="0.0"></div>
    <div class="card"><label>Eta · 3D glyph coupling</label><input id="eta" type="number" step="0.05" value="0.0"></div>
    <div class="card"><label>dt</label><input id="dt" type="number" step="0.01" value="0.10"></div>
    <div class="card"><label>Max cycles</label><input id="cycles" type="number" value="64"></div>
    <div class="card"><label>MSE target</label><input id="target" type="number" step="0.0001" value="0.001"></div>
  </div>

  <div class="layout">
    <section>
      <div class="card">
        <canvas id="view" width="1000" height="620"></canvas>
        <div class="controls">
          <button id="play">Play</button>
          <input id="frame" type="range" min="0" max="0" value="0" />
          <span id="frameLabel" class="muted">0 / 0</span>
        </div>
        <p class="muted">Drag to orbit · wheel/pinch to zoom · Play traverses authoritative runtime frames.</p>
      </div>
      <div class="grid" style="margin-top:10px">
        <div class="card"><div class="muted">Stop reason</div><div id="stop" class="metric">—</div></div>
        <div class="card"><div class="muted">Cycles</div><div id="cycleCount" class="metric">0</div></div>
        <div class="card"><div class="muted">Final MSE</div><div id="mse" class="metric">—</div></div>
        <div class="card"><div class="muted">Active 3D cells</div><div id="active" class="metric">0</div></div>
        <div class="card"><div class="muted">Six-face links</div><div id="links" class="metric">0</div></div>
        <div class="card"><div class="muted">Spatial RMS radius</div><div id="radius" class="metric">—</div></div>
        <div class="card"><div class="muted">Latent entries</div><div id="latent" class="metric">0</div></div>
        <div class="card"><div class="muted">Omega journal</div><div id="journal" class="status">not run</div></div>
      </div>
    </section>

    <aside>
      <div class="card">
        <label>Sparse 3D input cells</label>
        <textarea id="cells">[
  {"x": 20, "y": 20, "z": 20, "value": 0.86},
  {"x": 21, "y": 20, "z": 20, "value": 0.62},
  {"x": 20, "y": 21, "z": 20, "value": -0.55},
  {"x": 20, "y": 20, "z": 21, "value": 0.44},
  {"x": 21, "y": 21, "z": 21, "value": -0.31}
]</textarea>
        <button id="run" style="margin-top:9px">Execute 3D closed loop</button>
      </div>
      <div class="card" style="margin-top:10px">
        <div class="muted">Persisted run ID</div>
        <div id="runId" class="metric">—</div>
        <button id="verify" style="margin-top:9px" disabled>Verify persisted run</button>
      </div>
      <div class="card" style="margin-top:10px">
        <div class="muted">3D geometry</div>
        <pre id="geometry">Ready.</pre>
      </div>
      <div class="card" style="margin-top:10px">
        <div class="muted">Runtime receipt</div>
        <pre id="result">Ready.</pre>
      </div>
    </aside>
  </div>
</main>
<script>
const $ = (id) => document.getElementById(id);
let runtimeFrames = [], frameIndex = 0, playing = false, lastPlayback = 0;
let yaw = -0.7, pitch = 0.45, zoom = 1.0, dragging = false, lastX = 0, lastY = 0;
let currentRunId = null;
const canvas = $('view');
const gl = canvas.getContext('webgl', {antialias:true, alpha:false});
if (!gl) $('result').textContent = 'WebGL unavailable; API execution remains usable.';
let program, positionBuffer, valueBuffer, aPosition, aValue, uYaw, uPitch, uZoom, uCenter, uScale;
function shader(type, source) {
  const item = gl.createShader(type); gl.shaderSource(item, source); gl.compileShader(item);
  if (!gl.getShaderParameter(item, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(item));
  return item;
}
function initGL() {
  if (!gl) return;
  const vs = shader(gl.VERTEX_SHADER, `
    attribute vec3 a_position; attribute float a_value;
    uniform float u_yaw, u_pitch, u_zoom, u_scale; uniform vec3 u_center;
    varying float v_value;
    void main(){
      vec3 p=(a_position-u_center)*u_scale;
      float cy=cos(u_yaw), sy=sin(u_yaw), cp=cos(u_pitch), sp=sin(u_pitch);
      p=vec3(cy*p.x+sy*p.z,p.y,-sy*p.x+cy*p.z);
      p=vec3(p.x,cp*p.y-sp*p.z,sp*p.y+cp*p.z);
      float depth=max(1.0,4.0+p.z);
      gl_Position=vec4(p.x*u_zoom*1.8,p.y*u_zoom*1.8,depth-2.0,depth);
      gl_PointSize=min(18.0,max(4.0,7.0+abs(a_value)*7.0)); v_value=a_value;
    }`);
  const fs = shader(gl.FRAGMENT_SHADER, `
    precision mediump float; varying float v_value;
    void main(){ vec2 d=gl_PointCoord-vec2(.5); if(dot(d,d)>.25) discard;
      float a=min(1.0,abs(v_value)); vec3 p=vec3(.20,.78,1.0), n=vec3(1.0,.30,.68);
      vec3 c=mix(vec3(.72,.78,.90),v_value>=0.0?p:n,.45+.55*a); gl_FragColor=vec4(c,1.0); }`);
  program=gl.createProgram(); gl.attachShader(program,vs); gl.attachShader(program,fs); gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
  aPosition=gl.getAttribLocation(program,'a_position'); aValue=gl.getAttribLocation(program,'a_value');
  uYaw=gl.getUniformLocation(program,'u_yaw'); uPitch=gl.getUniformLocation(program,'u_pitch'); uZoom=gl.getUniformLocation(program,'u_zoom');
  uCenter=gl.getUniformLocation(program,'u_center'); uScale=gl.getUniformLocation(program,'u_scale');
  positionBuffer=gl.createBuffer(); valueBuffer=gl.createBuffer(); gl.enable(gl.DEPTH_TEST); gl.clearColor(.018,.026,.055,1);
}
function resizeCanvas(){ const dpr=Math.min(2,window.devicePixelRatio||1), w=Math.max(1,Math.floor(canvas.clientWidth*dpr)), h=Math.max(1,Math.floor(canvas.clientHeight*dpr)); if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h;} }
function currentFrame(){ return runtimeFrames[frameIndex] || {points:[],metrics:{}}; }
function draw(){
  if (!gl || !program) return; resizeCanvas();
  const frame=currentFrame(), pts=frame.points||[], positions=new Float32Array(pts.length*3), values=new Float32Array(pts.length);
  for(let i=0;i<pts.length;i++){ positions[i*3]=pts[i].x; positions[i*3+1]=pts[i].y; positions[i*3+2]=pts[i].z; values[i]=pts[i].value; }
  const m=frame.metrics||{}, lo=m.bounds_min||[0,0,0], hi=m.bounds_max||[1,1,1], center=[(lo[0]+hi[0])/2,(lo[1]+hi[1])/2,(lo[2]+hi[2])/2];
  const span=Math.max(1,hi[0]-lo[0],hi[1]-lo[1],hi[2]-lo[2]);
  gl.viewport(0,0,canvas.width,canvas.height); gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT); gl.useProgram(program);
  gl.bindBuffer(gl.ARRAY_BUFFER,positionBuffer); gl.bufferData(gl.ARRAY_BUFFER,positions,gl.DYNAMIC_DRAW); gl.enableVertexAttribArray(aPosition); gl.vertexAttribPointer(aPosition,3,gl.FLOAT,false,0,0);
  gl.bindBuffer(gl.ARRAY_BUFFER,valueBuffer); gl.bufferData(gl.ARRAY_BUFFER,values,gl.DYNAMIC_DRAW); gl.enableVertexAttribArray(aValue); gl.vertexAttribPointer(aValue,1,gl.FLOAT,false,0,0);
  gl.uniform1f(uYaw,yaw); gl.uniform1f(uPitch,pitch); gl.uniform1f(uZoom,zoom); gl.uniform3fv(uCenter,new Float32Array(center)); gl.uniform1f(uScale,1.6/span); gl.drawArrays(gl.POINTS,0,pts.length);
}
function selectFrame(index){
  frameIndex=Math.max(0,Math.min(index,Math.max(0,runtimeFrames.length-1))); $('frame').value=String(frameIndex); $('frameLabel').textContent=`${frameIndex} / ${Math.max(0,runtimeFrames.length-1)}`;
  const f=currentFrame(), m=f.metrics||{}; $('active').textContent=m.active_cells ?? 0; $('links').textContent=m.six_face_links ?? 0; $('radius').textContent=m.rms_radius==null?'—':Number(m.rms_radius).toFixed(4);
  $('geometry').textContent=JSON.stringify({cycle:f.cycle,state_digest:f.state_digest,metrics:m},null,2); draw();
}
function animate(t){ if(playing&&runtimeFrames.length>1&&t-lastPlayback>220){frameIndex=(frameIndex+1)%runtimeFrames.length;selectFrame(frameIndex);lastPlayback=t;} draw(); requestAnimationFrame(animate); }
canvas.addEventListener('pointerdown',e=>{dragging=true;lastX=e.clientX;lastY=e.clientY;canvas.setPointerCapture(e.pointerId);});
canvas.addEventListener('pointermove',e=>{if(!dragging)return;yaw+=(e.clientX-lastX)*.01;pitch+=(e.clientY-lastY)*.01;pitch=Math.max(-1.45,Math.min(1.45,pitch));lastX=e.clientX;lastY=e.clientY;});
canvas.addEventListener('pointerup',()=>dragging=false); canvas.addEventListener('pointercancel',()=>dragging=false);
canvas.addEventListener('wheel',e=>{e.preventDefault();zoom*=Math.exp(-e.deltaY*.001);zoom=Math.max(.25,Math.min(4,zoom));},{passive:false});
$('play').addEventListener('click',()=>{playing=!playing;$('play').textContent=playing?'Pause':'Play';});
$('frame').addEventListener('input',()=>{playing=false;$('play').textContent='Play';selectFrame(Number($('frame').value));});
$('verify').addEventListener('click',async()=>{ if(!currentRunId)return; try{ const r=await fetch(`/codec/3d/runs/${currentRunId}/verify`), d=await r.json(); if(!r.ok)throw new Error(d.detail||JSON.stringify(d)); $('result').textContent=JSON.stringify(d,null,2); }catch(e){$('result').textContent=String(e);} });
$('run').addEventListener('click',async()=>{
  $('run').disabled=true; $('result').textContent='Executing and persisting 3D loop…'; playing=false; $('play').textContent='Play';
  try{
    const body={cells:JSON.parse($('cells').value),side:64,quantization_step:Number($('step').value),alpha:Number($('alpha').value),lambda_residual:Number($('lambda').value),eta:Number($('eta').value),dt:Number($('dt').value),expand_halo:true,max_cycles:Number($('cycles').value),reconstruction_mse_target:Number($('target').value),stop_on_fixed_point:true,frame_stride:1,max_render_points:4096,max_frames:128,persist:true};
    const response=await fetch('/codec/3d/run',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(body)}), data=await response.json(); if(!response.ok)throw new Error(data.detail||JSON.stringify(data));
    runtimeFrames=data.frames||[]; frameIndex=Math.max(0,runtimeFrames.length-1); $('frame').max=String(Math.max(0,runtimeFrames.length-1)); $('stop').textContent=data.stop_reason; $('cycleCount').textContent=data.cycles_executed; $('mse').textContent=data.final_reconstruction_mse==null?'—':Number(data.final_reconstruction_mse).toExponential(3); $('latent').textContent=data.latent_entries; $('journal').textContent=data.journal_verified?`verified · ${data.journal_entries} receipts`:'verification failed';
    currentRunId=data.run_id||null; $('runId').textContent=currentRunId||'not persisted'; $('verify').disabled=!currentRunId;
    $('result').textContent=JSON.stringify({run_id:data.run_id,persisted:data.persisted,spatial_mode:data.spatial_mode,codec:data.codec,stop_reason:data.stop_reason,cycles_executed:data.cycles_executed,converged:data.converged,final_reconstruction_mse:data.final_reconstruction_mse,latent_entries:data.latent_entries,packed_latent_bytes_estimate:data.packed_latent_bytes_estimate,latent_digest:data.latent_digest,final_state_digest:data.final_state_digest,journal_verified:data.journal_verified,journal_head_hash:data.journal_head_hash},null,2); selectFrame(frameIndex);
  }catch(error){$('result').textContent=String(error);}finally{$('run').disabled=false;}
});
try{initGL();requestAnimationFrame(animate);}catch(error){$('result').textContent=String(error);}
</script>
</body>
</html>"""
