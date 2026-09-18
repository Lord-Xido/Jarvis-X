(function () {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const sdk = new AE3D.AE3DSDK({ recursionDepth: 4, beta: 0.85, foldGain: 0.06, learningRate: 0.02 });
  window.sdk = sdk;
  window.engine = sdk.engine;

  let scene, camera, renderer, controls, nodes, nodeGroup;
  const dummy = new THREE.Object3D();
  const cold = new THREE.Color(0x0ea5e9), hot = new THREE.Color(0x10b981), idle = new THREE.Color(0x1e293b);

  function log(...args) {
    const line = args.map(v => typeof v === 'string' ? v : JSON.stringify(v, null, 2)).join(' ');
    $('console').textContent += line + '\n';
    $('console').scrollTop = $('console').scrollHeight;
  }
  window.print = log;

  function setup3D() {
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x020617);
    scene.fog = new THREE.FogExp2(0x020617, 0.018);
    camera = new THREE.PerspectiveCamera(50, 1, 0.1, 500);
    camera.position.set(0, 20, 34);
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    $('view3d').appendChild(renderer.domElement);
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;

    scene.add(new THREE.AmbientLight(0xffffff, 0.7));
    const key = new THREE.DirectionalLight(0x67e8f9, 1.2); key.position.set(20,30,20); scene.add(key);
    const fill = new THREE.PointLight(0xa855f7, 1.5, 70); fill.position.set(-12,-6,-8); scene.add(fill);

    const geo = new THREE.SphereGeometry(0.28, 10, 10);
    const mat = new THREE.MeshStandardMaterial({ roughness: 0.28, metalness: 0.65 });
    nodes = new THREE.InstancedMesh(geo, mat, sdk.engine.voxels);
    nodeGroup = new THREE.Group();
    nodeGroup.add(nodes);
    scene.add(nodeGroup);

    const R = 10, r = 4.2;
    for (let p = 0; p < sdk.engine.voxels; p++) {
      const [x,y,z] = sdk.engine.coords(p);
      const u = 2*Math.PI * ((x + y/6) / 6);
      const v = 2*Math.PI * (z / 6) + y * 0.09;
      dummy.position.set((R+r*Math.cos(v))*Math.cos(u), r*Math.sin(v), (R+r*Math.cos(v))*Math.sin(u));
      dummy.scale.setScalar(0.85);
      dummy.updateMatrix();
      nodes.setMatrixAt(p, dummy.matrix);
      nodes.setColorAt(p, idle);
    }
    nodes.instanceMatrix.needsUpdate = true;
    if (nodes.instanceColor) nodes.instanceColor.needsUpdate = true;

    const wire = new THREE.Mesh(
      new THREE.TorusGeometry(R, r, 18, 72),
      new THREE.MeshBasicMaterial({ color: 0x22d3ee, wireframe: true, transparent: true, opacity: 0.08 })
    );
    wire.rotation.x = Math.PI/2;
    nodeGroup.add(wire);

    resize3D();
    new ResizeObserver(resize3D).observe($('view3d'));
    requestAnimationFrame(render3D);
  }

  function resize3D() {
    if (!renderer) return;
    const el = $('view3d');
    const w = Math.max(1, el.clientWidth), h = Math.max(1, el.clientHeight);
    renderer.setSize(w,h,false);
    camera.aspect = w/h;
    camera.updateProjectionMatrix();
  }

  function render3D() {
    requestAnimationFrame(render3D);
    controls.update();
    nodeGroup.rotation.y += 0.0015;
    renderer.render(scene,camera);
  }

  function update3D() {
    let maxE = 1e-9;
    for (const e of sdk.engine.voxelEnergy) if (e > maxE) maxE = e;
    for (let p=0; p<sdk.engine.voxels; p++) {
      const active = sdk.engine.activeMask[p] > 0;
      const t = active ? Math.min(1, sdk.engine.voxelEnergy[p] / maxE) : 0;
      const c = active ? cold.clone().lerp(hot, t) : idle;
      nodes.setColorAt(p,c);
    }
    if (nodes.instanceColor) nodes.instanceColor.needsUpdate = true;
  }

  function fmt(v, digits=4) { return Number.isFinite(v) ? Number(v).toFixed(digits) : '∞'; }

  function updateHUD(result) {
    const m = result.metrics;
    $('m-acc').textContent = (100*m.byteAccuracy).toFixed(2)+'%';
    $('m-ber').textContent = m.ber.toExponential(2);
    $('m-mse').textContent = fmt(m.mse,5);
    $('m-psnr').textContent = Number.isFinite(m.psnr) ? m.psnr.toFixed(2)+' dB' : '∞ dB';
    $('m-active').textContent = `${m.activeVoxels}/${sdk.engine.voxels}`;
    $('m-stability').textContent = (100*m.stability).toFixed(2)+'%';
    $('m-qmse').textContent = m.quantMSE.toExponential(2);
    $('m-time').textContent = m.elapsedMs.toFixed(3)+' ms';
    $('m-rate').textContent = Math.round(m.bytesPerSecond).toLocaleString()+' B/s';
    $('m-clock').textContent = result.clock.toLocaleString();
    $('hex-in').textContent = sdk.hex(result.input);
    $('hex-out').textContent = sdk.hex(result.output);
    $('text-out').textContent = sdk.decodeText(result.output);
    update3D();
  }

  function runEngine(depthOverride) {
    const text = $('input-text').value;
    sdk.configure({
      beta: Number($('beta').value),
      foldGain: Number($('fold').value),
      recursionDepth: Number($('depth').value)
    });
    const result = sdk.run(text, depthOverride ? {recursionDepth: depthOverride} : undefined);
    updateHUD(result);
    return result;
  }

  async function runEditor() {
    const code = $('editor').value;
    const AsyncFunction = Object.getPrototypeOf(async function(){}).constructor;
    try {
      const fn = new AsyncFunction('sdk','engine','print', code);
      await fn(sdk, sdk.engine, log);
      log('✓ script completed');
    } catch (err) {
      log('✗', err.stack || String(err));
    }
  }

  function exportModel() {
    const blob = new Blob([JSON.stringify(sdk.exportModel())], {type:'application/json'});
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'ae3d-model.json';
    a.click();
    setTimeout(()=>URL.revokeObjectURL(a.href), 1000);
  }

  function bindUI() {
    $('btn-run').onclick = () => runEngine();
    $('btn-step').onclick = () => runEngine(1);
    $('btn-turbo').onclick = () => {
      const result = runEngine(1000);
      log(`1000-pass macro test: ${result.metrics.elapsedMs.toFixed(2)} ms, accuracy ${(100*result.metrics.byteAccuracy).toFixed(2)}%`);
    };
    $('btn-train').onclick = () => {
      const r = sdk.train($('input-text').value, {epochs: 10});
      log(`trained: epochs=${r.epochs} loss=${r.loss.toFixed(5)} time=${r.elapsedMs.toFixed(2)}ms`);
      runEngine();
    };
    $('btn-reset').onclick = () => { sdk.engine.reset(); log('engine reset'); runEngine(); };
    $('btn-script').onclick = runEditor;
    $('btn-clear-console').onclick = () => $('console').textContent = '';
    $('btn-export').onclick = exportModel;
    ['beta','fold','depth'].forEach(id => $(id).oninput = () => {
      $('beta-v').textContent = Number($('beta').value).toFixed(2);
      $('fold-v').textContent = Number($('fold').value).toFixed(2);
      $('depth-v').textContent = $('depth').value;
    });
  }

  window.addEventListener('load', () => {
    setup3D();
    bindUI();
    $('editor').value = `// AE-3D SDK script\nconst r = sdk.run("AUTOENCODER_3D_BYTES");\nprint("decoded:", sdk.decodeText(r.output));\nprint("metrics:", r.metrics);\n\n// Train the byte codebook on the current phrase:\nconst t = sdk.train("AUTOENCODER_3D_BYTES", { epochs: 3 });\nprint("training:", t);\n`;
    log('AE-3D IDE ready');
    runEngine();
  });
})();
