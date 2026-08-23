"""Three.js enterprise browser control plane for the Dr Moagi 3D OS service."""

DR_MOAGI_OS_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>JARVIS-X OS • 3D Enterprise Sparse Runtime</title>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
  <script src="https://unpkg.com/lucide@latest"></script>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    :root {
      color-scheme: dark;
      --bg: #020617;
      --panel: rgba(15, 23, 42, 0.88);
      --panel-solid: #0f172a;
      --cyan: #38bdf8;
      --purple: #c084fc;
      --green: #4ade80;
      --red: #fb7185;
      --text: #f8fafc;
      --muted: #94a3b8;
      --line: rgba(56, 189, 248, 0.22);
    }
    html, body { width: 100%; height: 100%; background: var(--bg); color: var(--text); }
    body {
      overflow: hidden;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    button, input, select { font: inherit; }
    #app-container {
      width: 100vw;
      height: 100vh;
      display: grid;
      grid-template-rows: 42px 1fr 28px;
      grid-template-columns: 320px minmax(0, 1fr) 370px;
      grid-template-areas:
        "header header header"
        "sidebar viewport chat"
        "statusbar statusbar statusbar";
    }
    .glass-panel {
      background: var(--panel);
      backdrop-filter: blur(16px);
      -webkit-backdrop-filter: blur(16px);
      z-index: 10;
    }
    #os-header {
      grid-area: header;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 0 16px;
      border-bottom: 1px solid var(--line);
      background: rgba(15, 23, 42, 0.97);
      overflow: hidden;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 10px;
      white-space: nowrap;
      font-size: .9rem;
      font-weight: 800;
      color: var(--cyan);
      letter-spacing: .08em;
    }
    .badge {
      font-size: .62rem;
      background: rgba(168, 85, 247, .2);
      color: var(--purple);
      padding: 2px 8px;
      border-radius: 4px;
      border: 1px solid rgba(168, 85, 247, .4);
      font-weight: 800;
    }
    .os-status-pills { display: flex; gap: 8px; font-size: .68rem; color: var(--muted); overflow: hidden; }
    .pill {
      display: flex;
      align-items: center;
      gap: 6px;
      white-space: nowrap;
      background: rgba(30, 41, 59, .65);
      padding: 3px 9px;
      border-radius: 12px;
      border: 1px solid rgba(255,255,255,.05);
    }
    .pill-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--muted); }
    .pill-dot.ready { background: var(--green); box-shadow: 0 0 8px rgba(74,222,128,.8); }
    .pill-dot.halted { background: var(--red); box-shadow: 0 0 8px rgba(251,113,133,.8); }

    #os-sidebar {
      grid-area: sidebar;
      display: flex;
      flex-direction: column;
      gap: 13px;
      padding: 14px;
      border-right: 1px solid var(--line);
      overflow-y: auto;
    }
    .section-title {
      font-size: .68rem;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: .1em;
      color: var(--cyan);
      margin-bottom: 7px;
      display: flex;
      justify-content: space-between;
      gap: 10px;
      align-items: center;
    }
    .section-note { font-size: .58rem; color: #64748b; text-transform: none; letter-spacing: 0; }
    .form-group { display: flex; flex-direction: column; gap: 7px; }
    label { font-size: .7rem; color: var(--muted); display: flex; justify-content: space-between; gap: 8px; }
    label .val { color: var(--cyan); font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-weight: 800; }
    select, input[type="text"] {
      background: rgba(2, 6, 23, .75);
      border: 1px solid rgba(56, 189, 248, .28);
      color: var(--text);
      padding: 8px 10px;
      border-radius: 6px;
      font-size: .78rem;
      outline: none;
      width: 100%;
    }
    input[type="range"] {
      -webkit-appearance: none;
      appearance: none;
      width: 100%;
      height: 4px;
      border-radius: 2px;
      background: rgba(56, 189, 248, .2);
      outline: none;
    }
    input[type="range"]::-webkit-slider-thumb {
      -webkit-appearance: none;
      width: 14px;
      height: 14px;
      border-radius: 50%;
      background: var(--cyan);
      cursor: pointer;
      box-shadow: 0 0 10px rgba(56,189,248,.8);
    }
    .telemetry-card {
      background: rgba(2, 6, 23, .58);
      border: 1px solid rgba(255,255,255,.06);
      border-radius: 8px;
      padding: 10px;
      display: flex;
      flex-direction: column;
      gap: 6px;
    }
    .telemetry-row { display: flex; justify-content: space-between; align-items: center; gap: 12px; font-size: .69rem; }
    .telemetry-label { color: var(--muted); }
    .telemetry-val {
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-weight: 800;
      color: var(--cyan);
      text-align: right;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      max-width: 165px;
    }
    .highlight-purple { color: var(--purple); }
    .highlight-green { color: var(--green); }
    .highlight-red { color: var(--red); }
    .btn-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 7px; }
    .btn {
      background: rgba(30, 41, 59, .9);
      border: 1px solid rgba(56,189,248,.3);
      color: #e2e8f0;
      padding: 8px 10px;
      min-height: 36px;
      border-radius: 6px;
      font-size: .72rem;
      font-weight: 700;
      cursor: pointer;
      transition: all .18s ease;
      text-align: center;
    }
    .btn:hover { background: rgba(56,189,248,.18); border-color: var(--cyan); color: #fff; }
    .btn-primary { background: rgba(56,189,248,.22); border-color: var(--cyan); color: var(--cyan); }
    .btn-danger { border-color: rgba(251,113,133,.45); color: #fecdd3; }
    .btn-wide { width: 100%; }

    #os-viewport { grid-area: viewport; position: relative; width: 100%; height: 100%; overflow: hidden; }
    #three-canvas { width: 100%; height: 100%; display: block; }
    #viewport-overlay {
      position: absolute;
      left: 12px;
      right: 12px;
      bottom: 12px;
      display: flex;
      gap: 6px;
      flex-wrap: wrap;
      pointer-events: none;
      z-index: 5;
    }
    .chip {
      border: 1px solid rgba(56,189,248,.2);
      background: rgba(2,6,23,.76);
      color: #94a3b8;
      padding: 5px 8px;
      border-radius: 999px;
      font-size: .62rem;
      backdrop-filter: blur(8px);
    }

    #os-chat {
      grid-area: chat;
      display: flex;
      flex-direction: column;
      border-left: 1px solid var(--line);
      background: rgba(15,23,42,.92);
      min-width: 0;
    }
    .chat-header {
      padding: 12px 14px;
      border-bottom: 1px solid rgba(255,255,255,.08);
      font-size: .77rem;
      font-weight: 800;
      color: var(--cyan);
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .chat-body {
      flex: 1;
      padding: 12px;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 11px;
      font-size: .75rem;
      line-height: 1.45;
    }
    .msg { display: flex; flex-direction: column; gap: 4px; max-width: 94%; }
    .msg.user { align-self: flex-end; }
    .msg.bot { align-self: flex-start; }
    .msg-sender { font-size: .61rem; font-weight: 800; color: var(--muted); }
    .msg-bubble {
      padding: 9px 11px;
      border-radius: 8px;
      white-space: pre-wrap;
      background: rgba(30,41,59,.8);
      border: 1px solid rgba(255,255,255,.05);
      color: #e2e8f0;
      word-break: break-word;
    }
    .msg.user .msg-bubble { background: rgba(56,189,248,.18); border-color: rgba(56,189,248,.35); }
    .msg.bot .msg-bubble { background: rgba(2,6,23,.55); border-color: rgba(168,85,247,.28); }
    .chat-input-area { padding: 10px; border-top: 1px solid rgba(255,255,255,.08); display: flex; gap: 8px; }
    .chat-input-area input {
      flex: 1;
      min-width: 0;
      background: rgba(2,6,23,.8);
      border: 1px solid rgba(56,189,248,.25);
      color: var(--text);
      padding: 8px 11px;
      border-radius: 6px;
      font-size: .75rem;
      outline: none;
    }

    #os-statusbar {
      grid-area: statusbar;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 0 16px;
      background: rgba(2,6,23,.97);
      border-top: 1px solid var(--line);
      font-size: .66rem;
      color: #64748b;
      overflow: hidden;
    }
    #os-statusbar span { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    ::-webkit-scrollbar { width: 4px; height: 4px; }
    ::-webkit-scrollbar-thumb { background: rgba(56,189,248,.3); border-radius: 4px; }

    @media (max-width: 1100px) {
      body { overflow: auto; }
      #app-container {
        min-height: 100vh;
        height: auto;
        grid-template-rows: 42px minmax(520px, 70vh) 360px 28px;
        grid-template-columns: 280px minmax(0,1fr);
        grid-template-areas:
          "header header"
          "sidebar viewport"
          "chat chat"
          "statusbar statusbar";
      }
      #os-sidebar, #os-chat { overflow-y: auto; }
    }
    @media (max-width: 720px) {
      #app-container {
        grid-template-rows: auto auto 60vh 390px auto;
        grid-template-columns: 1fr;
        grid-template-areas: "header" "sidebar" "viewport" "chat" "statusbar";
      }
      #os-header { min-height: 54px; align-items: flex-start; padding-top: 8px; padding-bottom: 8px; }
      .os-status-pills { display: none; }
      #os-sidebar { border-right: 0; border-bottom: 1px solid var(--line); }
      #os-chat { border-left: 0; border-top: 1px solid var(--line); }
      #os-statusbar { min-height: 44px; flex-wrap: wrap; padding: 7px 12px; }
    }
  </style>
</head>
<body>
  <div id="app-container">
    <header id="os-header" class="glass-panel">
      <div class="brand">
        <i data-lucide="cpu" style="width:18px;color:#38bdf8"></i>
        JARVIS-X CLOUD OS
        <span class="badge">DM–vΩΞ⁺ CONTROL PLANE</span>
      </div>
      <div class="os-status-pills">
        <div class="pill"><div class="pill-dot" id="life-dot"></div><span id="life-pill">Lifecycle: offline</span></div>
        <div class="pill" id="fp-pill">Fixed point: not measured</div>
        <div class="pill">Substrate: Sparse Uint64</div>
      </div>
    </header>

    <aside id="os-sidebar" class="glass-panel">
      <div>
        <div class="section-title">3D Runtime Horizon <span class="section-note">view filter</span></div>
        <select id="layer-select">
          <option value="all">Full Unified 3D Stack</option>
          <option value="singularity">Fixed-Point Core</option>
          <option value="encoder">Inward Encoder Shells</option>
          <option value="decoder">Outward Decoder Shells</option>
          <option value="substrate">Measured Sparse Voxel State</option>
        </select>
      </div>

      <div class="form-group">
        <div class="section-title">Visualization Controls <span class="section-note">render-only</span></div>
        <label>Shell Depth <span class="val" id="depth-val">6 Levels</span></label>
        <input type="range" id="depth-slider" min="1" max="8" value="6">
        <label>Visual Flow Speed <span class="val" id="velocity-val">3.0x</span></label>
        <input type="range" id="velocity-slider" min="0.5" max="6.0" step="0.1" value="3.0">
        <label>Field Glow Intensity <span class="val" id="energy-val">4.2x</span></label>
        <input type="range" id="energy-slider" min="1.0" max="8.0" step="0.2" value="4.2">
      </div>

      <button class="btn btn-primary btn-wide" id="execute-btn">⚡ EXECUTE VERIFIED OS STEP</button>
      <div class="btn-grid">
        <button class="btn" id="demo-btn">Boot Demo</button>
        <button class="btn" id="run-btn">Run ×8</button>
        <button class="btn" id="auto-btn">Auto Run</button>
        <button class="btn" id="stop-btn">Stop</button>
        <button class="btn" id="reset-btn">Reset Halt</button>
        <button class="btn btn-danger" id="shutdown-btn">Shutdown</button>
      </div>

      <div>
        <div class="section-title">Measured Runtime Telemetry</div>
        <div class="telemetry-card">
          <div class="telemetry-row"><span class="telemetry-label">Active / Logical Cells</span><span class="telemetry-val" id="cells-val">0 / 0</span></div>
          <div class="telemetry-row"><span class="telemetry-label">Packed / Logical Words</span><span class="telemetry-val" id="words-val">0 / 0</span></div>
          <div class="telemetry-row"><span class="telemetry-label">Logical Phase Velocity</span><span class="telemetry-val" id="phase-val">—</span></div>
          <div class="telemetry-row"><span class="telemetry-label">Spatial Entropy</span><span class="telemetry-val" id="entropy-val">0</span></div>
          <div class="telemetry-row"><span class="telemetry-label">Logical Kinetic Metric</span><span class="telemetry-val" id="kinetic-val">—</span></div>
          <div class="telemetry-row"><span class="telemetry-label">Reconstruction MSE</span><span class="telemetry-val highlight-purple" id="mse-val">—</span></div>
          <div class="telemetry-row"><span class="telemetry-label">Fixed-Point Residual</span><span class="telemetry-val highlight-purple" id="fp-val">—</span></div>
          <div class="telemetry-row"><span class="telemetry-label">Journal Integrity</span><span class="telemetry-val highlight-green" id="journal-val">—</span></div>
          <div class="telemetry-row"><span class="telemetry-label">State Digest</span><span class="telemetry-val" id="hash-val">—</span></div>
        </div>
      </div>
    </aside>

    <main id="os-viewport">
      <div id="three-canvas"></div>
      <div id="viewport-overlay">
        <span class="chip">drag: orbit</span>
        <span class="chip">wheel/pinch: zoom</span>
        <span class="chip">cyan points: measured active sparse cells</span>
        <span class="chip">shells/rings/particles: visualization</span>
      </div>
    </main>

    <section id="os-chat" class="glass-panel">
      <div class="chat-header">
        <span>JARVIS-X OPERATIONAL SHELL</span>
        <i data-lucide="terminal" style="width:16px;color:#c084fc"></i>
      </div>
      <div class="chat-body" id="chat-body">
        <div class="msg bot">
          <span class="msg-sender">JARVIS-X OS KERNEL</span>
          <div class="msg-bubble">3D control plane ready. Runtime values shown in the console are read from the live Dr Moagi OS API; shell geometry and field glow are visualization layers. Type /help for commands.</div>
        </div>
      </div>
      <div class="chat-input-area">
        <input type="text" id="chat-input" placeholder="Execute OS command..." autocomplete="off">
        <button class="btn btn-primary" id="send-btn">Send</button>
      </div>
    </section>

    <footer id="os-statusbar">
      <span id="system-status">System: connecting…</span>
      <span>Invariant: provisional state is non-authoritative until validated commit</span>
      <span id="scheduler-status">Scheduler: —</span>
    </footer>
  </div>

  <script>
    if (window.lucide) lucide.createIcons();

    let scene, camera, renderer, controls;
    let matrixGroup, singularityCore, innerGlow, particleStreams, particlePositions, voxelPoints;
    let concentricShells = [], emRings = [];
    let animTime = 0, lastSnapshotCycle = null, lastStatus = null, passPulse = 0;

    const $ = (id) => document.getElementById(id);
    const finite = (value) => typeof value === 'number' && Number.isFinite(value);
    const scientific = (value, digits = 3) => finite(Number(value)) ? Number(value).toExponential(digits) : '—';
    const fixed = (value, digits = 6) => finite(Number(value)) ? Number(value).toFixed(digits) : '—';
    const shortHash = (value) => typeof value === 'string' && value.length > 12 ? `${value.slice(0, 8)}…${value.slice(-4)}` : (value || '—');

    async function api(path, options = undefined) {
      const response = await fetch(path, options);
      let payload;
      try { payload = await response.json(); }
      catch { payload = { detail: await response.text() }; }
      if (!response.ok) throw new Error(payload.detail || JSON.stringify(payload));
      return payload;
    }

    function setText(id, value) { $(id).textContent = value === null || value === undefined ? '—' : String(value); }

    function appendMessage(sender, text, kind = 'bot') {
      const body = $('chat-body');
      const msg = document.createElement('div');
      msg.className = `msg ${kind}`;
      const who = document.createElement('span');
      who.className = 'msg-sender';
      who.textContent = sender;
      const bubble = document.createElement('div');
      bubble.className = 'msg-bubble';
      bubble.textContent = text;
      msg.appendChild(who);
      msg.appendChild(bubble);
      body.appendChild(msg);
      body.scrollTop = body.scrollHeight;
    }

    function appendBot(text) { appendMessage('JARVIS-X OS KERNEL', text, 'bot'); }
    function appendUser(text) { appendMessage('OPERATOR', text, 'user'); }

    function disposeGroup(group) {
      if (!group) return;
      group.traverse((obj) => {
        if (obj.geometry) obj.geometry.dispose();
        if (obj.material) {
          if (Array.isArray(obj.material)) obj.material.forEach((m) => m.dispose());
          else obj.material.dispose();
        }
      });
    }

    function initThree() {
      const container = $('three-canvas');
      scene = new THREE.Scene();
      scene.background = new THREE.Color(0x020617);
      scene.fog = new THREE.FogExp2(0x020617, 0.008);
      camera = new THREE.PerspectiveCamera(45, Math.max(1, container.clientWidth) / Math.max(1, container.clientHeight), 0.1, 1000);
      camera.position.set(0, 20, 60);
      renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
      renderer.setSize(container.clientWidth, container.clientHeight);
      container.appendChild(renderer.domElement);
      controls = new THREE.OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;
      controls.dampingFactor = 0.05;
      controls.minDistance = 8;
      controls.maxDistance = 130;
      scene.add(new THREE.AmbientLight(0x1e293b, 2.0));
      const centerLight = new THREE.PointLight(0xa855f7, 5, 90);
      centerLight.position.set(0, 0, 0);
      scene.add(centerLight);
      const cyanPoint = new THREE.PointLight(0x00f3ff, 2.5, 70);
      cyanPoint.position.set(30, 30, 30);
      scene.add(cyanPoint);
      window.addEventListener('resize', onWindowResize);
    }

    function build3DMatrix() {
      if (matrixGroup) {
        scene.remove(matrixGroup);
        disposeGroup(matrixGroup);
      }
      matrixGroup = new THREE.Group();
      const depth = parseInt($('depth-slider').value, 10);
      const glow = parseFloat($('energy-slider').value);

      const coreGeom = new THREE.IcosahedronGeometry(2.8, 2);
      const coreMat = new THREE.MeshStandardMaterial({
        color: 0xa855f7, wireframe: true, emissive: 0x9333ea, emissiveIntensity: 0.55 + glow * 0.09
      });
      singularityCore = new THREE.Mesh(coreGeom, coreMat);
      singularityCore.userData.role = 'core';
      matrixGroup.add(singularityCore);

      const innerGeom = new THREE.SphereGeometry(1.35, 28, 28);
      const innerMat = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: Math.min(.95, .28 + glow * .08) });
      innerGlow = new THREE.Mesh(innerGeom, innerMat);
      innerGlow.userData.role = 'core';
      matrixGroup.add(innerGlow);

      concentricShells = [];
      const colors = [0x00f3ff, 0x38bdf8, 0xa855f7, 0xff00a0, 0x10b981];
      for (let i = 1; i <= depth; i++) {
        const radius = 5 + i * 4.2;
        const color = colors[i % colors.length];
        const shell = new THREE.Mesh(
          new THREE.IcosahedronGeometry(radius, 1),
          new THREE.MeshStandardMaterial({
            color, wireframe: true, transparent: true,
            opacity: Math.max(.08, .36 - i * .03), emissive: color, emissiveIntensity: .2 + glow * .035
          })
        );
        shell.userData.role = i <= Math.ceil(depth / 2) ? 'decoder' : 'encoder';
        matrixGroup.add(shell);
        concentricShells.push(shell);
      }

      emRings = [];
      for (let i = 0; i < 3; i++) {
        const ring = new THREE.Mesh(
          new THREE.TorusGeometry(10 + i * 6.5, .10 + glow * .006, 12, 100),
          new THREE.MeshBasicMaterial({
            color: i % 2 === 0 ? 0x00f3ff : 0xff00a0,
            transparent: true,
            opacity: Math.min(.82, .20 + glow * .07)
          })
        );
        ring.rotation.x = Math.PI / (i + 1.2);
        ring.rotation.y = Math.PI / (i + 1.8);
        ring.userData.role = 'visual-field';
        matrixGroup.add(ring);
        emRings.push(ring);
      }

      createParticleStreams(depth);
      scene.add(matrixGroup);
      if (lastStatus && lastStatus.loaded) refreshSnapshot(true);
      applyHorizonVisibility();
    }

    function createParticleStreams(depth) {
      const count = 800;
      const geom = new THREE.BufferGeometry();
      particlePositions = new Float32Array(count * 3);
      const maxRadius = 5 + depth * 4.2;
      for (let i = 0; i < count; i++) resetParticle(i, maxRadius, true);
      geom.setAttribute('position', new THREE.BufferAttribute(particlePositions, 3));
      particleStreams = new THREE.Points(
        geom,
        new THREE.PointsMaterial({ color: 0x38bdf8, size: .34, transparent: true, opacity: .68 })
      );
      particleStreams.userData.role = 'visual-flow';
      matrixGroup.add(particleStreams);
    }

    function resetParticle(index, radius, randomRadius = false) {
      const r = randomRadius ? Math.random() * radius + 3 : radius;
      const theta = Math.random() * Math.PI * 2;
      const phi = (Math.random() - .5) * Math.PI;
      particlePositions[index * 3] = r * Math.cos(theta) * Math.cos(phi);
      particlePositions[index * 3 + 1] = r * Math.sin(phi);
      particlePositions[index * 3 + 2] = r * Math.sin(theta) * Math.cos(phi);
    }

    function updateVoxelGeometry(cells, side) {
      if (!matrixGroup) return;
      if (voxelPoints) {
        matrixGroup.remove(voxelPoints);
        if (voxelPoints.geometry) voxelPoints.geometry.dispose();
        if (voxelPoints.material) voxelPoints.material.dispose();
        voxelPoints = null;
      }
      if (!Array.isArray(cells) || !cells.length) return;
      const positions = new Float32Array(cells.length * 3);
      const colors = new Float32Array(cells.length * 3);
      const center = (Math.max(1, side) - 1) / 2;
      const scale = 52 / Math.max(1, side - 1);
      const cyan = new THREE.Color(0x38bdf8);
      const gold = new THREE.Color(0xfbbf24);
      cells.forEach((cell, i) => {
        positions[i * 3] = (cell.x - center) * scale;
        positions[i * 3 + 1] = (cell.y - center) * scale;
        positions[i * 3 + 2] = (cell.z - center) * scale;
        const magnitude = Math.min(1, Math.abs(Number(cell.value) || 0));
        const color = cyan.clone().lerp(gold, magnitude);
        colors[i * 3] = color.r;
        colors[i * 3 + 1] = color.g;
        colors[i * 3 + 2] = color.b;
      });
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
      geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
      voxelPoints = new THREE.Points(
        geometry,
        new THREE.PointsMaterial({ size: .72, vertexColors: true, transparent: true, opacity: .95, sizeAttenuation: true })
      );
      voxelPoints.userData.role = 'substrate';
      matrixGroup.add(voxelPoints);
      applyHorizonVisibility();
    }

    function applyHorizonVisibility() {
      const target = $('layer-select').value;
      if (!matrixGroup) return;
      if (singularityCore) singularityCore.visible = target === 'all' || target === 'singularity' || target === 'encoder' || target === 'decoder';
      if (innerGlow) innerGlow.visible = singularityCore.visible;
      concentricShells.forEach((shell) => {
        shell.visible = target === 'all' || target === shell.userData.role;
      });
      emRings.forEach((ring) => ring.visible = target === 'all' || target === 'substrate');
      if (particleStreams) particleStreams.visible = target === 'all' || target === 'encoder' || target === 'decoder';
      if (voxelPoints) voxelPoints.visible = target === 'all' || target === 'substrate';

      const distance = { singularity: 20, encoder: 38, decoder: 45, substrate: 60, all: 60 }[target] || 60;
      animateCamera(distance);
    }

    function animateCamera(distance) {
      const duration = 650;
      const start = camera.position.clone();
      const end = new THREE.Vector3(0, distance * .3, distance);
      const started = performance.now();
      function step(now) {
        let p = Math.min((now - started) / duration, 1);
        p = .5 - Math.cos(p * Math.PI) / 2;
        camera.position.lerpVectors(start, end, p);
        controls.target.set(0, 0, 0);
        if (p < 1) requestAnimationFrame(step);
      }
      requestAnimationFrame(step);
    }

    function updateVisualLabels() {
      setText('depth-val', `${$('depth-slider').value} Levels`);
      setText('velocity-val', `${parseFloat($('velocity-slider').value).toFixed(1)}x`);
      setText('energy-val', `${parseFloat($('energy-slider').value).toFixed(1)}x`);
    }

    function renderTelemetry(status) {
      lastStatus = status;
      const plane = status.bitplane || {};
      const report = status.last_report || null;
      setText('cells-val', `${status.active_cells || 0} / ${status.logical_cells || 0}`);
      setText('words-val', `${plane.packed_words || 0} / ${plane.logical_words || 0}`);
      setText('phase-val', report ? fixed(Number(report.phase_velocity), 8) : '—');
      setText('entropy-val', fixed(Number(plane.entropy || 0), 8));
      setText('kinetic-val', report ? scientific(Number(report.kinetic_energy), 3) : '—');
      setText('mse-val', report ? scientific(Number(report.reconstruction_mse), 3) : '—');
      setText('fp-val', report && report.fixed_point_residual !== null ? scientific(Number(report.fixed_point_residual), 3) : '—');
      setText('journal-val', status.journal_valid ? 'VALID' : 'INVALID');
      $('journal-val').className = `telemetry-val ${status.journal_valid ? 'highlight-green' : 'highlight-red'}`;
      setText('hash-val', shortHash(status.state_hash));
      setText('life-pill', `Lifecycle: ${status.lifecycle}`);
      const dot = $('life-dot');
      dot.className = `pill-dot ${status.lifecycle === 'halted' ? 'halted' : (status.lifecycle !== 'offline' ? 'ready' : '')}`;
      if (report && report.fixed_point_converged) setText('fp-pill', `Fixed point: converged (${scientific(Number(report.fixed_point_residual), 2)})`);
      else if (report && report.fixed_point_residual !== null) setText('fp-pill', `Fixed point residual: ${scientific(Number(report.fixed_point_residual), 2)}`);
      else setText('fp-pill', 'Fixed point: not measured');
      setText('system-status', `System: ${status.system || 'Dr Moagi 3D OS'} • ${status.lifecycle}`);
      setText('scheduler-status', `Scheduler: ${status.scheduler_running ? 'running' : 'stopped'} • cycle ${status.cycle || 0}`);
    }

    async function refreshSnapshot(force = false) {
      if (!lastStatus || !lastStatus.loaded) {
        updateVoxelGeometry([], lastStatus ? lastStatus.side : 1);
        return;
      }
      if (!force && lastSnapshotCycle === lastStatus.cycle) return;
      const snap = await api('/v1/os/snapshot?limit=1800');
      lastSnapshotCycle = lastStatus.cycle;
      updateVoxelGeometry(snap.cells || [], lastStatus.side || 1);
    }

    async function refresh(forceSnapshot = false) {
      try {
        const status = await api('/v1/os/status');
        renderTelemetry(status);
        await refreshSnapshot(forceSnapshot);
      } catch (error) {
        setText('system-status', `System: API unavailable • ${error.message}`);
      }
    }

    async function action(label, fn, forceSnapshot = true) {
      try {
        const result = await fn();
        appendBot(`${label}: accepted by service.\n${summarizeResult(result)}`);
        passPulse = 1;
        await refresh(forceSnapshot);
        return result;
      } catch (error) {
        appendBot(`${label}: ${error.message}`);
        await refresh(false);
        return null;
      }
    }

    function summarizeResult(result) {
      if (!result || typeof result !== 'object') return String(result || '');
      if (Array.isArray(result.reports)) {
        const final = result.reports[result.reports.length - 1];
        return final ? `cycles=${result.reports.length}, committed=${final.committed}, residual=${scientific(Number(final.fixed_point_residual), 2)}` : 'no cycle reports';
      }
      if ('committed' in result) return `committed=${result.committed}, cycle=${result.cycle}, reconstruction_mse=${scientific(Number(result.reconstruction_mse), 2)}`;
      if ('lifecycle' in result) return `lifecycle=${result.lifecycle}, cycle=${result.cycle}, active_cells=${result.active_cells}`;
      return 'operation complete';
    }

    async function triggerOSPass() {
      if (!lastStatus || !lastStatus.loaded) {
        appendBot('No sparse field is loaded. Use Boot Demo or /demo first.');
        return;
      }
      await action('Verified OS step', () => api('/v1/os/step', { method: 'POST' }));
    }

    async function bootDemo() {
      await action('Boot demo', async () => {
        await api('/v1/os/boot', { method: 'POST' });
        return api('/v1/os/demo', { method: 'POST' });
      });
    }

    async function runCycles(count = 8) {
      if (!lastStatus || !lastStatus.loaded) return appendBot('No sparse field is loaded. Use /demo first.');
      count = Math.max(1, Math.min(4096, Number(count) || 8));
      await action(`Run ×${count}`, () => api('/v1/os/run', {
        method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ cycles: count })
      }));
    }

    async function startAutorun(intervalMs = 350) {
      if (!lastStatus || !lastStatus.loaded) return appendBot('No sparse field is loaded. Use /demo first.');
      const seconds = Math.max(.05, Math.min(60, Number(intervalMs) / 1000 || .35));
      await action('Autorun start', () => api('/v1/os/autorun/start', {
        method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ interval_seconds: seconds })
      }), false);
    }

    async function processOSCommand(raw) {
      const parts = raw.trim().split(/\s+/);
      const command = (parts[0] || '').toLowerCase();
      if (command === '/help') {
        appendBot('AVAILABLE COMMANDS:\n/status — measured runtime state\n/demo — boot and load demo sparse field\n/step or /matrix — one verified transaction\n/run [n] — run bounded cycles\n/auto [ms] — start autorun\n/stop — stop autorun\n/reset — clear HALTED state\n/sys — hashes, policy and journal state\n/clear — clear shell');
      } else if (command === '/status') {
        await refresh(false);
        const s = lastStatus;
        const r = s && s.last_report;
        appendBot(s ? `SYSTEM STATUS:\nlifecycle=${s.lifecycle}\ncycle=${s.cycle}\nactive_cells=${s.active_cells}/${s.logical_cells}\nbit_density=${fixed(Number((s.bitplane || {}).density || 0), 8)}\nreconstruction_mse=${r ? scientific(Number(r.reconstruction_mse), 3) : '—'}\nfixed_point_residual=${r && r.fixed_point_residual !== null ? scientific(Number(r.fixed_point_residual), 3) : '—'}\njournal_valid=${s.journal_valid}` : 'Status unavailable.');
      } else if (command === '/demo') {
        await bootDemo();
      } else if (command === '/step' || command === '/matrix') {
        await triggerOSPass();
      } else if (command === '/run') {
        await runCycles(parts[1] || 8);
      } else if (command === '/auto') {
        await startAutorun(parts[1] || 350);
      } else if (command === '/stop') {
        await action('Autorun stop', () => api('/v1/os/autorun/stop', { method: 'POST' }), false);
      } else if (command === '/reset') {
        await action('Halt reset', () => api('/v1/os/halt/reset', { method: 'POST' }), false);
      } else if (command === '/sys') {
        await refresh(false);
        const s = lastStatus;
        appendBot(s ? `CONTROL STATE:\nmode=${s.mode}\npolicy=${JSON.stringify(s.policy)}\nstate_hash=${s.state_hash}\njournal_head=${s.journal_head}\njournal_valid=${s.journal_valid}\nscheduler_running=${s.scheduler_running}\nhalt_reason=${s.halt_reason || 'none'}` : 'Status unavailable.');
      } else if (command === '/clear') {
        $('chat-body').innerHTML = '';
      } else {
        appendBot('This shell is a bounded OS control plane, not a general-purpose language model. Use /help for executable commands.');
      }
    }

    function submitUserMessage() {
      const input = $('chat-input');
      const text = input.value.trim();
      if (!text) return;
      appendUser(text);
      input.value = '';
      processOSCommand(text);
    }

    function animate() {
      requestAnimationFrame(animate);
      const speed = parseFloat($('velocity-slider').value) * .01;
      animTime += speed;
      if (passPulse > 0) passPulse = Math.max(0, passPulse - .018);
      if (singularityCore) {
        singularityCore.rotation.y = animTime * .9;
        singularityCore.rotation.x = animTime * .45;
        const pulse = 1 + passPulse * .2;
        singularityCore.scale.setScalar(pulse);
      }
      concentricShells.forEach((shell, i) => {
        const dir = i % 2 === 0 ? 1 : -1;
        shell.rotation.y = animTime * (.2 + i * .04) * dir;
        shell.rotation.z = animTime * .1 * dir;
      });
      emRings.forEach((ring, i) => ring.rotation.z = animTime * (.35 + i * .1));
      if (particleStreams && particlePositions) {
        const depth = parseInt($('depth-slider').value, 10);
        const maxRadius = 5 + depth * 4.2;
        const contraction = Math.min(.08, .004 + parseFloat($('velocity-slider').value) * .006);
        for (let i = 0; i < particlePositions.length / 3; i++) {
          const idx = i * 3;
          const x = particlePositions[idx], y = particlePositions[idx + 1], z = particlePositions[idx + 2];
          const dist = Math.sqrt(x*x + y*y + z*z);
          particlePositions[idx] *= (1 - contraction);
          particlePositions[idx + 1] *= (1 - contraction);
          particlePositions[idx + 2] *= (1 - contraction);
          if (dist < 1.5) resetParticle(i, maxRadius, false);
        }
        particleStreams.geometry.attributes.position.needsUpdate = true;
      }
      controls.update();
      renderer.render(scene, camera);
    }

    function onWindowResize() {
      if (!renderer || !camera) return;
      const container = $('three-canvas');
      if (!container.clientWidth || !container.clientHeight) return;
      camera.aspect = container.clientWidth / container.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(container.clientWidth, container.clientHeight);
    }

    function bindControls() {
      $('layer-select').addEventListener('change', applyHorizonVisibility);
      $('depth-slider').addEventListener('input', updateVisualLabels);
      $('depth-slider').addEventListener('change', build3DMatrix);
      $('velocity-slider').addEventListener('input', updateVisualLabels);
      $('energy-slider').addEventListener('input', updateVisualLabels);
      $('energy-slider').addEventListener('change', build3DMatrix);
      $('execute-btn').addEventListener('click', triggerOSPass);
      $('demo-btn').addEventListener('click', bootDemo);
      $('run-btn').addEventListener('click', () => runCycles(8));
      $('auto-btn').addEventListener('click', () => startAutorun(350));
      $('stop-btn').addEventListener('click', () => action('Autorun stop', () => api('/v1/os/autorun/stop', { method: 'POST' }), false));
      $('reset-btn').addEventListener('click', () => action('Halt reset', () => api('/v1/os/halt/reset', { method: 'POST' }), false));
      $('shutdown-btn').addEventListener('click', () => action('Shutdown', () => api('/v1/os/shutdown', { method: 'POST' }), false));
      $('send-btn').addEventListener('click', submitUserMessage);
      $('chat-input').addEventListener('keydown', (event) => { if (event.key === 'Enter') submitUserMessage(); });
    }

    initThree();
    bindControls();
    updateVisualLabels();
    build3DMatrix();
    animate();
    refresh(true);
    setInterval(() => refresh(false), 1000);
  </script>
</body>
</html>"""
