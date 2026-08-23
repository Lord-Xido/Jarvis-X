"""Three.js enterprise browser control plane for the Dr Moagi 3D OS service."""

DR_MOAGI_OS_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>JARVIS-X OS • Measured 3D Self-Optimizing Runtime</title>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
  <script src="https://unpkg.com/lucide@latest"></script>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    :root {
      color-scheme: dark;
      --bg: #020617;
      --panel: rgba(15, 23, 42, 0.90);
      --panel-deep: rgba(2, 6, 23, 0.76);
      --cyan: #38bdf8;
      --purple: #c084fc;
      --green: #4ade80;
      --amber: #fbbf24;
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
      grid-template-rows: 46px 1fr 30px;
      grid-template-columns: 340px minmax(0, 1fr) 390px;
      grid-template-areas:
        "header header header"
        "sidebar viewport shell"
        "statusbar statusbar statusbar";
    }
    .glass-panel {
      background: var(--panel);
      backdrop-filter: blur(16px);
      -webkit-backdrop-filter: blur(16px);
    }
    #os-header {
      grid-area: header;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 14px;
      padding: 0 16px;
      border-bottom: 1px solid var(--line);
      background: rgba(15, 23, 42, 0.98);
      overflow: hidden;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 10px;
      white-space: nowrap;
      color: var(--cyan);
      font-weight: 850;
      font-size: .88rem;
      letter-spacing: .07em;
    }
    .brand-icon {
      width: 27px;
      height: 27px;
      border-radius: 9px;
      display: grid;
      place-items: center;
      background: linear-gradient(135deg, rgba(56,189,248,.30), rgba(192,132,252,.30), rgba(251,191,36,.22));
      border: 1px solid rgba(192,132,252,.50);
      box-shadow: 0 0 24px rgba(168,85,247,.20);
    }
    .badge {
      font-size: .60rem;
      background: rgba(168, 85, 247, .18);
      color: var(--purple);
      padding: 2px 7px;
      border-radius: 4px;
      border: 1px solid rgba(168, 85, 247, .42);
      font-weight: 800;
    }
    .os-status-pills { display: flex; gap: 7px; font-size: .65rem; color: var(--muted); overflow: hidden; }
    .pill {
      display: flex;
      align-items: center;
      gap: 6px;
      white-space: nowrap;
      background: rgba(30, 41, 59, .68);
      padding: 4px 8px;
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
      gap: 12px;
      padding: 13px;
      border-right: 1px solid var(--line);
      overflow-y: auto;
    }
    .section-title {
      font-size: .66rem;
      font-weight: 850;
      text-transform: uppercase;
      letter-spacing: .10em;
      color: var(--cyan);
      margin-bottom: 7px;
      display: flex;
      justify-content: space-between;
      gap: 10px;
      align-items: center;
    }
    .section-note { font-size: .56rem; color: #64748b; text-transform: none; letter-spacing: 0; font-weight: 600; }
    .form-group { display: flex; flex-direction: column; gap: 7px; }
    label { font-size: .68rem; color: var(--muted); display: flex; justify-content: space-between; gap: 8px; }
    label .val { color: var(--cyan); font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-weight: 800; }
    select, input[type="text"] {
      background: rgba(2, 6, 23, .78);
      border: 1px solid rgba(56, 189, 248, .28);
      color: var(--text);
      padding: 8px 10px;
      border-radius: 6px;
      font-size: .76rem;
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
      padding: 9px;
      display: flex;
      flex-direction: column;
      gap: 6px;
    }
    .telemetry-row { display: flex; justify-content: space-between; align-items: center; gap: 11px; font-size: .67rem; }
    .telemetry-label { color: var(--muted); }
    .telemetry-val {
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-weight: 800;
      color: var(--cyan);
      text-align: right;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      max-width: 175px;
    }
    .highlight-purple { color: var(--purple); }
    .highlight-green { color: var(--green); }
    .highlight-amber { color: var(--amber); }
    .highlight-red { color: var(--red); }
    .btn-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 7px; }
    .btn {
      background: rgba(30, 41, 59, .92);
      border: 1px solid rgba(56,189,248,.3);
      color: #e2e8f0;
      padding: 8px 10px;
      min-height: 35px;
      border-radius: 6px;
      font-size: .70rem;
      font-weight: 750;
      cursor: pointer;
      transition: all .18s ease;
      text-align: center;
    }
    .btn:hover { background: rgba(56,189,248,.18); border-color: var(--cyan); color: #fff; }
    .btn:disabled { opacity: .45; cursor: not-allowed; }
    .btn-primary { background: rgba(56,189,248,.22); border-color: var(--cyan); color: var(--cyan); }
    .btn-meta {
      background: linear-gradient(90deg, rgba(8,145,178,.22), rgba(147,51,234,.25), rgba(217,119,6,.20));
      border-color: rgba(192,132,252,.60);
      color: #e9d5ff;
      box-shadow: 0 0 18px rgba(168,85,247,.10);
    }
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
      background: rgba(2,6,23,.78);
      color: #94a3b8;
      padding: 5px 8px;
      border-radius: 999px;
      font-size: .60rem;
      backdrop-filter: blur(8px);
    }
    #metric-panel {
      position: absolute;
      top: 12px;
      right: 12px;
      width: 220px;
      z-index: 6;
      pointer-events: none;
      border: 1px solid rgba(192,132,252,.28);
      border-radius: 9px;
      background: rgba(2,6,23,.76);
      backdrop-filter: blur(10px);
      padding: 9px;
    }
    #metric-title { color: var(--purple); font-size: .63rem; font-weight: 850; letter-spacing: .06em; margin-bottom: 6px; }
    #metric-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 3px; font: 700 .60rem ui-monospace, monospace; color: #cbd5e1; }
    #metric-grid span { background: rgba(15,23,42,.70); border: 1px solid rgba(148,163,184,.08); border-radius: 4px; padding: 4px; text-align: center; }
    #metric-note { margin-top: 5px; color: #64748b; font-size: .52rem; line-height: 1.3; }
    #node-tooltip {
      position: absolute;
      display: none;
      z-index: 12;
      pointer-events: none;
      min-width: 210px;
      max-width: 290px;
      background: rgba(2,6,23,.95);
      border: 1px solid rgba(192,132,252,.45);
      border-radius: 7px;
      padding: 8px;
      font: .59rem ui-monospace, monospace;
      color: #cbd5e1;
      white-space: pre-line;
      box-shadow: 0 8px 30px rgba(0,0,0,.35);
    }

    #os-shell {
      grid-area: shell;
      display: flex;
      flex-direction: column;
      border-left: 1px solid var(--line);
      background: rgba(15,23,42,.93);
      min-width: 0;
    }
    .shell-header {
      padding: 11px 13px;
      border-bottom: 1px solid rgba(255,255,255,.08);
      font-size: .75rem;
      font-weight: 800;
      color: var(--cyan);
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .shell-body {
      flex: 1;
      padding: 11px;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 10px;
      font-size: .73rem;
      line-height: 1.45;
    }
    .msg { display: flex; flex-direction: column; gap: 4px; max-width: 94%; }
    .msg.user { align-self: flex-end; }
    .msg.bot { align-self: flex-start; }
    .msg-sender { font-size: .59rem; font-weight: 800; color: var(--muted); }
    .msg-bubble {
      padding: 9px 10px;
      border-radius: 8px;
      white-space: pre-wrap;
      background: rgba(30,41,59,.80);
      border: 1px solid rgba(255,255,255,.05);
      color: #e2e8f0;
      word-break: break-word;
    }
    .msg.user .msg-bubble { background: rgba(56,189,248,.18); border-color: rgba(56,189,248,.35); }
    .msg.bot .msg-bubble { background: rgba(2,6,23,.55); border-color: rgba(168,85,247,.28); }
    .shell-input { padding: 9px; border-top: 1px solid rgba(255,255,255,.08); display: flex; gap: 7px; }
    .shell-input input {
      flex: 1;
      min-width: 0;
      background: rgba(2,6,23,.80);
      border: 1px solid rgba(56,189,248,.25);
      color: var(--text);
      padding: 8px 10px;
      border-radius: 6px;
      font-size: .73rem;
      outline: none;
    }

    #os-statusbar {
      grid-area: statusbar;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 0 16px;
      background: rgba(2,6,23,.98);
      border-top: 1px solid var(--line);
      font-size: .64rem;
      color: #64748b;
      overflow: hidden;
    }
    #os-statusbar span { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    ::-webkit-scrollbar { width: 4px; height: 4px; }
    ::-webkit-scrollbar-thumb { background: rgba(56,189,248,.3); border-radius: 4px; }

    @media (max-width: 1120px) {
      body { overflow: auto; }
      #app-container {
        min-height: 100vh;
        height: auto;
        grid-template-rows: 46px minmax(540px, 70vh) 390px 30px;
        grid-template-columns: 300px minmax(0,1fr);
        grid-template-areas:
          "header header"
          "sidebar viewport"
          "shell shell"
          "statusbar statusbar";
      }
    }
    @media (max-width: 760px) {
      #app-container {
        grid-template-rows: auto auto 62vh 410px auto;
        grid-template-columns: 1fr;
        grid-template-areas: "header" "sidebar" "viewport" "shell" "statusbar";
      }
      #os-header { min-height: 56px; align-items: flex-start; padding-top: 8px; padding-bottom: 8px; }
      .os-status-pills { display: none; }
      #os-sidebar { border-right: 0; border-bottom: 1px solid var(--line); }
      #os-shell { border-left: 0; border-top: 1px solid var(--line); }
      #metric-panel { width: 190px; }
      #os-statusbar { min-height: 46px; flex-wrap: wrap; padding: 7px 12px; }
    }
  </style>
</head>
<body>
  <div id="app-container">
    <header id="os-header" class="glass-panel">
      <div class="brand">
        <div class="brand-icon"><i data-lucide="infinity" style="width:17px;color:#e9d5ff"></i></div>
        JARVIS-X SYSTEM ARCHITECTURE LOOP
        <span class="badge">DM–vΩΞ⁺ • M ↺ M</span>
      </div>
      <div class="os-status-pills">
        <div class="pill"><div class="pill-dot" id="life-dot"></div><span id="life-pill">Lifecycle: offline</span></div>
        <div class="pill" id="fp-pill">ΔΨ: not measured</div>
        <div class="pill" id="meta-pill">Meta epoch: 0</div>
        <div class="pill" id="claim-pill">SOTA: unverified</div>
      </div>
    </header>

    <aside id="os-sidebar" class="glass-panel">
      <div>
        <div class="section-title">Recursive Architecture View <span class="section-note">measured + visual</span></div>
        <select id="layer-select">
          <option value="all">Unified Runtime + Meta Lattice</option>
          <option value="meta">3D Meta-Optimization Lattice</option>
          <option value="substrate">Measured Sparse Voxel State</option>
          <option value="encoder">Inward Encoder Shells</option>
          <option value="decoder">Outward Decoder Shells</option>
          <option value="core">Fixed-Point Core</option>
        </select>
      </div>

      <div class="form-group">
        <div class="section-title">Visualization Controls <span class="section-note">render-only</span></div>
        <label>Shell Depth <span class="val" id="depth-val">6 Levels</span></label>
        <input type="range" id="depth-slider" min="1" max="8" value="6">
        <label>Visual Flow Speed <span class="val" id="velocity-val">3.0x</span></label>
        <input type="range" id="velocity-slider" min="0.5" max="6.0" step="0.1" value="3.0">
        <label>Field Glow <span class="val" id="glow-val">4.2x</span></label>
        <input type="range" id="glow-slider" min="1.0" max="8.0" step="0.2" value="4.2">
      </div>

      <button class="btn btn-primary btn-wide" id="execute-btn">⚡ EXECUTE VERIFIED OS STEP</button>
      <button class="btn btn-meta btn-wide" id="meta-btn">∞ TURN SYSTEM INWARD / OPTIMIZE</button>
      <div class="btn-grid">
        <button class="btn" id="demo-btn">Boot Demo</button>
        <button class="btn" id="run-btn">Run ×8</button>
        <button class="btn" id="auto-btn">Auto Run</button>
        <button class="btn" id="stop-btn">Stop</button>
        <button class="btn" id="lock-btn">Meta Gate: ARMED</button>
        <button class="btn btn-danger" id="shutdown-btn">Shutdown</button>
      </div>

      <div>
        <div class="section-title">Measured Runtime Telemetry</div>
        <div class="telemetry-card">
          <div class="telemetry-row"><span class="telemetry-label">Active / Logical Cells</span><span class="telemetry-val" id="cells-val">0 / 0</span></div>
          <div class="telemetry-row"><span class="telemetry-label">Bit Density</span><span class="telemetry-val" id="density-val">—</span></div>
          <div class="telemetry-row"><span class="telemetry-label">Logical Phase Velocity</span><span class="telemetry-val" id="phase-val">—</span></div>
          <div class="telemetry-row"><span class="telemetry-label">Reconstruction MSE</span><span class="telemetry-val highlight-purple" id="mse-val">—</span></div>
          <div class="telemetry-row"><span class="telemetry-label">DM-DD Residual RMS</span><span class="telemetry-val highlight-purple" id="dd-val">—</span></div>
          <div class="telemetry-row"><span class="telemetry-label">Fixed-Point ΔΨ</span><span class="telemetry-val highlight-purple" id="fp-val">—</span></div>
          <div class="telemetry-row"><span class="telemetry-label">DMOS2 Transport</span><span class="telemetry-val" id="transport-val">—</span></div>
          <div class="telemetry-row"><span class="telemetry-label">State / Journal</span><span class="telemetry-val highlight-green" id="integrity-val">—</span></div>
        </div>
      </div>

      <div>
        <div class="section-title">Inward Meta-Optimizer</div>
        <div class="telemetry-card">
          <div class="telemetry-row"><span class="telemetry-label">Epoch</span><span class="telemetry-val" id="meta-epoch-val">0</span></div>
          <div class="telemetry-row"><span class="telemetry-label">Baseline Score</span><span class="telemetry-val" id="baseline-score-val">—</span></div>
          <div class="telemetry-row"><span class="telemetry-label">Best Score</span><span class="telemetry-val highlight-green" id="best-score-val">—</span></div>
          <div class="telemetry-row"><span class="telemetry-label">Relative Improvement</span><span class="telemetry-val highlight-green" id="improvement-val">—</span></div>
          <div class="telemetry-row"><span class="telemetry-label">Evaluations</span><span class="telemetry-val" id="evals-val">0</span></div>
          <div class="telemetry-row"><span class="telemetry-label">Promotion</span><span class="telemetry-val" id="promotion-val">—</span></div>
          <div class="telemetry-row"><span class="telemetry-label">External SOTA</span><span class="telemetry-val highlight-amber" id="sota-val">UNVERIFIED</span></div>
        </div>
      </div>
    </aside>

    <main id="os-viewport">
      <div id="three-canvas"></div>
      <div id="metric-panel">
        <div id="metric-title">SPARSE STATE METRIC g = I + C / tr(C)</div>
        <div id="metric-grid">
          <span>1.000</span><span>0.000</span><span>0.000</span>
          <span>0.000</span><span>1.000</span><span>0.000</span>
          <span>0.000</span><span>0.000</span><span>1.000</span>
        </div>
        <div id="metric-note">SPD geometry descriptor derived from measured active-cell covariance; not the Three.js world transform.</div>
      </div>
      <div id="node-tooltip"></div>
      <div id="viewport-overlay">
        <span class="chip">cyan points: authoritative sparse cells</span>
        <span class="chip">meta center: incumbent configuration</span>
        <span class="chip">green/red nodes: measured better/worse replay score</span>
        <span class="chip">grey nodes: unevaluated neighbors</span>
        <span class="chip">shells/rings/particles: visualization only</span>
      </div>
    </main>

    <section id="os-shell" class="glass-panel">
      <div class="shell-header">
        <span>JARVIS-X TRANSACTIONAL SHELL</span>
        <i data-lucide="terminal" style="width:16px;color:#c084fc"></i>
      </div>
      <div class="shell-body" id="shell-body">
        <div class="msg bot">
          <span class="msg-sender">JARVIS-X OS KERNEL</span>
          <div class="msg-bubble">Measured control plane ready. Runtime state, fixed-point residuals, meta-candidate scores and promotion decisions come from the live OS API. Visual shells and flow particles are non-authoritative render layers. Type /help.</div>
        </div>
      </div>
      <div class="shell-input">
        <input type="text" id="shell-input" placeholder="Execute bounded OS command..." autocomplete="off">
        <button class="btn btn-primary" id="send-btn">Send</button>
      </div>
    </section>

    <footer id="os-statusbar">
      <span id="system-status">System: connecting…</span>
      <span>Invariant: PROVISIONAL ≠ AUTHORITATIVE until ΠΛ and transport verification pass</span>
      <span id="scheduler-status">Scheduler: —</span>
    </footer>
  </div>

  <script>
    if (window.lucide) lucide.createIcons();

    let scene, camera, renderer, controls;
    let matrixGroup, metaGroup, singularityCore, innerGlow, particleStreams, particlePositions, voxelPoints;
    let concentricShells = [], fieldRings = [], metaMeshes = [];
    let animTime = 0, lastSnapshotCycle = null, lastStatus = null, lastLattice = null, passPulse = 0;
    let metaArmed = true, metaBusy = false;
    let currentCells = [];

    const raycaster = new THREE.Raycaster();
    const pointer = new THREE.Vector2();
    const $ = (id) => document.getElementById(id);
    const finite = (value) => typeof value === 'number' && Number.isFinite(value);
    const scientific = (value, digits = 3) => finite(Number(value)) ? Number(value).toExponential(digits) : '—';
    const fixed = (value, digits = 6) => finite(Number(value)) ? Number(value).toFixed(digits) : '—';
    const shortHash = (value) => typeof value === 'string' && value.length > 12 ? `${value.slice(0, 8)}…${value.slice(-4)}` : (value || '—');
    const vectorKey = (v) => `${Number(v.compression || 0)},${Number(v.adaptation || 0)},${Number(v.dynamics || 0)}`;

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
      const body = $('shell-body');
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

    function disposeObject(object) {
      if (!object) return;
      object.traverse((obj) => {
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
      controls.dampingFactor = .05;
      controls.minDistance = 7;
      controls.maxDistance = 140;
      scene.add(new THREE.AmbientLight(0x1e293b, 2.0));
      const coreLight = new THREE.PointLight(0xa855f7, 4.5, 90);
      coreLight.position.set(0, 0, 0);
      scene.add(coreLight);
      const cyanLight = new THREE.PointLight(0x00f3ff, 2.4, 80);
      cyanLight.position.set(30, 30, 30);
      scene.add(cyanLight);
      renderer.domElement.addEventListener('pointermove', onPointerMove);
      renderer.domElement.addEventListener('pointerleave', () => $('node-tooltip').style.display = 'none');
      window.addEventListener('resize', onWindowResize);
    }

    function buildVisualArchitecture() {
      if (matrixGroup) {
        scene.remove(matrixGroup);
        disposeObject(matrixGroup);
      }
      matrixGroup = new THREE.Group();
      const depth = parseInt($('depth-slider').value, 10);
      const glow = parseFloat($('glow-slider').value);

      singularityCore = new THREE.Mesh(
        new THREE.IcosahedronGeometry(2.5, 2),
        new THREE.MeshStandardMaterial({ color: 0xa855f7, wireframe: true, emissive: 0x9333ea, emissiveIntensity: .45 + glow * .08 })
      );
      singularityCore.userData.role = 'core';
      matrixGroup.add(singularityCore);

      innerGlow = new THREE.Mesh(
        new THREE.SphereGeometry(1.18, 24, 24),
        new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: Math.min(.88, .20 + glow * .07) })
      );
      innerGlow.userData.role = 'core';
      matrixGroup.add(innerGlow);

      concentricShells = [];
      const colors = [0x00f3ff, 0x38bdf8, 0xa855f7, 0xff00a0, 0x10b981];
      for (let i = 1; i <= depth; i++) {
        const radius = 5 + i * 4.1;
        const color = colors[i % colors.length];
        const shell = new THREE.Mesh(
          new THREE.IcosahedronGeometry(radius, 1),
          new THREE.MeshStandardMaterial({
            color, wireframe: true, transparent: true,
            opacity: Math.max(.07, .34 - i * .03), emissive: color, emissiveIntensity: .16 + glow * .03
          })
        );
        shell.userData.role = i <= Math.ceil(depth / 2) ? 'decoder' : 'encoder';
        matrixGroup.add(shell);
        concentricShells.push(shell);
      }

      fieldRings = [];
      for (let i = 0; i < 3; i++) {
        const ring = new THREE.Mesh(
          new THREE.TorusGeometry(9 + i * 6.2, .08 + glow * .005, 12, 96),
          new THREE.MeshBasicMaterial({
            color: i % 2 === 0 ? 0x00f3ff : 0xff00a0,
            transparent: true,
            opacity: Math.min(.72, .14 + glow * .06)
          })
        );
        ring.rotation.x = Math.PI / (i + 1.2);
        ring.rotation.y = Math.PI / (i + 1.8);
        ring.userData.role = 'visual-field';
        matrixGroup.add(ring);
        fieldRings.push(ring);
      }

      createParticleStreams(depth);
      scene.add(matrixGroup);
      if (currentCells.length) updateVoxelGeometry(currentCells, lastStatus ? lastStatus.side : 1);
      applyHorizonVisibility();
    }

    function createParticleStreams(depth) {
      const count = 720;
      const geom = new THREE.BufferGeometry();
      particlePositions = new Float32Array(count * 3);
      const maxRadius = 5 + depth * 4.1;
      for (let i = 0; i < count; i++) resetParticle(i, maxRadius, true);
      geom.setAttribute('position', new THREE.BufferAttribute(particlePositions, 3));
      particleStreams = new THREE.Points(
        geom,
        new THREE.PointsMaterial({ color: 0x38bdf8, size: .32, transparent: true, opacity: .62 })
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
      currentCells = Array.isArray(cells) ? cells : [];
      if (!matrixGroup) return;
      if (voxelPoints) {
        matrixGroup.remove(voxelPoints);
        disposeObject(voxelPoints);
        voxelPoints = null;
      }
      updateStateMetric(currentCells, side);
      if (!currentCells.length) return;
      const positions = new Float32Array(currentCells.length * 3);
      const colors = new Float32Array(currentCells.length * 3);
      const center = (Math.max(1, side) - 1) / 2;
      const scale = 52 / Math.max(1, side - 1);
      const cyan = new THREE.Color(0x38bdf8);
      const gold = new THREE.Color(0xfbbf24);
      currentCells.forEach((cell, i) => {
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
        new THREE.PointsMaterial({ size: .70, vertexColors: true, transparent: true, opacity: .96, sizeAttenuation: true })
      );
      voxelPoints.userData.role = 'substrate';
      matrixGroup.add(voxelPoints);
      applyHorizonVisibility();
    }

    function updateStateMetric(cells, side) {
      const out = $('metric-grid');
      if (!Array.isArray(cells) || cells.length < 2) {
        out.innerHTML = ['1.000','0.000','0.000','0.000','1.000','0.000','0.000','0.000','1.000'].map(v => `<span>${v}</span>`).join('');
        return;
      }
      const scale = Math.max(1, Number(side) - 1);
      let mx = 0, my = 0, mz = 0;
      const points = cells.map((cell) => {
        const p = [Number(cell.x) / scale - .5, Number(cell.y) / scale - .5, Number(cell.z) / scale - .5];
        mx += p[0]; my += p[1]; mz += p[2];
        return p;
      });
      mx /= points.length; my /= points.length; mz /= points.length;
      const c = [[0,0,0],[0,0,0],[0,0,0]];
      points.forEach((p) => {
        const d = [p[0]-mx, p[1]-my, p[2]-mz];
        for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) c[i][j] += d[i] * d[j];
      });
      for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) c[i][j] /= points.length;
      const trace = Math.max(1e-12, c[0][0] + c[1][1] + c[2][2]);
      const g = c.map((row, i) => row.map((v, j) => (i === j ? 1 : 0) + v / trace));
      out.innerHTML = g.flat().map(v => `<span>${v.toFixed(3)}</span>`).join('');
    }

    function buildMetaLattice(lattice) {
      lastLattice = lattice;
      if (metaGroup) {
        scene.remove(metaGroup);
        disposeObject(metaGroup);
      }
      metaGroup = new THREE.Group();
      metaMeshes = [];
      const nodes = Array.isArray(lattice && lattice.nodes) ? lattice.nodes : [];
      const report = lattice && lattice.last_report;
      const evaluations = report && Array.isArray(report.evaluations) ? report.evaluations : [];
      const measured = new Map();
      const stageRank = { baseline: 3, confirm: 2, probe: 1 };
      evaluations.forEach((item) => {
        const key = vectorKey(item.vector || {});
        const prior = measured.get(key);
        if (!prior || (stageRank[item.stage] || 0) >= (stageRank[prior.stage] || 0)) measured.set(key, item);
      });
      const baselineScore = report && report.baseline && report.baseline.metrics ? Number(report.baseline.metrics.score) : null;
      const bestKey = report && report.best ? vectorKey(report.best.vector || {}) : null;
      const spacing = 5.1;
      const center = new THREE.Vector3(0, 0, 0);

      nodes.forEach((node) => {
        const v = node.vector || {};
        const key = vectorKey(v);
        const pos = new THREE.Vector3(Number(v.compression || 0) * spacing, Number(v.adaptation || 0) * spacing, Number(v.dynamics || 0) * spacing);
        const evaluation = measured.get(key) || null;
        const isCenter = key === '0,0,0';
        const isBest = bestKey === key;
        let color = 0x475569;
        let emissive = 0x0f172a;
        let radius = isCenter ? 1.05 : .72;
        if (isCenter) { color = 0x38bdf8; emissive = 0x075985; }
        if (evaluation && baselineScore !== null) {
          const score = Number(evaluation.metrics.score);
          if (score < baselineScore) { color = 0x4ade80; emissive = 0x14532d; }
          else if (score > baselineScore) { color = 0xfb7185; emissive = 0x7f1d1d; }
          else { color = 0x38bdf8; emissive = 0x075985; }
          if (evaluation.stage === 'confirm') radius *= 1.18;
        }
        if (isBest && evaluation) {
          color = report.promoted ? 0xfbbf24 : 0xc084fc;
          emissive = report.promoted ? 0x92400e : 0x581c87;
          radius *= 1.20;
        }
        const mesh = new THREE.Mesh(
          new THREE.SphereGeometry(radius, 20, 20),
          new THREE.MeshStandardMaterial({ color, emissive, emissiveIntensity: .58, roughness: .30, metalness: .45 })
        );
        mesh.position.copy(pos);
        mesh.userData.metaNode = true;
        mesh.userData.node = node;
        mesh.userData.evaluation = evaluation;
        mesh.userData.isBest = isBest;
        metaGroup.add(mesh);
        metaMeshes.push(mesh);
        if (!isCenter) {
          const lineGeometry = new THREE.BufferGeometry().setFromPoints([center, pos]);
          const line = new THREE.Line(
            lineGeometry,
            new THREE.LineBasicMaterial({ color: evaluation ? 0x64748b : 0x1e293b, transparent: true, opacity: evaluation ? .42 : .18 })
          );
          metaGroup.add(line);
        }
      });
      metaGroup.rotation.x = .12;
      metaGroup.rotation.y = -.18;
      scene.add(metaGroup);
      applyHorizonVisibility();
    }

    function onPointerMove(event) {
      if (!renderer || !camera || !metaMeshes.length) return;
      const rect = renderer.domElement.getBoundingClientRect();
      pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(pointer, camera);
      const hit = raycaster.intersectObjects(metaMeshes, false)[0];
      const tooltip = $('node-tooltip');
      if (!hit) { tooltip.style.display = 'none'; return; }
      const mesh = hit.object;
      const node = mesh.userData.node || {};
      const e = mesh.userData.evaluation;
      const v = node.vector || {};
      const cfg = node.config || {};
      const lines = [
        `ΔC = (${v.compression || 0}, ${v.adaptation || 0}, ${v.dynamics || 0})`,
        `role = ${node.role || 'candidate'}`,
        `block=${cfg.block_size}  q=${cfg.quantization}  prune=${cfg.prune_epsilon}`,
        `lr=${cfg.deep_distiller_learning_rate}  dd_passes=${cfg.deep_distiller_passes}`,
        `contract=${cfg.contraction}  fp_passes=${cfg.fixed_point_passes}`,
      ];
      if (e && e.metrics) {
        lines.push(`stage=${e.stage}  score=${Number(e.metrics.score).toExponential(3)}`);
        lines.push(`rec=${Number(e.metrics.reconstruction_mse).toExponential(2)}  drift=${Number(e.metrics.anchor_drift_mse).toExponential(2)}`);
      } else lines.push('not evaluated in latest bounded search');
      tooltip.textContent = lines.join('\n');
      tooltip.style.display = 'block';
      tooltip.style.left = `${Math.min(rect.width - 300, Math.max(8, event.clientX - rect.left + 14))}px`;
      tooltip.style.top = `${Math.min(rect.height - 150, Math.max(8, event.clientY - rect.top + 14))}px`;
    }

    function applyHorizonVisibility() {
      const target = $('layer-select').value;
      if (singularityCore) singularityCore.visible = target === 'all' || target === 'core' || target === 'encoder' || target === 'decoder';
      if (innerGlow) innerGlow.visible = singularityCore ? singularityCore.visible : false;
      concentricShells.forEach((shell) => shell.visible = target === 'all' || target === shell.userData.role);
      fieldRings.forEach((ring) => ring.visible = target === 'all' || target === 'substrate');
      if (particleStreams) particleStreams.visible = target === 'all' || target === 'encoder' || target === 'decoder';
      if (voxelPoints) voxelPoints.visible = target === 'all' || target === 'substrate';
      if (metaGroup) metaGroup.visible = target === 'all' || target === 'meta';
      const distance = { meta: 24, core: 18, encoder: 38, decoder: 44, substrate: 60, all: 60 }[target] || 60;
      animateCamera(distance);
    }

    function animateCamera(distance) {
      if (!camera || !controls) return;
      const duration = 650;
      const start = camera.position.clone();
      const end = new THREE.Vector3(0, distance * .30, distance);
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
      setText('glow-val', `${parseFloat($('glow-slider').value).toFixed(1)}x`);
    }

    function renderTelemetry(status) {
      lastStatus = status;
      const plane = status.bitplane || {};
      const report = status.last_report || null;
      const transport = status.transport || {};
      const meta = status.meta_optimizer || {};
      const metaReport = meta.last_report || null;
      setText('cells-val', `${status.active_cells || 0} / ${status.logical_cells || 0}`);
      setText('density-val', scientific(Number(plane.density || 0), 3));
      setText('phase-val', report ? scientific(Number(report.phase_velocity), 3) : '—');
      setText('mse-val', report ? scientific(Number(report.reconstruction_mse), 3) : '—');
      setText('dd-val', report && report.distiller_residual_rms !== null ? scientific(Number(report.distiller_residual_rms), 3) : '—');
      setText('fp-val', report && report.fixed_point_residual !== null ? scientific(Number(report.fixed_point_residual), 3) : '—');
      setText('transport-val', transport.encoded_bytes !== undefined ? `${transport.encoded_bytes} B` : '—');
      const integrity = Boolean(status.journal_valid) && Boolean(meta.journal_valid !== false);
      setText('integrity-val', `${shortHash(status.state_hash)} / ${integrity ? 'VALID' : 'INVALID'}`);
      $('integrity-val').className = `telemetry-val ${integrity ? 'highlight-green' : 'highlight-red'}`;

      setText('life-pill', `Lifecycle: ${status.lifecycle}`);
      const dot = $('life-dot');
      dot.className = `pill-dot ${status.lifecycle === 'halted' ? 'halted' : (status.lifecycle !== 'offline' ? 'ready' : '')}`;
      if (report && report.fixed_point_residual !== null) setText('fp-pill', `ΔΨ: ${scientific(Number(report.fixed_point_residual), 2)}`);
      else setText('fp-pill', 'ΔΨ: not measured');
      setText('meta-pill', `Meta epoch: ${meta.epoch || 0}`);
      setText('claim-pill', meta.external_sota_verified ? 'SOTA: verified' : 'SOTA: unverified');

      setText('meta-epoch-val', meta.epoch || 0);
      setText('baseline-score-val', metaReport && metaReport.baseline ? scientific(Number(metaReport.baseline.metrics.score), 3) : '—');
      setText('best-score-val', metaReport && metaReport.best ? scientific(Number(metaReport.best.metrics.score), 3) : '—');
      setText('improvement-val', metaReport ? `${(Number(metaReport.relative_improvement || 0) * 100).toFixed(3)}%` : '—');
      setText('evals-val', metaReport ? metaReport.evaluated_candidates : 0);
      setText('promotion-val', metaReport ? (metaReport.promoted ? 'PROMOTED' : 'RETAINED') : '—');
      $('promotion-val').className = `telemetry-val ${metaReport && metaReport.promoted ? 'highlight-green' : ''}`;
      setText('sota-val', meta.external_sota_verified ? 'VERIFIED' : 'UNVERIFIED');

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

    async function refreshMeta(force = false) {
      if (metaBusy && !force) return;
      const lattice = await api('/v1/os/meta/lattice');
      const priorEpoch = lastLattice ? Number(lastLattice.epoch || 0) : -1;
      const nextEpoch = Number(lattice.epoch || 0);
      if (force || !lastLattice || priorEpoch !== nextEpoch) buildMetaLattice(lattice);
      else lastLattice = lattice;
    }

    async function refresh(forceSnapshot = false, forceMeta = false) {
      try {
        const status = await api('/v1/os/status');
        renderTelemetry(status);
        await refreshSnapshot(forceSnapshot);
        await refreshMeta(forceMeta);
      } catch (error) {
        setText('system-status', `System: API unavailable • ${error.message}`);
      }
    }

    function summarizeResult(result) {
      if (!result || typeof result !== 'object') return String(result || '');
      if (result.report && result.report.best) {
        const r = result.report;
        return `meta_epoch=${result.status && result.status.meta_optimizer ? result.status.meta_optimizer.epoch : '?'}\nevaluations=${r.evaluated_candidates}\nbaseline=${scientific(Number(r.baseline.metrics.score), 3)}\nbest=${scientific(Number(r.best.metrics.score), 3)}\nimprovement=${(Number(r.relative_improvement || 0) * 100).toFixed(3)}%\npromotion=${r.promoted ? 'PROMOTED' : 'RETAINED'}\nclaim=${r.claim_status}`;
      }
      if (Array.isArray(result.reports)) {
        const final = result.reports[result.reports.length - 1];
        return final ? `cycles=${result.reports.length}, committed=${final.committed}, residual=${scientific(Number(final.fixed_point_residual), 2)}` : 'no cycle reports';
      }
      if ('committed' in result) return `committed=${result.committed}, cycle=${result.cycle}, reconstruction_mse=${scientific(Number(result.reconstruction_mse), 2)}`;
      if ('lifecycle' in result) return `lifecycle=${result.lifecycle}, cycle=${result.cycle}, active_cells=${result.active_cells}`;
      return 'operation complete';
    }

    async function action(label, fn, forceSnapshot = true, forceMeta = false) {
      try {
        const result = await fn();
        appendBot(`${label}: accepted by service.\n${summarizeResult(result)}`);
        passPulse = 1;
        await refresh(forceSnapshot, forceMeta);
        return result;
      } catch (error) {
        appendBot(`${label}: ${error.message}`);
        await refresh(false, false);
        return null;
      }
    }

    async function triggerOSPass() {
      if (!lastStatus || !lastStatus.loaded) return appendBot('No sparse field is loaded. Use Boot Demo or /demo first.');
      await action('Verified OS step', () => api('/v1/os/step', { method: 'POST' }));
    }

    async function bootDemo() {
      await action('Boot demo', async () => {
        await api('/v1/os/boot', { method: 'POST' });
        return api('/v1/os/demo', { method: 'POST' });
      }, true, true);
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
      }), false, false);
    }

    async function turnSystemInward() {
      if (!metaArmed) return appendBot('Meta optimizer is disarmed in this console. Re-arm the UI gate first. Server-side Πmeta is never bypassed.');
      if (!lastStatus || !lastStatus.loaded) return appendBot('Load a sparse state before meta-optimization.');
      if (lastStatus.scheduler_running) return appendBot('Stop autorun before meta-optimization so the authoritative state remains fixed during replay.');
      if (metaBusy) return;
      metaBusy = true;
      $('meta-btn').disabled = true;
      $('meta-btn').textContent = '∞ REPLAYING 3D CANDIDATES…';
      appendBot('Turning the runtime inward onto its bounded 3D configuration. The authoritative state remains unchanged while candidates are replayed.');
      try {
        const result = await api('/v1/os/meta/optimize', { method: 'POST' });
        appendBot(`Meta optimization complete.\n${summarizeResult(result)}`);
        passPulse = 1;
        if (result.lattice) buildMetaLattice(result.lattice);
        await refresh(true, true);
        $('layer-select').value = 'meta';
        applyHorizonVisibility();
      } catch (error) {
        appendBot(`Meta optimization rejected: ${error.message}`);
        await refresh(false, false);
      } finally {
        metaBusy = false;
        $('meta-btn').disabled = false;
        $('meta-btn').textContent = '∞ TURN SYSTEM INWARD / OPTIMIZE';
      }
    }

    function toggleMetaArmed() {
      metaArmed = !metaArmed;
      setText('lock-btn', `Meta Gate: ${metaArmed ? 'ARMED' : 'OFF'}`);
      $('meta-btn').disabled = !metaArmed;
    }

    async function processOSCommand(raw) {
      const parts = raw.trim().split(/\s+/);
      const command = (parts[0] || '').toLowerCase();
      if (command === '/help') {
        appendBot('AVAILABLE COMMANDS:\n/status — measured runtime state\n/demo — boot and load demo sparse field\n/step or /matrix — one verified transaction\n/run [n] — run bounded cycles\n/auto [ms] — start autorun\n/stop — stop autorun\n/meta — bounded inward 3D optimization\n/lattice — latest meta lattice summary\n/reset — clear HALTED state\n/sys — hashes, policy and meta state\n/clear — clear shell');
      } else if (command === '/status') {
        await refresh(false, false);
        const s = lastStatus;
        const r = s && s.last_report;
        const m = s && s.meta_optimizer;
        appendBot(s ? `SYSTEM STATUS:\nlifecycle=${s.lifecycle}\ncycle=${s.cycle}\nactive_cells=${s.active_cells}/${s.logical_cells}\nbit_density=${scientific(Number((s.bitplane || {}).density || 0), 3)}\nreconstruction_mse=${r ? scientific(Number(r.reconstruction_mse), 3) : '—'}\ndmdd_residual=${r && r.distiller_residual_rms !== null ? scientific(Number(r.distiller_residual_rms), 3) : '—'}\nfixed_point_delta=${r && r.fixed_point_residual !== null ? scientific(Number(r.fixed_point_residual), 3) : '—'}\nmeta_epoch=${m ? m.epoch : 0}\njournal_valid=${s.journal_valid}` : 'Status unavailable.');
      } else if (command === '/demo') {
        await bootDemo();
      } else if (command === '/step' || command === '/matrix') {
        await triggerOSPass();
      } else if (command === '/run') {
        await runCycles(parts[1] || 8);
      } else if (command === '/auto') {
        await startAutorun(parts[1] || 350);
      } else if (command === '/stop') {
        await action('Autorun stop', () => api('/v1/os/autorun/stop', { method: 'POST' }), false, false);
      } else if (command === '/meta') {
        await turnSystemInward();
      } else if (command === '/lattice') {
        await refreshMeta(true);
        const l = lastLattice || {};
        const r = l.last_report;
        appendBot(r ? `META LATTICE:\nepoch=${l.epoch}\nnodes=${Array.isArray(l.nodes) ? l.nodes.length : 0}\nevaluations=${r.evaluated_candidates}\nbaseline=${scientific(Number(r.baseline.metrics.score), 3)}\nbest=${scientific(Number(r.best.metrics.score), 3)}\npromoted=${r.promoted}\nclaim=${r.claim_status}` : `META LATTICE:\nepoch=${l.epoch || 0}\nnodes=${Array.isArray(l.nodes) ? l.nodes.length : 0}\nno measured search epoch yet`);
      } else if (command === '/reset') {
        await action('Halt reset', () => api('/v1/os/halt/reset', { method: 'POST' }), false, false);
      } else if (command === '/sys') {
        await refresh(false, false);
        const s = lastStatus;
        const m = s && s.meta_optimizer;
        appendBot(s ? `CONTROL STATE:\nmode=${s.mode}\npolicy=${JSON.stringify(s.policy)}\nstate_hash=${s.state_hash}\njournal_head=${s.journal_head}\njournal_valid=${s.journal_valid}\nmeta_epoch=${m ? m.epoch : 0}\nmeta_journal=${m ? m.journal_head : '—'}\nmeta_journal_valid=${m ? m.journal_valid : false}\nexternal_sota_verified=${m ? m.external_sota_verified : false}\nscheduler_running=${s.scheduler_running}\nhalt_reason=${s.halt_reason || 'none'}` : 'Status unavailable.');
      } else if (command === '/clear') {
        $('shell-body').innerHTML = '';
      } else {
        appendBot('This shell is a bounded OS control plane, not a general-purpose language model. Use /help for executable commands.');
      }
    }

    function submitUserMessage() {
      const input = $('shell-input');
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
        singularityCore.rotation.y = animTime * .85;
        singularityCore.rotation.x = animTime * .42;
        singularityCore.scale.setScalar(1 + passPulse * .22);
      }
      concentricShells.forEach((shell, i) => {
        const dir = i % 2 === 0 ? 1 : -1;
        shell.rotation.y = animTime * (.18 + i * .035) * dir;
        shell.rotation.z = animTime * .09 * dir;
      });
      fieldRings.forEach((ring, i) => ring.rotation.z = animTime * (.30 + i * .09));
      if (metaGroup && metaGroup.visible) {
        metaGroup.rotation.y = -.18 + Math.sin(animTime * .15) * .05;
        metaMeshes.forEach((mesh) => {
          if (mesh.userData.isBest) mesh.scale.setScalar(1 + .08 * Math.sin(animTime * 4));
        });
      }
      if (particleStreams && particlePositions) {
        const depth = parseInt($('depth-slider').value, 10);
        const maxRadius = 5 + depth * 4.1;
        const contraction = Math.min(.075, .004 + parseFloat($('velocity-slider').value) * .0055);
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
      $('depth-slider').addEventListener('change', buildVisualArchitecture);
      $('velocity-slider').addEventListener('input', updateVisualLabels);
      $('glow-slider').addEventListener('input', updateVisualLabels);
      $('glow-slider').addEventListener('change', buildVisualArchitecture);
      $('execute-btn').addEventListener('click', triggerOSPass);
      $('meta-btn').addEventListener('click', turnSystemInward);
      $('demo-btn').addEventListener('click', bootDemo);
      $('run-btn').addEventListener('click', () => runCycles(8));
      $('auto-btn').addEventListener('click', () => startAutorun(350));
      $('stop-btn').addEventListener('click', () => action('Autorun stop', () => api('/v1/os/autorun/stop', { method: 'POST' }), false, false));
      $('lock-btn').addEventListener('click', toggleMetaArmed);
      $('shutdown-btn').addEventListener('click', () => action('Shutdown', () => api('/v1/os/shutdown', { method: 'POST' }), false, false));
      $('send-btn').addEventListener('click', submitUserMessage);
      $('shell-input').addEventListener('keydown', (event) => { if (event.key === 'Enter') submitUserMessage(); });
    }

    initThree();
    bindControls();
    updateVisualLabels();
    buildVisualArchitecture();
    animate();
    refresh(true, true);
    setInterval(() => refresh(false, false), 1000);
  </script>
</body>
</html>"""
