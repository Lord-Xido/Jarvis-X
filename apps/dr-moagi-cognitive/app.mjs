import {
  DEFAULT_CONTROL_STATE,
  QUALITY_PROFILES,
  adaptiveQualityDecision,
  boundedPush,
  compileParameterPatch,
  convergenceState,
  describeControlState,
  fieldPointFromOriginal,
  formatQ16_48,
  makeSessionSnapshot,
  normalizeBackendBase,
  normalizeControlState,
  normalizedResidual,
  parseFieldCommand,
  parseSessionSnapshot,
  saturationRatio,
  updateResidualMemory,
} from './core.mjs';

const $ = id => document.getElementById(id);
const STORAGE_KEY = 'dmCognitiveSessionV2';
const MAX_PARTICLES = QUALITY_PROFILES.at(-1).particles;
const MAX_JETS = QUALITY_PROFILES.at(-1).jets;
const RESIDUAL_SAMPLE = 256;
const TELEMETRY_LIMIT = 180;
const COMMAND_HISTORY_LIMIT = 100;
const REQUEST_TIMEOUT_MS = 6000;

let controlState = {...DEFAULT_CONTROL_STATE};
let localGate = 'ACCEPT';
let omegaMemory = 0;
let fixedResidual = 0;
let paused = false;
let backendBase = '';
let qualityMode = 'AUTO';
let qualityIndex = 1;
let commandHistory = [];
let commandCursor = 0;
let collapsed = false;
let appTime = 0;
let selectedParticle = null;
let lastQualityChange = 0;
let lastTelemetryPush = 0;
let auditHead = 'GENESIS';
let auditSequence = 0;
let auditQueue = Promise.resolve();

const backend = {
  state: 'LOCAL ONLY',
  connected: false,
  verified: false,
  booted: false,
  lastError: '',
  lastSeen: 0,
  manifest: null,
};

const telemetry = {fps: [], residual: [], omega: []};

let scene;
let camera;
let renderer;
let controls;
let particles;
let particleGeometry;
let originals;
let positions;
let colors;
let jets;
let jetPositions;
let coreMesh;
let phiRing;
let lambdaShell;
let omegaShell;
let thetaShell;
let raycaster;
let pointer;
let previousSample = null;
let fps = 0;
let frameCounter = 0;
let fpsWindowStart = performance.now();
let lastFrame = performance.now();

function loadSession() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return;
    const session = parseSessionSnapshot(raw);
    controlState = {...session.controlState};
    backendBase = session.backendBase;
    qualityMode = session.qualityMode;
    paused = session.paused;
    commandHistory = [...session.commandHistory];
    commandCursor = commandHistory.length;
    $('tts-toggle').checked = session.ttsEnabled;
  } catch (error) {
    console.warn('Ignoring invalid saved session', error);
    localStorage.removeItem(STORAGE_KEY);
  }
}

function persistSession() {
  try {
    const snapshot = makeSessionSnapshot({
      controlState,
      backendBase,
      qualityMode,
      ttsEnabled: $('tts-toggle')?.checked,
      paused,
      commandHistory,
    });
    localStorage.setItem(STORAGE_KEY, JSON.stringify(snapshot));
  } catch (error) {
    console.warn('Session persistence failed', error);
  }
}

function appendMessage(sender, text, cls = 'cyan') {
  const box = document.createElement('article');
  box.className = 'message';
  const head = document.createElement('header');
  head.className = `message-head ${cls}`;
  const who = document.createElement('span');
  who.textContent = sender;
  const when = document.createElement('span');
  when.textContent = new Date().toLocaleTimeString();
  head.append(who, when);
  const body = document.createElement('pre');
  body.textContent = String(text);
  box.append(head, body);
  $('chat-messages').append(box);
  $('chat-messages').scrollTop = $('chat-messages').scrollHeight;
}

function appendCompiler(text, kind = '') {
  const line = document.createElement('div');
  line.className = kind;
  line.textContent = `[${new Date().toLocaleTimeString()}] ${text}`;
  $('compiler-output').append(line);
  $('compiler-output').scrollTop = $('compiler-output').scrollHeight;
}

function speak(text) {
  if (!$('tts-toggle').checked || !('speechSynthesis' in window)) return;
  speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(String(text).slice(0, 700));
  utterance.rate = 1.02;
  utterance.onstart = () => $('voice-indicator').textContent = 'TTS: SPEAKING';
  utterance.onend = () => $('voice-indicator').textContent = 'TTS: IDLE';
  utterance.onerror = () => $('voice-indicator').textContent = 'TTS: ERROR';
  speechSynthesis.speak(utterance);
}

async function sha256Hex(text) {
  if (!globalThis.crypto?.subtle) return 'UNAVAILABLE';
  const bytes = new TextEncoder().encode(text);
  const digest = await crypto.subtle.digest('SHA-256', bytes);
  return [...new Uint8Array(digest)].map(value => value.toString(16).padStart(2, '0')).join('');
}

function audit(action, details = {}) {
  auditQueue = auditQueue.then(async () => {
    auditSequence += 1;
    const record = JSON.stringify({sequence: auditSequence, action, details, previous: auditHead});
    auditHead = await sha256Hex(record);
    $('audit-head').textContent = auditHead === 'UNAVAILABLE' ? 'UNAVAILABLE' : auditHead.slice(0, 16);
    const line = document.createElement('div');
    line.textContent = `${String(auditSequence).padStart(4, '0')} ${action} · ${auditHead.slice(0, 12)}`;
    $('audit-log').prepend(line);
    while ($('audit-log').children.length > 30) $('audit-log').lastChild.remove();
  });
}

function applyCandidate(candidate, reason) {
  const validated = normalizeControlState(candidate);
  const before = JSON.stringify(controlState);
  controlState = validated;
  localGate = before === JSON.stringify(controlState) ? 'HOLD' : 'ACCEPT';
  updateHUD();
  persistSession();
  audit('LOCAL_PROMOTION', {reason, gate: localGate, controlState});
  return localGate === 'ACCEPT' ? `PI_LOCAL ACCEPT · ${reason}` : 'PI_LOCAL HOLD · no state change';
}

function resetLocalState() {
  controlState = {...DEFAULT_CONTROL_STATE};
  omegaMemory = 0;
  fixedResidual = 0;
  localGate = 'ACCEPT';
  appTime = 0;
  previousSample = null;
  telemetry.fps.length = 0;
  telemetry.residual.length = 0;
  telemetry.omega.length = 0;
  updateHUD();
  updateEditorFromState();
  persistSession();
  audit('LOCAL_RESET');
}

function normalizeQualityMode(mode) {
  const upper = String(mode || 'AUTO').toUpperCase();
  if (!['AUTO', 'LOW', 'MEDIUM', 'HIGH'].includes(upper)) return 'AUTO';
  return upper;
}

function applyQuality(mode, explicitIndex = null) {
  qualityMode = normalizeQualityMode(mode);
  if (explicitIndex !== null) qualityIndex = Math.max(0, Math.min(QUALITY_PROFILES.length - 1, explicitIndex));
  else if (qualityMode !== 'AUTO') qualityIndex = QUALITY_PROFILES.findIndex(profile => profile.name === qualityMode);
  const profile = QUALITY_PROFILES[qualityIndex];
  particleGeometry?.setDrawRange(0, profile.particles);
  jets?.geometry.setDrawRange(0, profile.jets);
  if (renderer) renderer.setPixelRatio(Math.min(devicePixelRatio || 1, profile.pixelRatio));
  $('quality-select').value = qualityMode;
  $('hud-quality').textContent = qualityMode === 'AUTO' ? `AUTO/${profile.name}` : profile.name;
  $('particle-count').textContent = profile.particles.toLocaleString();
  persistSession();
}

function adaptiveQuality(now) {
  if (qualityMode !== 'AUTO' || now - lastQualityChange < 5000 || fps <= 0) return;
  const next = adaptiveQualityDecision({fps, currentIndex: qualityIndex});
  if (next !== qualityIndex) {
    qualityIndex = next;
    lastQualityChange = now;
    applyQuality('AUTO', next);
    appendMessage('RENDERER', `Adaptive quality → ${QUALITY_PROFILES[next].name} (${QUALITY_PROFILES[next].particles.toLocaleString()} particles)`, 'amber');
    audit('QUALITY_CHANGE', {fps, profile: QUALITY_PROFILES[next].name});
  }
}

async function firmwareRequest(path, method = 'GET', body = null) {
  if (!backendBase) throw new Error('No firmware API connected. Use /connect URL or Registers → CONNECT.');
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  try {
    const options = {method, headers: {'Content-Type': 'application/json'}, signal: controller.signal};
    if (body !== null) options.body = JSON.stringify(body);
    const response = await fetch(`${backendBase}${path}`, options);
    let data;
    try { data = await response.json(); } catch { data = {status: response.status}; }
    if (!response.ok) throw new Error(data?.detail || `firmware API HTTP ${response.status}`);
    backend.connected = true;
    backend.lastSeen = Date.now();
    backend.lastError = '';
    return data;
  } catch (error) {
    backend.connected = false;
    backend.lastError = error.name === 'AbortError' ? 'request timeout' : error.message;
    throw error;
  } finally {
    clearTimeout(timeout);
    updateBackendUI();
  }
}

async function connectBackend(value = $('backend-url').value) {
  backendBase = normalizeBackendBase(value);
  if (!backendBase) throw new Error('Enter an absolute firmware API URL.');
  $('backend-url').value = backendBase;
  const health = await firmwareRequest('/healthz');
  backend.state = String(health.status || 'CONNECTED').toUpperCase();
  backend.booted = Boolean(health.booted);
  updateBackendUI();
  persistSession();
  audit('BACKEND_CONNECT', {backendBase, status: backend.state});
  return health;
}

function disconnectBackend() {
  backendBase = '';
  backend.state = 'LOCAL ONLY';
  backend.connected = false;
  backend.verified = false;
  backend.booted = false;
  backend.lastError = '';
  backend.manifest = null;
  updateBackendUI();
  persistSession();
  audit('BACKEND_DISCONNECT');
}

async function refreshBackendStatus(silent = false) {
  if (!backendBase) return null;
  try {
    const status = await firmwareRequest('/v1/firmware/status');
    backend.state = 'CONNECTED';
    backend.booted = Boolean(status.booted);
    updateBackendUI();
    if (!silent) appendMessage('FIRMWARE', JSON.stringify(status, null, 2), 'purple');
    return status;
  } catch (error) {
    backend.state = 'UNREACHABLE';
    updateBackendUI();
    if (!silent) throw error;
    return null;
  }
}

function updateBackendUI() {
  const state = !backendBase ? 'LOCAL ONLY' : backend.connected ? (backend.booted ? 'BOOTED' : backend.verified ? 'VERIFIED' : 'CONNECTED') : backend.state;
  $('backend-state').textContent = state;
  $('backend-state').className = backend.connected ? 'ok' : backendBase ? 'bad' : 'hold';
  $('backend-url').value = backendBase;
  $('hud-backend').textContent = state;
  $('hud-backend').className = backend.connected ? 'ok' : backendBase ? 'bad' : 'hold';
  $('backend-error').textContent = backend.lastError || '—';
}

function localStatusText() {
  const state = describeControlState(controlState, fixedResidual, omegaMemory, localGate);
  const convergence = convergenceState(fixedResidual, omegaMemory);
  return [
    `mode=${state.mode}`,
    `coupling=${state.fieldCoupling.toFixed(3)}`,
    `beta=${state.betaShift.toFixed(3)}`,
    `density=${state.density.toFixed(2)}`,
    `saturation=${(state.saturation * 100).toFixed(1)}%`,
    `DeltaPsi=${state.residual.toExponential(3)}`,
    `Omega=${state.omegaMemory.toExponential(3)}`,
    `PI_LOCAL=${state.gate}`,
    `converged=${convergence.converged}`,
    `quality=${qualityMode}/${QUALITY_PROFILES[qualityIndex].name}`,
    `backend=${backendBase || 'LOCAL ONLY'}`,
  ].join('\n');
}

const HELP_TEXT = `LOCAL COMMANDS
inward | outward
coupling 1.8
beta 1.1
density 30
/status  /pause  /resume  /reset  /snapshot  /export

FIRMWARE COMMANDS
/connect https://firmware.example
/disconnect  /health  /manifest  /verify  /boot  /run 4

All local mutations are validated before promotion. The browser executes no arbitrary source code.`;

async function executeCommand(query) {
  const text = String(query || '').trim();
  if (!text) return '';
  commandHistory = boundedPush(commandHistory, text, COMMAND_HISTORY_LIMIT);
  commandCursor = commandHistory.length;
  persistSession();
  audit('COMMAND', {text});

  if (/^\/help$/i.test(text)) return HELP_TEXT;
  if (/^\/status$/i.test(text)) {
    const remote = backendBase ? await refreshBackendStatus(true) : null;
    return `${localStatusText()}${remote ? `\nremoteBooted=${Boolean(remote.booted)}` : ''}`;
  }
  if (/^\/pause$/i.test(text)) { paused = true; updatePauseUI(); persistSession(); return 'Simulation paused. Rendering remains interactive.'; }
  if (/^\/resume$/i.test(text)) { paused = false; updatePauseUI(); persistSession(); return 'Simulation resumed.'; }
  if (/^\/reset$/i.test(text)) { resetLocalState(); return 'Local authoritative control state restored to canonical defaults.'; }
  if (/^\/snapshot$/i.test(text)) { downloadCanvasSnapshot(); return '3D canvas snapshot exported.'; }
  if (/^\/export$/i.test(text)) { exportSession(); return 'Session snapshot exported.'; }
  if (/^\/disconnect$/i.test(text)) { disconnectBackend(); return 'Firmware backend disconnected. Local runtime remains active.'; }
  if (/^\/connect\s+/i.test(text)) {
    const url = text.replace(/^\/connect\s+/i, '').trim();
    const result = await connectBackend(url);
    return `Firmware backend connected: ${backendBase}\nhealth=${JSON.stringify(result)}`;
  }
  if (/^\/health$/i.test(text)) return JSON.stringify(await firmwareRequest('/healthz'), null, 2);
  if (/^\/manifest$/i.test(text)) {
    const result = await firmwareRequest('/v1/firmware/manifest');
    backend.manifest = result;
    audit('FIRMWARE_MANIFEST', {format: result.format || result.magic || 'unknown'});
    return JSON.stringify(result, null, 2);
  }
  if (/^\/verify$/i.test(text)) {
    const result = await firmwareRequest('/v1/firmware/verify', 'POST');
    backend.verified = Boolean(result.signature_valid ?? true);
    updateBackendUI();
    audit('FIRMWARE_VERIFY', {signatureValid: backend.verified});
    return JSON.stringify(result, null, 2);
  }
  if (/^\/boot$/i.test(text)) {
    const result = await firmwareRequest('/v1/firmware/boot', 'POST');
    backend.booted = Boolean(result.booted ?? true);
    updateBackendUI();
    audit('FIRMWARE_BOOT', {booted: backend.booted});
    return JSON.stringify(result, null, 2);
  }
  const runMatch = text.match(/^\/run(?:\s+(\d+))?$/i);
  if (runMatch) {
    const cycles = Math.min(10000, Math.max(1, Number(runMatch[1] || 1)));
    const result = await firmwareRequest('/v1/firmware/run', 'POST', {cycles});
    backend.booted = true;
    updateBackendUI();
    audit('FIRMWARE_RUN', {cycles});
    return JSON.stringify(result, null, 2);
  }

  const parsed = parseFieldCommand(text, controlState);
  if (!parsed.changed) return `No executable local mutation detected.\n\n${HELP_TEXT}`;
  return applyCandidate(parsed.candidate, parsed.changes.join(', '));
}

async function submitCommand() {
  const input = $('chat-input');
  const query = input.value.trim();
  if (!query) return;
  appendMessage('USER', query, 'cyan');
  input.value = '';
  $('send-btn').disabled = true;
  $('send-btn').textContent = 'PROCESSING';
  try {
    const response = await executeCommand(query);
    appendMessage('DM-vΩΞ⁺ CONTROL', response, 'purple');
    speak(response.split('\n').slice(0, 4).join('. '));
  } catch (error) {
    localGate = 'HOLD';
    updateHUD();
    appendMessage('SYSTEM ERROR', error.message, 'red');
    audit('COMMAND_REJECT', {error: error.message});
  } finally {
    $('send-btn').disabled = false;
    $('send-btn').textContent = 'TRANSMIT';
  }
}

function compileEditor() {
  appendCompiler('Validating parameter candidate...');
  try {
    const report = compileParameterPatch($('code-editor').value, controlState);
    const result = applyCandidate(report.candidate, report.assignments.join(', '));
    appendCompiler(result, localGate === 'ACCEPT' ? 'ok' : 'hold');
  } catch (error) {
    localGate = 'HOLD';
    updateHUD();
    appendCompiler(`PI_LOCAL HOLD · ${error.message}`, 'bad');
    audit('PATCH_REJECT', {error: error.message});
  }
}

function updateEditorFromState() {
  $('code-editor').value = `# Bounded control parameters only.\nmode = ${controlState.inversionMode}\ncoupling = ${controlState.fieldCoupling}\nbeta = ${controlState.betaShift}\ndensity = ${controlState.density}\n`;
}

function updatePauseUI() {
  $('pause-btn').textContent = paused ? '▶ RESUME' : 'Ⅱ PAUSE';
  $('runtime-state').textContent = paused ? 'PAUSED' : 'RUNNING';
  $('runtime-dot').className = `status-dot ${paused ? 'amber-bg' : 'cyan-bg pulse'}`;
}

function updateHUD() {
  const state = describeControlState(controlState, fixedResidual, omegaMemory, localGate);
  $('hud-mode').textContent = state.mode;
  $('hud-coupling').textContent = state.fieldCoupling.toFixed(3);
  $('hud-beta').textContent = state.betaShift.toFixed(3);
  $('hud-sat').textContent = `${(state.saturation * 100).toFixed(1)}%`;
  $('hud-residual').textContent = state.residual.toExponential(3);
  $('hud-omega').textContent = state.omegaMemory.toExponential(3);
  $('hud-gate').textContent = localGate;
  $('hud-gate').className = localGate === 'ACCEPT' ? 'ok' : 'hold';
  $('chat-residual').textContent = state.residual.toExponential(3);
  $('reg-phi').textContent = formatQ16_48(state.fieldCoupling);
  $('reg-k1').textContent = formatQ16_48(state.betaShift);
  $('reg-density').textContent = formatQ16_48(controlState.density);
  $('reg-state').textContent = state.mode === 'INWARD' ? 'Psi → Phi · INWARD' : 'Omega → Theta · OUTWARD';
  $('reg-gate').textContent = localGate;
  $('convergence-state').textContent = convergenceState(fixedResidual, omegaMemory).converged ? 'FIXED-POINT TOLERANCE' : 'EVOLVING';
  updateBackendUI();
}

function mulberry32(seed) {
  return function random() {
    let t = seed += 0x6D2B79F5;
    t = Math.imul(t ^ t >>> 15, t | 1);
    t ^= t + Math.imul(t ^ t >>> 7, t | 61);
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

function init3D() {
  if (!globalThis.THREE) throw new Error('Three.js failed to load. Check network access and reload.');
  const container = $('canvas-container');
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x02040a);
  scene.fog = new THREE.FogExp2(0x02040a, 0.012);
  camera = new THREE.PerspectiveCamera(58, 1, 0.1, 300);
  camera.position.set(31, 22, 40);
  renderer = new THREE.WebGLRenderer({antialias: true, alpha: false, powerPreference: 'high-performance'});
  renderer.outputEncoding = THREE.sRGBEncoding;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.2;
  container.appendChild(renderer.domElement);
  controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.045;
  controls.minDistance = 5;
  controls.maxDistance = 130;

  scene.add(new THREE.AmbientLight(0x172554, 1.7));
  const cyanLight = new THREE.PointLight(0x22d3ee, 3.5, 70); cyanLight.position.set(0, 4, 0); scene.add(cyanLight);
  const purpleLight = new THREE.PointLight(0xa855f7, 2.5, 80); purpleLight.position.set(16, 20, 10); scene.add(purpleLight);
  const grid = new THREE.GridHelper(80, 40, 0x164e63, 0x0b2233); grid.position.y = -22; scene.add(grid);

  buildField();
  buildLayers();
  buildJets();
  raycaster = new THREE.Raycaster();
  raycaster.params.Points.threshold = 0.55;
  pointer = new THREE.Vector2();
  renderer.domElement.addEventListener('pointerdown', inspectPointer);
  window.addEventListener('resize', resize3D);
  resize3D();
  applyQuality(qualityMode);
}

function buildField() {
  const rng = mulberry32(0xD00D2026);
  originals = new Float32Array(MAX_PARTICLES * 3);
  positions = new Float32Array(MAX_PARTICLES * 3);
  colors = new Float32Array(MAX_PARTICLES * 3);
  const c1 = new THREE.Color(0x22d3ee);
  const c2 = new THREE.Color(0x8b5cf6);
  const c3 = new THREE.Color(0xec4899);
  const color = new THREE.Color();
  for (let i = 0; i < MAX_PARTICLES; i += 1) {
    const u = rng() * Math.PI * 2;
    const v = rng() * Math.PI * 2;
    const noise = (rng() - 0.5) * 2.4;
    const major = 18;
    const minor = 7.5 + noise;
    const x = (major + minor * Math.cos(v)) * Math.cos(u);
    const y = minor * Math.sin(v);
    const z = (major + minor * Math.cos(v)) * Math.sin(u);
    const j = i * 3;
    originals[j] = positions[j] = x;
    originals[j + 1] = positions[j + 1] = y;
    originals[j + 2] = positions[j + 2] = z;
    const mix = Math.min(1, Math.hypot(x, y, z) / 29);
    color.copy(c1).lerp(c2, mix);
    if (i % 17 === 0) color.lerp(c3, 0.42);
    colors[j] = color.r; colors[j + 1] = color.g; colors[j + 2] = color.b;
  }
  particleGeometry = new THREE.BufferGeometry();
  particleGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  particleGeometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  const material = new THREE.PointsMaterial({size: 0.2, vertexColors: true, transparent: true, opacity: 0.86, blending: THREE.AdditiveBlending, depthWrite: false});
  particles = new THREE.Points(particleGeometry, material);
  scene.add(particles);
}

function buildLayers() {
  coreMesh = new THREE.Mesh(new THREE.IcosahedronGeometry(2.1, 4), new THREE.MeshStandardMaterial({color: 0xec4899, wireframe: true, emissive: 0x22d3ee, emissiveIntensity: 1.25}));
  scene.add(coreMesh);
  phiRing = new THREE.Mesh(new THREE.TorusGeometry(11.5, 0.28, 12, 96), new THREE.MeshBasicMaterial({color: 0xa855f7, wireframe: true, transparent: true, opacity: 0.75}));
  phiRing.rotation.x = Math.PI / 2;
  scene.add(phiRing);
  lambdaShell = new THREE.Mesh(new THREE.IcosahedronGeometry(6.5, 2), new THREE.MeshBasicMaterial({color: 0x22d3ee, wireframe: true, transparent: true, opacity: 0.13}));
  omegaShell = new THREE.Mesh(new THREE.SphereGeometry(9, 24, 14), new THREE.MeshBasicMaterial({color: 0xec4899, wireframe: true, transparent: true, opacity: 0.08}));
  thetaShell = new THREE.Mesh(new THREE.TorusKnotGeometry(4.2, 0.08, 90, 8, 2, 3), new THREE.MeshBasicMaterial({color: 0xf59e0b, transparent: true, opacity: 0.42}));
  scene.add(lambdaShell, omegaShell, thetaShell);
}

function buildJets() {
  const rng = mulberry32(0xA17E2026);
  jetPositions = new Float32Array(MAX_JETS * 3);
  const jetColors = new Float32Array(MAX_JETS * 3);
  for (let i = 0; i < MAX_JETS; i += 1) {
    const direction = rng() > 0.5 ? 1 : -1;
    const y = rng() * 27 * direction;
    const spread = Math.abs(y) / 27 * 6 + 0.15;
    const angle = rng() * Math.PI * 2;
    const j = i * 3;
    jetPositions[j] = Math.cos(angle) * spread;
    jetPositions[j + 1] = y;
    jetPositions[j + 2] = Math.sin(angle) * spread;
    jetColors[j] = 0.96; jetColors[j + 1] = 0.62; jetColors[j + 2] = 0.08;
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(jetPositions, 3));
  geometry.setAttribute('color', new THREE.BufferAttribute(jetColors, 3));
  jets = new THREE.Points(geometry, new THREE.PointsMaterial({size: 0.18, vertexColors: true, transparent: true, opacity: 0.75, blending: THREE.AdditiveBlending, depthWrite: false}));
  scene.add(jets);
}

function resize3D() {
  if (!renderer) return;
  const rect = $('canvas-container').getBoundingClientRect();
  camera.aspect = Math.max(0.1, rect.width / Math.max(1, rect.height));
  camera.updateProjectionMatrix();
  renderer.setSize(rect.width, rect.height, false);
}

function inspectPointer(event) {
  const rect = renderer.domElement.getBoundingClientRect();
  pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(pointer, camera);
  const hit = raycaster.intersectObject(particles, false)[0];
  if (!hit || hit.index >= QUALITY_PROFILES[qualityIndex].particles) return;
  selectedParticle = hit.index;
  const j = hit.index * 3;
  const x = positions[j]; const y = positions[j + 1]; const z = positions[j + 2];
  $('selected-particle').textContent = `#${hit.index} · (${x.toFixed(2)}, ${y.toFixed(2)}, ${z.toFixed(2)}) · r=${Math.hypot(x, y, z).toFixed(2)}`;
}

function updateField() {
  const profile = QUALITY_PROFILES[qualityIndex];
  for (let i = 0; i < profile.particles; i += 1) {
    const j = i * 3;
    const p = fieldPointFromOriginal(originals[j], originals[j + 1], originals[j + 2], appTime, controlState);
    positions[j] = p[0]; positions[j + 1] = p[1]; positions[j + 2] = p[2];
  }
  particleGeometry.attributes.position.needsUpdate = true;

  const speed = Math.max(0.05, controlState.fieldCoupling);
  for (let i = 0; i < profile.jets; i += 1) {
    const j = i * 3;
    let y = jetPositions[j + 1];
    if (controlState.inversionMode === 'INWARD') {
      y -= y * 0.035 * speed;
      if (Math.abs(y) < 0.7) y = (i % 2 ? 1 : -1) * 27;
    } else {
      const direction = y >= 0 ? 1 : -1;
      y += (0.25 + Math.abs(y) * 0.018) * speed * direction;
      if (Math.abs(y) > 29) y = direction * 0.5;
    }
    jetPositions[j + 1] = y;
  }
  jets.geometry.attributes.position.needsUpdate = true;

  particles.rotation.y = appTime * 0.035 * speed;
  coreMesh.rotation.set(appTime * 0.35, appTime * 0.52, 0);
  const pulse = 1 + saturationRatio(controlState) * 0.5 + omegaMemory * 0.3;
  coreMesh.scale.setScalar(Math.min(2.5, pulse));
  phiRing.rotation.z = -appTime * 0.8 * speed;
  lambdaShell.rotation.y = appTime * 0.12;
  omegaShell.rotation.x = -appTime * 0.09;
  thetaShell.rotation.y = appTime * 0.2;
}

function sampleResidual() {
  const profile = QUALITY_PROFILES[qualityIndex];
  const sample = new Float64Array(RESIDUAL_SAMPLE * 3);
  const step = Math.max(1, Math.floor(profile.particles / RESIDUAL_SAMPLE));
  for (let i = 0; i < RESIDUAL_SAMPLE; i += 1) {
    const source = Math.min(profile.particles - 1, i * step) * 3;
    const target = i * 3;
    sample[target] = positions[source];
    sample[target + 1] = positions[source + 1];
    sample[target + 2] = positions[source + 2];
  }
  fixedResidual = previousSample ? normalizedResidual(previousSample, sample) : 0;
  omegaMemory = updateResidualMemory(omegaMemory, fixedResidual, 0.9);
  previousSample = sample;
  updateHUD();
}

function pushTelemetry(now) {
  if (now - lastTelemetryPush < 500) return;
  lastTelemetryPush = now;
  telemetry.fps = boundedPush(telemetry.fps, fps, TELEMETRY_LIMIT);
  telemetry.residual = boundedPush(telemetry.residual, fixedResidual, TELEMETRY_LIMIT);
  telemetry.omega = boundedPush(telemetry.omega, omegaMemory, TELEMETRY_LIMIT);
  drawTelemetry();
}

function drawSeries(ctx, values, width, height, strokeStyle, maxOverride = null) {
  if (values.length < 2) return;
  const max = maxOverride ?? Math.max(...values, 1e-9);
  ctx.strokeStyle = strokeStyle;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  values.forEach((value, index) => {
    const x = index / (values.length - 1) * width;
    const y = height - Math.min(1, value / max) * height;
    if (index === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();
}

function drawTelemetry() {
  const canvas = $('telemetry-chart');
  const rect = canvas.getBoundingClientRect();
  const ratio = Math.min(2, devicePixelRatio || 1);
  canvas.width = Math.max(1, Math.floor(rect.width * ratio));
  canvas.height = Math.max(1, Math.floor(rect.height * ratio));
  const ctx = canvas.getContext('2d');
  ctx.scale(ratio, ratio);
  const width = rect.width; const height = rect.height;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = '#020617'; ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = '#122033'; ctx.lineWidth = 1;
  for (let i = 1; i < 4; i += 1) { const y = i * height / 4; ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(width, y); ctx.stroke(); }
  drawSeries(ctx, telemetry.fps, width, height, '#4ade80', 80);
  const residualMax = Math.max(...telemetry.residual, ...telemetry.omega, 1e-6);
  drawSeries(ctx, telemetry.residual, width, height, '#22d3ee', residualMax);
  drawSeries(ctx, telemetry.omega, width, height, '#ec4899', residualMax);
}

function renderFrame(now) {
  const dt = Math.min(0.05, (now - lastFrame) / 1000);
  lastFrame = now;
  if (!paused) {
    appTime += dt;
    updateField();
    frameCounter += 1;
    if (frameCounter % 8 === 0) sampleResidual();
  }
  controls.update();
  renderer.render(scene, camera);

  const elapsed = now - fpsWindowStart;
  if (elapsed >= 500) {
    fps = frameCounter * 1000 / elapsed;
    frameCounter = 0;
    fpsWindowStart = now;
    $('fps-counter').textContent = `${fps.toFixed(0)} FPS`;
    adaptiveQuality(now);
  }
  pushTelemetry(now);
  requestAnimationFrame(renderFrame);
}

function downloadCanvasSnapshot() {
  renderer.render(scene, camera);
  const link = document.createElement('a');
  link.download = `dm-cognitive-${new Date().toISOString().replace(/[:.]/g, '-')}.png`;
  link.href = renderer.domElement.toDataURL('image/png');
  link.click();
  audit('CANVAS_SNAPSHOT');
}

function exportSession() {
  const snapshot = makeSessionSnapshot({controlState, backendBase, qualityMode, ttsEnabled: $('tts-toggle').checked, paused, commandHistory});
  const payload = {...snapshot, telemetry, auditHead, exportedAt: new Date().toISOString()};
  const blob = new Blob([JSON.stringify(payload, null, 2)], {type: 'application/json'});
  const link = document.createElement('a');
  link.href = URL.createObjectURL(blob);
  link.download = `dm-cognitive-session-${Date.now()}.json`;
  link.click();
  setTimeout(() => URL.revokeObjectURL(link.href), 1000);
  audit('SESSION_EXPORT');
}

async function importSession(file) {
  const text = await file.text();
  const session = parseSessionSnapshot(text);
  controlState = {...session.controlState};
  backendBase = session.backendBase;
  qualityMode = session.qualityMode;
  paused = session.paused;
  commandHistory = [...session.commandHistory];
  commandCursor = commandHistory.length;
  $('tts-toggle').checked = session.ttsEnabled;
  applyQuality(qualityMode);
  updatePauseUI();
  updateEditorFromState();
  updateHUD();
  persistSession();
  audit('SESSION_IMPORT');
  if (backendBase) refreshBackendStatus(true);
}

function setTab(name) {
  for (const tab of ['chat', 'code', 'registers', 'telemetry']) {
    $(`tab-${tab}`).classList.toggle('hidden', tab !== name);
    $(`tab-${tab}-btn`).classList.toggle('active', tab === name);
  }
  if (name === 'telemetry') drawTelemetry();
}

function toggleCollapse() {
  collapsed = !collapsed;
  $('control-plane').classList.toggle('collapsed', collapsed);
  $('expand-btn').classList.toggle('hidden', !collapsed);
  setTimeout(resize3D, 320);
}

function resetCamera() {
  camera.position.set(31, 22, 40);
  controls.target.set(0, 0, 0);
  controls.update();
}

function setupEvents() {
  for (const tab of ['chat', 'code', 'registers', 'telemetry']) $(`tab-${tab}-btn`).onclick = () => setTab(tab);
  $('collapse-btn').onclick = toggleCollapse;
  $('expand-btn').onclick = toggleCollapse;
  $('send-btn').onclick = submitCommand;
  $('chat-input').addEventListener('keydown', event => {
    if (event.key === 'Enter') { event.preventDefault(); submitCommand(); }
    else if (event.key === 'ArrowUp' && commandHistory.length) {
      event.preventDefault(); commandCursor = Math.max(0, commandCursor - 1); $('chat-input').value = commandHistory[commandCursor] || '';
    } else if (event.key === 'ArrowDown' && commandHistory.length) {
      event.preventDefault(); commandCursor = Math.min(commandHistory.length, commandCursor + 1); $('chat-input').value = commandHistory[commandCursor] || '';
    }
  });
  $('compile-btn').onclick = compileEditor;
  $('code-editor').addEventListener('keydown', event => { if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') { event.preventDefault(); compileEditor(); } });
  $('connect-btn').onclick = async () => {
    try { const result = await connectBackend(); appendMessage('FIRMWARE', `Connected. ${JSON.stringify(result)}`, 'purple'); }
    catch (error) { appendMessage('SYSTEM ERROR', error.message, 'red'); updateBackendUI(); }
  };
  $('disconnect-btn').onclick = disconnectBackend;
  $('pause-btn').onclick = () => { paused = !paused; updatePauseUI(); persistSession(); audit(paused ? 'PAUSE' : 'RESUME'); };
  $('camera-btn').onclick = resetCamera;
  $('snapshot-btn').onclick = downloadCanvasSnapshot;
  $('fullscreen-btn').onclick = () => document.fullscreenElement ? document.exitFullscreen() : $('workspace').requestFullscreen?.();
  $('quality-select').onchange = event => applyQuality(event.target.value);
  $('export-btn').onclick = exportSession;
  $('import-btn').onclick = () => $('import-file').click();
  $('import-file').onchange = async event => {
    const file = event.target.files?.[0]; if (!file) return;
    try { await importSession(file); appendMessage('SYSTEM', 'Session imported and validated.', 'green'); }
    catch (error) { appendMessage('SYSTEM ERROR', `Import rejected: ${error.message}`, 'red'); }
    event.target.value = '';
  };
  $('tts-toggle').onchange = persistSession;
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) lastFrame = performance.now();
  });
}

async function registerServiceWorker() {
  if (!('serviceWorker' in navigator) || location.protocol === 'file:') return;
  try {
    const registration = await navigator.serviceWorker.register('./sw.js');
    $('pwa-state').textContent = registration.active ? 'OFFLINE SHELL READY' : 'PWA REGISTERED';
  } catch (error) {
    $('pwa-state').textContent = 'PWA UNAVAILABLE';
    console.warn('Service worker registration failed', error);
  }
}

function boot() {
  loadSession();
  setupEvents();
  updateEditorFromState();
  updatePauseUI();
  updateBackendUI();
  appendMessage('DM-vΩΞ⁺ CONTROL', `Operational control plane initialized.\n${HELP_TEXT}`, 'purple');
  try {
    init3D();
    updateHUD();
    requestAnimationFrame(renderFrame);
    audit('WEBAPP_BOOT', {qualityMode});
  } catch (error) {
    $('canvas-fallback').classList.remove('hidden');
    $('canvas-fallback').textContent = `3D renderer unavailable: ${error.message}`;
    appendMessage('SYSTEM ERROR', error.message, 'red');
  }
  if (backendBase) refreshBackendStatus(true);
  setInterval(() => { if (backendBase) refreshBackendStatus(true); }, 5000);
  registerServiceWorker();
}

boot();
