import {
  MODALITY,
  MultimodalROM3D,
  base64ToBytes,
  bytesToBase64,
} from './core.mjs';

const ROM_KEY = 'jarvisx-dm3r-v2';
const LEDGER_KEY = 'jarvisx-dm3r-ledger-v2';
const MAX_RENDERED = 4096;
const runtime = new MultimodalROM3D({
  logicalExtent: 1_000_000,
  maxActiveCells: MAX_RENDERED,
  activeCells: 3072,
  latentDim: 96,
  reconstructionTolerance: 0.45,
});

const state = {
  auto: true,
  time: 0,
  drive: 0.55,
  lastTickAt: 0,
  tickPeriodMs: 800,
  previousFrame: performance.now(),
  smoothedFrameMs: 16.7,
  receiptHash: 'GENESIS',
  ledger: [],
  lastReceipt: null,
  useDemoFrames: false,
};

const ids = [
  'runtime-status', 'frame-state', 'm-active', 'm-bytes', 'm-ratio', 'm-rmse',
  'm-fixed', 'm-latency', 'm-throughput', 'm-version', 'm-rom', 'drive',
  'drive-value', 'step', 'toggle-auto', 'inward', 'demo', 'save-rom', 'load-rom',
  'export-rom', 'reset', 'ledger', 'ledger-head', 'commit-banner', 'viewport',
];
const el = Object.fromEntries(ids.map((id) => [id, document.getElementById(id)]));

function formatBytes(bytes) {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1024 ** 2) return (bytes / 1024).toFixed(1) + ' KiB';
  return (bytes / 1024 ** 2).toFixed(2) + ' MiB';
}

function formatNumber(value) {
  return new Intl.NumberFormat('en', { maximumFractionDigits: 1 }).format(value);
}

async function sha256(text) {
  const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(text));
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, '0')).join('');
}

async function appendReceipt(receipt, source) {
  const record = {
    ...receipt,
    source,
    previousReceiptHash: state.receiptHash,
    timestamp: new Date().toISOString(),
  };
  const hash = await sha256(JSON.stringify(record));
  const sealed = { ...record, hash };
  state.receiptHash = hash;
  state.ledger.unshift(sealed);
  state.ledger = state.ledger.slice(0, 24);
  localStorage.setItem(LEDGER_KEY, JSON.stringify(state.ledger));
  renderLedger();
}

function renderLedger() {
  el.ledger.textContent = '';
  for (const receipt of state.ledger) {
    const item = document.createElement('li');
    item.className = receipt.committed ? '' : 'reject';
    const strong = document.createElement('strong');
    strong.textContent = receipt.committed ? 'COMMIT' : 'ROLLBACK';
    item.append(strong, document.createTextNode(
      ' v' + receipt.version +
      ' · RMSE ' + Number(receipt.reconstructionDistance || 0).toFixed(5) +
      ' · fp ' + Number(receipt.fixedPointResidual || 0).toExponential(2) +
      ' · ' + String(receipt.hash || '').slice(0, 10)
    ));
    el.ledger.append(item);
  }
  el['ledger-head'].textContent = state.receiptHash.slice(0, 12);
}

function restoreLedger() {
  try {
    const records = JSON.parse(localStorage.getItem(LEDGER_KEY) || '[]');
    if (Array.isArray(records)) {
      state.ledger = records.slice(0, 24);
      state.receiptHash = state.ledger[0]?.hash || 'GENESIS';
    }
  } catch {
    state.ledger = [];
  }
  renderLedger();
}

function demoFrames() {
  const n = 512;
  const wave = (phase, frequency) => Array.from(
    { length: n },
    (_, i) => 0.5 + 0.45 * Math.sin(phase + i * frequency),
  );
  return [
    { modality: 'text', values: wave(state.time * 0.3, 0.031) },
    { modality: 'audio', values: wave(state.time * 1.2, 0.073) },
    { modality: 'image', values: wave(state.time * 0.5, 0.047) },
    { modality: 'video', values: wave(state.time * 0.8, 0.059) },
    { modality: 'tensor', values: wave(state.time * 0.2, 0.019) },
    { modality: 'sensor', values: wave(state.time * 0.9, 0.089) },
  ];
}

async function executeTick(source = 'manual') {
  const options = { timeSeconds: state.time, drive: state.drive };
  if (state.useDemoFrames) options.frames = demoFrames();
  const receipt = runtime.tick(options);
  state.lastReceipt = receipt;
  await appendReceipt(receipt, source);

  el['commit-banner'].textContent =
    (receipt.committed ? 'COMMIT' : 'ROLLBACK') +
    ' · v' + receipt.version +
    ' · RMSE ' + receipt.reconstructionDistance.toFixed(5) +
    ' · fixed ' + receipt.fixedPointResidual.toExponential(2);
  el['commit-banner'].className = 'commit-banner ' + (receipt.committed ? 'commit' : 'rollback');
  updateMetrics();
  updateInstances();
}

function saveROM({ announce = true } = {}) {
  const bytes = runtime.encodeROM();
  localStorage.setItem(ROM_KEY, bytesToBase64(bytes));
  if (announce) {
    el['commit-banner'].textContent = 'ROM saved · ' + formatBytes(bytes.byteLength) + ' · CRC32 protected';
    el['commit-banner'].className = 'commit-banner commit';
  }
  updateMetrics();
  return bytes;
}

async function loadROM({ automatic = false } = {}) {
  const encoded = localStorage.getItem(ROM_KEY);
  if (!encoded) return false;
  try {
    const result = runtime.decodeROM(base64ToBytes(encoded));
    await appendReceipt({
      committed: true,
      version: result.version,
      previousVersion: result.version,
      reconstructionDistance: 0,
      fixedPointResidual: 0,
      elapsedMs: 0,
      activeCells: result.activeCells,
      latentDim: result.latentDim,
      residentBytes: runtime.residentBytes(),
    }, automatic ? 'autoboot-rom' : 'manual-rom');
    el['commit-banner'].textContent = 'ROM loaded transactionally · ' + formatBytes(result.bytes);
    el['commit-banner'].className = 'commit-banner commit';
    updateInstances();
    updateMetrics();
    return true;
  } catch (error) {
    el['commit-banner'].textContent = 'ROM rejected · ' + error.message;
    el['commit-banner'].className = 'commit-banner rollback';
    return false;
  }
}

function exportROM() {
  const bytes = saveROM({ announce: false });
  const url = URL.createObjectURL(new Blob([bytes], { type: 'application/octet-stream' }));
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = 'dr-moagi-multimodal-v' + runtime.version + '.dm3r';
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 0);
}

function updateMetrics() {
  const receipt = state.lastReceipt;
  el['m-active'].textContent = formatNumber(runtime.activeCells);
  el['m-bytes'].textContent = formatBytes(runtime.residentBytes());
  el['m-ratio'].textContent = runtime.logicalCompressionRatio().toExponential(3) + ' : 1';
  el['m-rmse'].textContent = receipt ? receipt.reconstructionDistance.toFixed(6) : '—';
  el['m-fixed'].textContent = receipt ? receipt.fixedPointResidual.toExponential(3) : '—';
  el['m-latency'].textContent = receipt ? receipt.elapsedMs.toFixed(3) + ' ms' : '—';
  el['m-throughput'].textContent = receipt && receipt.elapsedMs > 0
    ? formatNumber(receipt.activeCells / (receipt.elapsedMs / 1000)) + ' cells/s'
    : '—';
  el['m-version'].textContent = String(runtime.version);
  el['m-rom'].textContent = formatBytes(runtime.encodeROM().byteLength);
  el['frame-state'].textContent = (1000 / state.smoothedFrameMs).toFixed(1) + ' FPS';
}

function bindControls() {
  el.drive.addEventListener('input', () => {
    state.drive = Number(el.drive.value);
    el['drive-value'].textContent = state.drive.toFixed(2);
  });
  el.step.addEventListener('click', () => executeTick('manual-step'));
  el['toggle-auto'].addEventListener('click', () => {
    state.auto = !state.auto;
    el['toggle-auto'].textContent = state.auto ? 'Pause auto' : 'Resume auto';
    el['runtime-status'].textContent = state.auto ? 'AUTO RUNNING' : 'PAUSED';
    el['runtime-status'].className = 'pill ' + (state.auto ? 'good' : '');
  });
  el.inward.addEventListener('click', async () => {
    const result = runtime.turnInward(2);
    el['commit-banner'].textContent = result.changed
      ? 'Sparse support contracted · ' + result.activeCells + ' resident cells'
      : 'Inward boundary reached · latent floor ' + runtime.latentDim;
    updateInstances();
    await executeTick('inward-turn');
  });
  el.demo.addEventListener('click', async () => {
    state.useDemoFrames = !state.useDemoFrames;
    el.demo.textContent = state.useDemoFrames ? 'Use synthetic world' : 'Ingest multimodal demo';
    await executeTick(state.useDemoFrames ? 'multimodal-ingest' : 'synthetic-ingest');
  });
  el['save-rom'].addEventListener('click', () => saveROM());
  el['load-rom'].addEventListener('click', () => loadROM());
  el['export-rom'].addEventListener('click', exportROM);
  el.reset.addEventListener('click', () => {
    localStorage.removeItem(ROM_KEY);
    localStorage.removeItem(LEDGER_KEY);
    location.reload();
  });
}

if (!globalThis.THREE) throw new Error('Three.js failed to load');

const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x02050a, 0.025);
const camera = new THREE.PerspectiveCamera(58, 1, 0.1, 200);
camera.position.set(0, 3.5, 27);
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
el.viewport.append(renderer.domElement);

const world = new THREE.Group();
scene.add(world);
scene.add(new THREE.AmbientLight(0x486580, 1.2));
const key = new THREE.PointLight(0x43d9ff, 13, 70);
key.position.set(10, 14, 12);
scene.add(key);
const fill = new THREE.PointLight(0xb786ff, 8, 70);
fill.position.set(-11, -7, 10);
scene.add(fill);

const boundary = new THREE.Mesh(
  new THREE.BoxGeometry(22, 22, 22),
  new THREE.MeshBasicMaterial({ color: 0x43d9ff, wireframe: true, transparent: true, opacity: 0.12 }),
);
world.add(boundary);

for (const radius of [2.0, 3.4, 5.2]) {
  const ring = new THREE.Mesh(
    new THREE.TorusGeometry(radius, 0.025, 8, 160),
    new THREE.MeshBasicMaterial({ color: 0x8e7dff, transparent: true, opacity: 0.28 }),
  );
  ring.rotation.x = Math.PI / 2;
  world.add(ring);
}

const core = new THREE.Mesh(
  new THREE.IcosahedronGeometry(1.2, 2),
  new THREE.MeshBasicMaterial({ color: 0xffd166, wireframe: true, transparent: true, opacity: 0.85 }),
);
world.add(core);

const geometry = new THREE.IcosahedronGeometry(0.065, 0);
const material = new THREE.MeshBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.88 });
const instances = new THREE.InstancedMesh(geometry, material, MAX_RENDERED);
instances.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
world.add(instances);

const dummy = new THREE.Object3D();
const color = new THREE.Color();
const palette = new Map([
  [MODALITY.text, 0x43d9ff],
  [MODALITY.audio, 0x52e3a4],
  [MODALITY.image, 0xffd166],
  [MODALITY.video, 0xff7fa5],
  [MODALITY.tensor, 0xb786ff],
  [MODALITY.sensor, 0xff9f43],
]);

function kineticScale(phase) {
  if (phase < 0.34) return 1 - (phase / 0.34) * 0.82;
  if (phase < 0.55) return 0.18;
  if (phase < 0.90) return 0.18 + ((phase - 0.55) / 0.35) * 0.67;
  return 0.85 + ((phase - 0.90) / 0.10) * 0.15;
}

function updateInstances() {
  const phase = (state.time * 0.13) % 1;
  const radiusScale = kineticScale(phase);
  for (let index = 0; index < runtime.activeCells; index += 1) {
    const p = runtime.normalizedPosition(index);
    const residualKick = runtime.residual[index] * 0.8;
    const swirl = Math.sin(state.time * 1.7 + index * 0.011) * (1 - radiusScale) * 0.3;
    dummy.position.set(
      (p.x * radiusScale + swirl * p.y) * 10,
      (p.y * radiusScale - swirl * p.x) * 10,
      (p.z * radiusScale + residualKick * 0.15) * 10,
    );
    const activation = runtime.committed[index];
    dummy.scale.setScalar(0.65 + activation * 1.45);
    dummy.updateMatrix();
    instances.setMatrixAt(index, dummy.matrix);
    color.setHex(palette.get(runtime.modality[index]) || 0xffffff);
    color.offsetHSL(0, 0, (activation - 0.5) * 0.16);
    instances.setColorAt(index, color);
  }
  instances.count = runtime.activeCells;
  instances.instanceMatrix.needsUpdate = true;
  if (instances.instanceColor) instances.instanceColor.needsUpdate = true;
}

let dragging = false;
let pointer = { x: 0, y: 0 };
renderer.domElement.addEventListener('pointerdown', (event) => {
  dragging = true;
  pointer = { x: event.clientX, y: event.clientY };
  renderer.domElement.setPointerCapture(event.pointerId);
});
renderer.domElement.addEventListener('pointermove', (event) => {
  if (!dragging) return;
  world.rotation.y += (event.clientX - pointer.x) * 0.006;
  world.rotation.x += (event.clientY - pointer.y) * 0.006;
  pointer = { x: event.clientX, y: event.clientY };
});
renderer.domElement.addEventListener('pointerup', (event) => {
  dragging = false;
  renderer.domElement.releasePointerCapture(event.pointerId);
});
renderer.domElement.addEventListener('wheel', (event) => {
  event.preventDefault();
  camera.position.z = Math.min(44, Math.max(15, camera.position.z + event.deltaY * 0.018));
}, { passive: false });

function resize() {
  const width = el.viewport.clientWidth;
  const height = el.viewport.clientHeight;
  renderer.setSize(width, height, false);
  camera.aspect = width / Math.max(height, 1);
  camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(el.viewport);

function animate(now) {
  requestAnimationFrame(animate);
  const deltaMs = Math.min(100, now - state.previousFrame);
  state.previousFrame = now;
  state.smoothedFrameMs = state.smoothedFrameMs * 0.92 + deltaMs * 0.08;
  state.time += deltaMs / 1000;

  if (!dragging) world.rotation.y += deltaMs / 1000 * 0.055;
  core.rotation.x -= deltaMs / 1000 * 0.8;
  core.rotation.y += deltaMs / 1000 * 1.0;
  core.scale.setScalar(1 + Math.sin(state.time * 3.2) * 0.1);

  updateInstances();
  if (state.auto && now - state.lastTickAt >= state.tickPeriodMs) {
    state.lastTickAt = now;
    executeTick('auto').catch((error) => {
      state.auto = false;
      el['runtime-status'].textContent = 'HALTED';
      el['commit-banner'].textContent = 'Runtime error · ' + error.message;
      el['commit-banner'].className = 'commit-banner rollback';
    });
  }

  updateMetrics();
  renderer.render(scene, camera);
}

restoreLedger();
bindControls();
updateInstances();
updateMetrics();
await loadROM({ automatic: true });
requestAnimationFrame(animate);
