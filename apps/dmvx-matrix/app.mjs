import { DMVXMatrixRuntime, base64ToBytes, bytesToBase64 } from './core.mjs';

const ROM_KEY = 'dmvx-matrix-rom-v1';
const LEDGER_KEY = 'dmvx-matrix-ledger-v1';
const MAX_LEDGER_ENTRIES = 24;
const MAX_RENDERED_CELLS = 4096;

const runtime = new DMVXMatrixRuntime({
  logicalExtent: 1000,
  maxActiveCells: MAX_RENDERED_CELLS,
  activeCells: 2048,
  latentDim: 64,
  reconstructionTolerance: 0.42,
});

const state = {
  auto: true,
  simulationTime: 0,
  drive: 0.5,
  previousFrame: performance.now(),
  smoothedFrameMs: 16.7,
  lastTickAt: 0,
  tickPeriodMs: 650,
  previousReceiptHash: 'GENESIS',
  ledger: [],
  lastReceipt: null,
};

const elements = Object.fromEntries([
  'runtime-status', 'frame-state', 'm-active', 'm-bytes', 'm-compression',
  'm-codec', 'm-rmse', 'm-throughput', 'm-frame', 'm-version', 'drive',
  'drive-value', 'step', 'toggle-auto', 'inward', 'save-rom', 'load-rom',
  'export-rom', 'reset', 'ledger', 'ledger-head', 'commit-banner', 'viewport',
].map((id) => [id, document.getElementById(id)]));

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KiB`;
  return `${(bytes / 1024 ** 2).toFixed(2)} MiB`;
}

function formatNumber(value) {
  return new Intl.NumberFormat('en', { maximumFractionDigits: 1 }).format(value);
}

async function sha256Hex(text) {
  const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(text));
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, '0')).join('');
}

async function appendReceipt(receipt, source = 'runtime') {
  const record = {
    ...receipt,
    source,
    timestamp: new Date().toISOString(),
    previousReceiptHash: state.previousReceiptHash,
  };
  const hash = await sha256Hex(JSON.stringify(record));
  const sealed = { ...record, hash };
  state.previousReceiptHash = hash;
  state.ledger.unshift(sealed);
  state.ledger = state.ledger.slice(0, MAX_LEDGER_ENTRIES);
  localStorage.setItem(LEDGER_KEY, JSON.stringify(state.ledger));
  renderLedger();
  return sealed;
}

function restoreLedger() {
  try {
    const parsed = JSON.parse(localStorage.getItem(LEDGER_KEY) || '[]');
    if (Array.isArray(parsed)) {
      state.ledger = parsed.slice(0, MAX_LEDGER_ENTRIES);
      state.previousReceiptHash = state.ledger[0]?.hash || 'GENESIS';
    }
  } catch {
    state.ledger = [];
  }
  renderLedger();
}

function renderLedger() {
  elements.ledger.textContent = '';
  for (const record of state.ledger) {
    const item = document.createElement('li');
    if (!record.committed) item.className = 'reject';
    const status = document.createElement('strong');
    status.textContent = record.committed ? 'COMMIT' : 'ROLLBACK';
    item.append(status, document.createTextNode(
      ` v${record.version} · RMSE ${Number(record.reconstructionDistance ?? 0).toFixed(5)} · `,
    ));
    const code = document.createElement('code');
    code.textContent = String(record.hash || '').slice(0, 14);
    item.append(code);
    elements.ledger.append(item);
  }
  elements.ledgerHead.textContent = state.previousReceiptHash.slice(0, 12);
}

function saveROM({ announce = true } = {}) {
  const bytes = runtime.encodeROM();
  localStorage.setItem(ROM_KEY, bytesToBase64(bytes));
  if (announce) {
    elements.commitBanner.textContent = `ROM saved · ${formatBytes(bytes.byteLength)} · CRC verified on load`;
    elements.commitBanner.className = 'commit-banner commit';
  }
  return bytes;
}

async function loadROM({ automatic = false } = {}) {
  const encoded = localStorage.getItem(ROM_KEY);
  if (!encoded) {
    if (!automatic) {
      elements.commitBanner.textContent = 'No persisted ROM image found';
      elements.commitBanner.className = 'commit-banner rollback';
    }
    return false;
  }
  try {
    const result = runtime.decodeROM(base64ToBytes(encoded));
    const receipt = {
      committed: true,
      version: result.version,
      previousVersion: result.version,
      step: result.step,
      reconstructionDistance: 0,
      elapsedMs: 0,
      activeCells: result.activeCells,
      latentDim: result.latentDim,
      residentBytes: runtime.residentBytes(),
    };
    await appendReceipt(receipt, automatic ? 'autoboot-rom' : 'manual-rom');
    elements.commitBanner.textContent = `ROM loaded transactionally · ${formatBytes(result.bytes)} · state v${result.version}`;
    elements.commitBanner.className = 'commit-banner commit';
    updateMetrics();
    updateInstances();
    return true;
  } catch (error) {
    elements.commitBanner.textContent = `ROM rejected · ${error.message}`;
    elements.commitBanner.className = 'commit-banner rollback';
    return false;
  }
}

async function executeTick(source = 'manual') {
  const receipt = runtime.tick({ timeSeconds: state.simulationTime, drive: state.drive });
  state.lastReceipt = receipt;
  const sealed = await appendReceipt(receipt, source);
  const status = receipt.committed ? 'COMMIT' : 'ROLLBACK';
  elements.commitBanner.textContent = `${status} · state v${receipt.version} · RMSE ${receipt.reconstructionDistance.toFixed(5)} · receipt ${sealed.hash.slice(0, 12)}`;
  elements.commitBanner.className = `commit-banner ${receipt.committed ? 'commit' : 'rollback'}`;
  if (receipt.committed && receipt.version % 10 === 0) saveROM({ announce: false });
  updateMetrics();
  updateInstances();
}

function updateMetrics() {
  const receipt = state.lastReceipt;
  elements['m-active'].textContent = `${formatNumber(runtime.activeCells)} / ${formatNumber(runtime.logicalCellCount)}`;
  elements['m-bytes'].textContent = formatBytes(runtime.residentBytes());
  elements['m-compression'].textContent = `${runtime.logicalCompressionRatio().toExponential(3)} : 1`;
  elements['m-codec'].textContent = receipt ? `${receipt.elapsedMs.toFixed(3)} ms` : '—';
  elements['m-rmse'].textContent = receipt ? receipt.reconstructionDistance.toFixed(6) : '—';
  elements['m-throughput'].textContent = receipt && receipt.elapsedMs > 0
    ? `${formatNumber(receipt.activeCells / (receipt.elapsedMs / 1000))} cells/s`
    : '—';
  elements['m-frame'].textContent = `${state.smoothedFrameMs.toFixed(2)} ms`;
  elements['m-version'].textContent = String(runtime.version);
  elements.frameState.textContent = `${(1000 / state.smoothedFrameMs).toFixed(1)} FPS`;
}

function exportROM() {
  const bytes = saveROM({ announce: false });
  const blob = new Blob([bytes], { type: 'application/octet-stream' });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = `dmvx-matrix-v${runtime.version}.bin`;
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 0);
  elements.commitBanner.textContent = `ROM exported · ${formatBytes(bytes.byteLength)}`;
  elements.commitBanner.className = 'commit-banner commit';
}

function bindControls() {
  elements.drive.addEventListener('input', () => {
    state.drive = Number(elements.drive.value);
    elements['drive-value'].textContent = state.drive.toFixed(2);
  });
  elements.step.addEventListener('click', () => executeTick('manual-step'));
  elements['toggle-auto'].addEventListener('click', () => {
    state.auto = !state.auto;
    elements['toggle-auto'].textContent = state.auto ? 'Pause auto' : 'Resume auto';
    elements['runtime-status'].textContent = state.auto ? 'AUTO RUNNING' : 'PAUSED';
    elements['runtime-status'].className = `pill ${state.auto ? 'good' : ''}`;
  });
  elements.inward.addEventListener('click', async () => {
    const result = runtime.turnInward(2);
    elements.commitBanner.textContent = result.changed
      ? `Inward active-set reduction committed · ${runtime.activeCells} resident cells`
      : `Inward boundary reached · latent dimension ${runtime.latentDim}`;
    elements.commitBanner.className = 'commit-banner commit';
    await executeTick('inward-turn');
  });
  elements['save-rom'].addEventListener('click', () => saveROM());
  elements['load-rom'].addEventListener('click', () => loadROM());
  elements['export-rom'].addEventListener('click', exportROM);
  elements.reset.addEventListener('click', async () => {
    localStorage.removeItem(ROM_KEY);
    localStorage.removeItem(LEDGER_KEY);
    location.reload();
  });
}

if (!globalThis.THREE) {
  throw new Error('Three.js failed to load');
}

const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x03060b, 0.026);
const camera = new THREE.PerspectiveCamera(58, 1, 0.1, 200);
camera.position.set(0, 4, 27);
camera.lookAt(0, 0, 0);
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
elements.viewport.append(renderer.domElement);

const group = new THREE.Group();
scene.add(group);
scene.add(new THREE.AmbientLight(0x46617f, 1.2));
const keyLight = new THREE.PointLight(0x39d6ff, 12, 70);
keyLight.position.set(12, 15, 16);
scene.add(keyLight);
const fillLight = new THREE.PointLight(0xb58cff, 9, 70);
fillLight.position.set(-14, -9, 8);
scene.add(fillLight);

const boundary = new THREE.Mesh(
  new THREE.BoxGeometry(22, 22, 22),
  new THREE.MeshBasicMaterial({ color: 0x39d6ff, wireframe: true, transparent: true, opacity: 0.13 }),
);
group.add(boundary);

const core = new THREE.Mesh(
  new THREE.IcosahedronGeometry(1.25, 2),
  new THREE.MeshBasicMaterial({ color: 0xffd166, wireframe: true, transparent: true, opacity: 0.75 }),
);
group.add(core);

const nodeGeometry = new THREE.IcosahedronGeometry(0.075, 0);
const nodeMaterial = new THREE.MeshBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.85 });
const instances = new THREE.InstancedMesh(nodeGeometry, nodeMaterial, MAX_RENDERED_CELLS);
instances.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
group.add(instances);
const dummy = new THREE.Object3D();
const color = new THREE.Color();

function updateInstances() {
  for (let index = 0; index < runtime.activeCells; index += 1) {
    const x = runtime.positions[index * 3] * 10;
    const y = runtime.positions[index * 3 + 1] * 10;
    const z = runtime.positions[index * 3 + 2] * 10;
    const value = runtime.committed[index];
    const scale = 0.65 + value * 1.75;
    dummy.position.set(x, y, z);
    dummy.scale.setScalar(scale);
    dummy.updateMatrix();
    instances.setMatrixAt(index, dummy.matrix);
    color.setHSL(0.54 - value * 0.2, 0.88, 0.48 + value * 0.12);
    instances.setColorAt(index, color);
  }
  instances.count = runtime.activeCells;
  instances.instanceMatrix.needsUpdate = true;
  if (instances.instanceColor) instances.instanceColor.needsUpdate = true;
}

let dragging = false;
let previousPointer = { x: 0, y: 0 };
renderer.domElement.addEventListener('pointerdown', (event) => {
  dragging = true;
  previousPointer = { x: event.clientX, y: event.clientY };
  renderer.domElement.setPointerCapture(event.pointerId);
});
renderer.domElement.addEventListener('pointermove', (event) => {
  if (!dragging) return;
  group.rotation.y += (event.clientX - previousPointer.x) * 0.006;
  group.rotation.x += (event.clientY - previousPointer.y) * 0.006;
  previousPointer = { x: event.clientX, y: event.clientY };
});
renderer.domElement.addEventListener('pointerup', (event) => {
  dragging = false;
  renderer.domElement.releasePointerCapture(event.pointerId);
});
renderer.domElement.addEventListener('pointercancel', () => { dragging = false; });
renderer.domElement.addEventListener('wheel', (event) => {
  event.preventDefault();
  camera.position.z = Math.min(44, Math.max(16, camera.position.z + event.deltaY * 0.018));
}, { passive: false });

function resize() {
  const width = elements.viewport.clientWidth;
  const height = elements.viewport.clientHeight;
  renderer.setSize(width, height, false);
  camera.aspect = width / Math.max(height, 1);
  camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(elements.viewport);

function animate(now) {
  requestAnimationFrame(animate);
  const deltaMs = Math.min(now - state.previousFrame, 100);
  state.previousFrame = now;
  state.smoothedFrameMs = state.smoothedFrameMs * 0.92 + deltaMs * 0.08;
  const deltaSeconds = deltaMs / 1000;
  state.simulationTime += deltaSeconds;

  if (!dragging) {
    group.rotation.y += deltaSeconds * 0.07;
    group.rotation.x = Math.sin(state.simulationTime * 0.14) * 0.08;
  }
  core.rotation.x -= deltaSeconds * 0.7;
  core.rotation.y += deltaSeconds * 0.9;
  const pulse = 1 + Math.sin(state.simulationTime * 3) * 0.12;
  core.scale.setScalar(pulse);

  if (state.auto && now - state.lastTickAt >= state.tickPeriodMs) {
    state.lastTickAt = now;
    executeTick('auto').catch((error) => {
      elements.commitBanner.textContent = `Runtime error · ${error.message}`;
      elements.commitBanner.className = 'commit-banner rollback';
      state.auto = false;
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
