import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { CSS2DRenderer, CSS2DObject } from "three/addons/renderers/CSS2DRenderer.js";
import { JarvisXRuntime, TraceBus } from "./runtime-core.mjs";

const SOURCE_CITIES = [
  { name: "Beijing", country: "China", lat: 39.9, lon: 116.4, pop: 21_710_000, elev: 43, tz: 480 },
  { name: "New York", country: "USA", lat: 40.7, lon: -74.0, pop: 8_419_000, elev: 10, tz: -240 },
  { name: "London", country: "UK", lat: 51.5, lon: -0.1, pop: 8_982_000, elev: 11, tz: 0 },
  { name: "Tokyo", country: "Japan", lat: 35.7, lon: 139.7, pop: 13_960_000, elev: 40, tz: 540 },
  { name: "Shanghai", country: "China", lat: 31.2, lon: 121.5, pop: 24_180_000, elev: 4, tz: 480 },
  { name: "Moscow", country: "Russia", lat: 55.8, lon: 37.6, pop: 11_920_000, elev: 156, tz: 180 },
  { name: "Sydney", country: "Australia", lat: -33.9, lon: 151.2, pop: 5_312_000, elev: 58, tz: 600 },
  { name: "Cape Town", country: "South Africa", lat: -33.9, lon: 18.4, pop: 4_337_000, elev: 5, tz: 120 },
  { name: "Johannesburg", country: "South Africa", lat: -26.2, lon: 28.0, pop: 5_635_000, elev: 1753, tz: 120 },
  { name: "Polokwane", country: "South Africa", lat: -23.9, lon: 29.5, pop: 628_999, elev: 1312, tz: 120 },
  { name: "Rio de Janeiro", country: "Brazil", lat: -22.9, lon: -43.2, pop: 6_760_000, elev: 2, tz: -180 },
  { name: "Dubai", country: "UAE", lat: 25.2, lon: 55.3, pop: 3_395_000, elev: 0, tz: 240 },
  { name: "Delhi", country: "India", lat: 28.6, lon: 77.2, pop: 16_790_000, elev: 216, tz: 330 },
  { name: "Cairo", country: "Egypt", lat: 30.0, lon: 31.2, pop: 9_500_000, elev: 23, tz: 120 },
];

const byId = (id) => document.getElementById(id);
const trace = new TraceBus(512);
const runtime = new JarvisXRuntime(SOURCE_CITIES, {
  trace,
  cache: { maxEntries: 128, ttlMs: 300_000 },
  neural: { inputSize: 12, hiddenSize: 16, outputSize: 4, learningRate: 0.025 },
});

const ui = {
  voice: false,
  busy: false,
  fps: 60,
  quality: "high",
  traceEvents: [],
  echoCount: 0,
  lastQualityChange: performance.now(),
  neuralPulse: 0,
  networkPulse: 0,
};

function emitEcho(text, x = window.innerWidth * 0.5, y = window.innerHeight * 0.45) {
  const layer = byId("echo-layer");
  const echo = document.createElement("span");
  echo.className = "echo-particle";
  echo.textContent = `⟡ ${text}`;
  echo.style.left = `${Math.max(10, Math.min(window.innerWidth - 180, x))}px`;
  echo.style.top = `${Math.max(40, Math.min(window.innerHeight - 80, y))}px`;
  layer.appendChild(echo);
  ui.echoCount += 1;
  window.setTimeout(() => {
    echo.remove();
    ui.echoCount = Math.max(0, ui.echoCount - 1);
  }, 2700);
}

function addMessage(text, sender, metaTag = "") {
  const container = byId("chat-messages");
  const message = document.createElement("div");
  message.className = `msg ${sender}`;
  const body = document.createElement("span");
  body.textContent = text;
  message.appendChild(body);
  if (metaTag) {
    const metadata = document.createElement("span");
    metadata.className = "meta-tag";
    metadata.textContent = metaTag;
    message.appendChild(metadata);
  }
  container.appendChild(message);
  while (container.childElementCount > 60) container.firstElementChild.remove();
  container.scrollTop = container.scrollHeight;
}

function speak(text) {
  if (!ui.voice || !("speechSynthesis" in window)) return;
  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.rate = 0.96;
  utterance.pitch = 0.92;
  window.speechSynthesis.speak(utterance);
}

function traceLabel(event) {
  const payload = event.payload;
  if (event.stage === "INSTRUCTION_RESOLVED") return payload.name;
  if (event.stage === "INSTRUCTION_EXECUTED") return `${payload.name} · ${payload.latency.toFixed(2)}ms`;
  if (event.stage === "NEURAL_BACKPROP") return `loss ${payload.loss.toFixed(3)}`;
  if (event.stage === "STATE_TRANSITION") return `${payload.previous} → ${payload.next}`;
  if (event.stage === "ROM_REFINEMENT_COMMITTED") return `${payload.before} → ${payload.after}`;
  if (event.stage === "INSTRUCTION_MUTATION_COMMITTED") return `ISA v${payload.version}`;
  if (event.stage === "ALIAS_LEARNED") return `${payload.alias} → ${payload.city}`;
  return event.stage.toLowerCase().replaceAll("_", " ");
}

trace.subscribe((event) => {
  ui.traceEvents.unshift(event);
  ui.traceEvents.length = Math.min(ui.traceEvents.length, 8);
  ui.neuralPulse = Math.min(1.8, ui.neuralPulse + 0.28);
  ui.networkPulse = Math.min(2, ui.networkPulse + 0.35);
  if (["INSTRUCTION_RESOLVED", "NEURAL_BACKPROP", "ROM_REFINEMENT_COMMITTED", "INSTRUCTION_MUTATION_COMMITTED", "ALIAS_LEARNED"].includes(event.stage)) {
    emitEcho(traceLabel(event), 90 + Math.random() * Math.max(100, window.innerWidth * 0.35), 130 + Math.random() * 160);
  }
  updateTelemetry();
});

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x05070a);
scene.fog = new THREE.FogExp2(0x05070a, 0.055);

const camera = new THREE.PerspectiveCamera(31, window.innerWidth / window.innerHeight, 0.1, 100);
camera.position.set(0, 0.45, 3.35);

let renderer;
try {
  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: "high-performance" });
} catch (error) {
  addMessage("WebGL could not initialize in this browser.", "bot", "renderer unavailable");
  throw error;
}
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.2;
renderer.outputEncoding = THREE.sRGBEncoding;
document.body.prepend(renderer.domElement);

const labelRenderer = new CSS2DRenderer();
labelRenderer.setSize(window.innerWidth, window.innerHeight);
Object.assign(labelRenderer.domElement.style, { position: "absolute", inset: "0", pointerEvents: "none" });
document.body.prepend(labelRenderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.autoRotate = true;
controls.autoRotateSpeed = 0.45;
controls.minDistance = 1.55;
controls.maxDistance = 8;
controls.enablePan = false;

scene.add(new THREE.AmbientLight(0x203050, 0.58));
const sunLight = new THREE.DirectionalLight(0xffeedd, 1.75);
sunLight.position.set(2, 3, 4);
scene.add(sunLight, sunLight.target);
const rimLight = new THREE.DirectionalLight(0x4488ff, 0.45);
rimLight.position.set(-3, -1, -2);
scene.add(rimLight);

const earthGroup = new THREE.Group();
scene.add(earthGroup);
const textureLoader = new THREE.TextureLoader();
textureLoader.setCrossOrigin("anonymous");
const mapTexture = textureLoader.load("https://threejs.org/examples/textures/planets/earth_atmos_2048.jpg");
const normalTexture = textureLoader.load("https://threejs.org/examples/textures/planets/earth_normal_2048.jpg");
const specularTexture = textureLoader.load("https://threejs.org/examples/textures/planets/earth_specular_2048.jpg");
const cloudTexture = textureLoader.load("https://threejs.org/examples/textures/planets/earth_clouds_1024.png");

const earthMaterial = new THREE.MeshPhongMaterial({
  map: mapTexture,
  normalMap: normalTexture,
  normalScale: new THREE.Vector2(1.2, 1.2),
  specularMap: specularTexture,
  specular: new THREE.Color(0x333333),
  shininess: 5,
});
const earth = new THREE.Mesh(new THREE.SphereGeometry(1, 80, 80), earthMaterial);
earthGroup.add(earth);

const cloudMaterial = new THREE.MeshPhongMaterial({
  map: cloudTexture,
  transparent: true,
  opacity: 0.2,
  blending: THREE.AdditiveBlending,
  side: THREE.DoubleSide,
  depthWrite: false,
});
const clouds = new THREE.Mesh(new THREE.SphereGeometry(1.008, 64, 64), cloudMaterial);
earthGroup.add(clouds);

const glowMaterial = new THREE.MeshPhongMaterial({
  color: 0x4a8cff,
  transparent: true,
  opacity: 0.1,
  side: THREE.BackSide,
  depthWrite: false,
});
const glowMesh = new THREE.Mesh(new THREE.SphereGeometry(1.025, 48, 48), glowMaterial);
earthGroup.add(glowMesh);

function fibonacciSphere(count, radius) {
  const points = [];
  const golden = Math.PI * (3 - Math.sqrt(5));
  for (let index = 0; index < count; index += 1) {
    const y = 1 - index / Math.max(1, count - 1) * 2;
    const radial = Math.sqrt(Math.max(0, 1 - y * y));
    const theta = golden * index;
    points.push(new THREE.Vector3(Math.cos(theta) * radial * radius, y * radius, Math.sin(theta) * radial * radius));
  }
  return points;
}

const mirrorPoints = fibonacciSphere(30, 1.34);
const pointGeometry = new THREE.BufferGeometry().setFromPoints(mirrorPoints);
const pointMaterial = new THREE.PointsMaterial({ color: 0x60f0ff, size: 0.022, transparent: true, opacity: 0.58, blending: THREE.AdditiveBlending });
const neuralPoints = new THREE.Points(pointGeometry, pointMaterial);
scene.add(neuralPoints);

const connectionVertices = [];
for (let index = 0; index < mirrorPoints.length; index += 1) {
  for (const offset of [1, 5]) {
    const target = mirrorPoints[(index + offset) % mirrorPoints.length];
    connectionVertices.push(mirrorPoints[index].x, mirrorPoints[index].y, mirrorPoints[index].z, target.x, target.y, target.z);
  }
}
const connectionGeometry = new THREE.BufferGeometry();
connectionGeometry.setAttribute("position", new THREE.Float32BufferAttribute(connectionVertices, 3));
const connectionMaterial = new THREE.LineBasicMaterial({ color: 0x3b82f6, transparent: true, opacity: 0.12, blending: THREE.AdditiveBlending });
const neuralConnections = new THREE.LineSegments(connectionGeometry, connectionMaterial);
scene.add(neuralConnections);

function latLonToPosition(lat, lon, radius = 1.02) {
  const phi = (90 - lat) * Math.PI / 180;
  const theta = lon * Math.PI / 180;
  return new THREE.Vector3(
    radius * Math.sin(phi) * Math.cos(theta),
    radius * Math.cos(phi),
    radius * Math.sin(phi) * Math.sin(theta),
  );
}

const markerGroup = new THREE.Group();
scene.add(markerGroup);
const markers = [];
for (const city of SOURCE_CITIES) {
  const position = latLonToPosition(city.lat, city.lon);
  const dotElement = document.createElement("div");
  Object.assign(dotElement.style, {
    width: "6px", height: "6px", background: "#60f0ff", borderRadius: "50%",
    boxShadow: "0 0 16px rgba(96,240,255,.45)", border: "1px solid rgba(255,255,255,.35)", pointerEvents: "none",
  });
  const dot = new CSS2DObject(dotElement);
  dot.position.copy(position);
  markerGroup.add(dot);

  const labelElement = document.createElement("div");
  labelElement.textContent = city.name;
  Object.assign(labelElement.style, {
    color: "rgba(255,255,255,.72)", fontSize: "9px", background: "rgba(0,0,0,.25)",
    padding: "1px 6px", borderRadius: "8px", border: "1px solid rgba(255,255,255,.04)",
    transform: "translateY(-12px)", pointerEvents: "none", whiteSpace: "nowrap",
  });
  const label = new CSS2DObject(labelElement);
  label.position.copy(position);
  markerGroup.add(label);
  markers.push({ city, dot, label });
}

function highlightMarkers(predicate) {
  for (const marker of markers) {
    const selected = predicate(marker.city);
    marker.dot.element.style.background = selected ? "#fcd34d" : "#60f0ff";
    marker.dot.element.style.boxShadow = selected ? "0 0 30px rgba(252,211,77,.7)" : "0 0 16px rgba(96,240,255,.45)";
    marker.label.element.style.color = selected ? "#fcd34d" : "rgba(255,255,255,.72)";
    marker.label.element.style.fontWeight = selected ? "700" : "400";
  }
}

function dayOfYearUTC(date) {
  const start = Date.UTC(date.getUTCFullYear(), 0, 0);
  const current = Date.UTC(date.getUTCFullYear(), date.getUTCMonth(), date.getUTCDate());
  return Math.floor((current - start) / 86_400_000);
}

function updateSun() {
  const now = new Date();
  const hours = now.getUTCHours() + now.getUTCMinutes() / 60;
  const angle = hours / 24 * Math.PI * 2;
  const declination = 23.44 * Math.sin((dayOfYearUTC(now) - 81) * (2 * Math.PI / 365));
  sunLight.position.set(5 * Math.sin(angle), 1.2 + Math.sin(declination * Math.PI / 180) * 1.5, 5 * Math.cos(angle));
  sunLight.target.position.set(0, 0, 0);
  sunLight.target.updateMatrixWorld();
}

let flyTarget = null;
let autoRotateTimer = null;
function flyToCity(city) {
  const end = latLonToPosition(city.lat, city.lon, 1.6).normalize().multiplyScalar(2.2);
  flyTarget = { start: camera.position.clone(), end, progress: 0 };
  controls.autoRotate = false;
  window.clearTimeout(autoRotateTimer);
  autoRotateTimer = window.setTimeout(() => { controls.autoRotate = true; }, 2400);
}

function setQuality(level) {
  if (!["high", "medium", "low"].includes(level) || level === ui.quality) return;
  if (performance.now() - ui.lastQualityChange < 3000) return;
  ui.lastQualityChange = performance.now();
  ui.quality = level;
  const segments = level === "high" ? 80 : level === "medium" ? 48 : 32;
  earth.geometry.dispose();
  earth.geometry = new THREE.SphereGeometry(1, segments, segments);
  clouds.visible = level !== "low";
  glowMesh.visible = level !== "low";
  neuralConnections.visible = level !== "low";
  renderer.setPixelRatio(level === "high" ? Math.min(window.devicePixelRatio, 2) : 1);
  trace.emit("LOD_CHANGED", { level, fps: ui.fps });
}

const neuralCanvas = byId("neural-canvas");
const neuralContext = neuralCanvas.getContext("2d");
function drawNeuralActivity() {
  const snapshot = runtime.snapshot().neural;
  const width = neuralCanvas.width;
  const height = neuralCanvas.height;
  neuralContext.clearRect(0, 0, width, height);
  neuralContext.strokeStyle = "rgba(96,240,255,.08)";
  neuralContext.lineWidth = 1;
  for (let row = 1; row < 4; row += 1) {
    const y = row * height / 4;
    neuralContext.beginPath(); neuralContext.moveTo(0, y); neuralContext.lineTo(width, y); neuralContext.stroke();
  }
  const gradient = neuralContext.createLinearGradient(0, 0, width, 0);
  gradient.addColorStop(0, "rgba(59,130,246,.25)");
  gradient.addColorStop(0.5, "rgba(96,240,255,.95)");
  gradient.addColorStop(1, "rgba(252,211,77,.4)");
  neuralContext.strokeStyle = gradient;
  neuralContext.lineWidth = 2;
  neuralContext.shadowBlur = 12 + ui.neuralPulse * 8;
  neuralContext.shadowColor = "rgba(96,240,255,.35)";
  neuralContext.beginPath();
  snapshot.hidden.forEach((activation, index) => {
    const x = index / Math.max(1, snapshot.hidden.length - 1) * width;
    const y = height / 2 - activation * height * 0.35;
    if (index === 0) neuralContext.moveTo(x, y); else neuralContext.lineTo(x, y);
  });
  neuralContext.stroke();
  neuralContext.shadowBlur = 0;
}

function updateTelemetry() {
  const snapshot = runtime.snapshot();
  const stateElement = byId("neural-state");
  stateElement.textContent = snapshot.state;
  stateElement.className = `state ${snapshot.state.toLowerCase()}`;
  byId("neural-loss").textContent = snapshot.neural.loss.toFixed(3);
  byId("neural-steps").textContent = String(snapshot.neural.steps);
  byId("neural-activation").textContent = snapshot.neural.activation.toFixed(3);
  byId("m-rom-size").textContent = `${snapshot.romBytes} bytes`;
  byId("m-rom-checksum").textContent = snapshot.romChecksum;
  byId("m-cache").textContent = `${snapshot.cache.entries} · ${(snapshot.cache.hitRate * 100).toFixed(0)}% hit`;
  byId("m-render").textContent = `${ui.fps} · ${ui.quality.toUpperCase()}`;
  byId("m-isa").textContent = `v${snapshot.instructions.version} · ${snapshot.instructions.mutationCount} mods`;
  byId("m-isa-hash").textContent = snapshot.instructions.manifestHash;
  byId("m-refinements").textContent = String(snapshot.romRefinements);
  byId("m-aliases").textContent = String(snapshot.aliases);
  byId("m-traces").textContent = String(snapshot.traceEvents);

  const hotList = byId("m-hot-list");
  hotList.replaceChildren();
  const hot = snapshot.hotCities.length ? snapshot.hotCities : ["no reinforced cities"];
  for (const name of hot) {
    const token = document.createElement("span");
    token.textContent = name;
    hotList.appendChild(token);
  }

  const traceList = byId("trace-list");
  traceList.replaceChildren();
  for (const event of ui.traceEvents) {
    const item = document.createElement("li");
    const stage = document.createElement("b");
    stage.textContent = event.stage;
    const detail = document.createElement("span");
    detail.textContent = traceLabel(event);
    item.append(stage, detail);
    traceList.appendChild(item);
  }
}

async function executeQuery(text) {
  if (ui.busy || !text.trim()) return;
  ui.busy = true;
  byId("chat-send").disabled = true;
  byId("chat-input").disabled = true;
  addMessage(text, "user");
  emitEcho(text.slice(0, 48), window.innerWidth * 0.55, window.innerHeight * 0.65);
  try {
    const result = await runtime.query(text);
    byId("instruction-indicator").textContent = `${result.instruction} · ${result.latency.toFixed(2)}ms`;
    if (result.city) {
      flyToCity(result.city);
      highlightMarkers((city) => city.name === result.city.name);
    } else if (result.cities) {
      const selected = new Set(result.cities.map((city) => city.name));
      highlightMarkers((city) => selected.has(city.name));
    } else {
      highlightMarkers(() => false);
    }
    const metaTag = `${result.instruction} · ${result.cacheHit ? "memory hit" : "executed"} · loss ${result.neural.loss.toFixed(3)}`;
    addMessage(result.msg, "bot", metaTag);
    speak(result.msg);
    updateTelemetry();
    window.setTimeout(() => {
      runtime.settle();
      updateTelemetry();
    }, 1000);
  } catch (error) {
    addMessage(`Runtime error: ${error.message}`, "bot", "transaction rolled back");
  } finally {
    ui.busy = false;
    byId("chat-send").disabled = false;
    byId("chat-input").disabled = false;
    byId("chat-input").focus();
  }
}

byId("chat-form").addEventListener("submit", (event) => {
  event.preventDefault();
  const input = byId("chat-input");
  const text = input.value;
  input.value = "";
  executeQuery(text);
});
for (const button of document.querySelectorAll("[data-query]")) {
  button.addEventListener("click", () => executeQuery(button.dataset.query));
}

function toggleChat() {
  const container = byId("chat-container");
  const collapsed = container.classList.toggle("collapsed");
  byId("chat-header").setAttribute("aria-expanded", String(!collapsed));
  byId("chat-toggle").textContent = collapsed ? "+" : "−";
  byId("chat-toggle").setAttribute("aria-label", collapsed ? "Expand chat" : "Collapse chat");
}
byId("chat-header").addEventListener("click", toggleChat);
byId("chat-header").addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") { event.preventDefault(); toggleChat(); }
});

byId("meta-toggle").addEventListener("click", () => {
  const panel = byId("meta-panel");
  const open = panel.classList.toggle("open");
  byId("meta-toggle").setAttribute("aria-expanded", String(open));
});

byId("voice-toggle").addEventListener("click", () => {
  ui.voice = !ui.voice;
  byId("voice-toggle").setAttribute("aria-pressed", String(ui.voice));
  byId("voice-toggle").textContent = `Voice: ${ui.voice ? "on" : "off"}`;
  if (!ui.voice && "speechSynthesis" in window) window.speechSynthesis.cancel();
});

byId("refine-now").addEventListener("click", () => {
  const result = runtime.refine();
  const message = result.romCommitted || result.instructionCommitted
    ? `Optimization committed. ROM search cost ${result.before} → ${result.after}; ISA ${result.instructionCommitted ? "reordered" : "unchanged"}.`
    : "No modification was committed because no measured improvement was available.";
  addMessage(message, "bot", "evidence-gated self-optimization");
  updateTelemetry();
});

renderer.domElement.addEventListener("webglcontextlost", (event) => {
  event.preventDefault();
  addMessage("Rendering paused because the WebGL context was lost.", "bot", "awaiting recovery");
});
renderer.domElement.addEventListener("webglcontextrestored", () => {
  addMessage("WebGL context restored.", "bot", "renderer recovered");
});

let frameCount = 0;
let lastFpsUpdate = performance.now();
function updateFPS(now) {
  frameCount += 1;
  const elapsed = now - lastFpsUpdate;
  if (elapsed < 1000) return;
  const instantaneous = frameCount * 1000 / elapsed;
  ui.fps = Math.round(ui.fps * 0.72 + instantaneous * 0.28);
  frameCount = 0;
  lastFpsUpdate = now;
  if (ui.quality === "high" && ui.fps < 38) setQuality("medium");
  else if (ui.quality === "medium" && ui.fps < 28) setQuality("low");
  else if (ui.quality === "medium" && ui.fps > 55) setQuality("high");
  else if (ui.quality === "low" && ui.fps > 44) setQuality("medium");
  updateTelemetry();
}

function animate(time) {
  requestAnimationFrame(animate);
  updateFPS(time);
  updateSun();
  clouds.rotation.y += 0.00012;
  neuralPoints.rotation.y -= 0.0002;
  neuralConnections.rotation.y -= 0.0002;
  ui.neuralPulse *= 0.94;
  ui.networkPulse *= 0.93;
  pointMaterial.size = 0.021 + ui.networkPulse * 0.008;
  pointMaterial.opacity = Math.min(0.95, 0.5 + ui.networkPulse * 0.22);
  connectionMaterial.opacity = Math.min(0.5, 0.1 + ui.networkPulse * 0.16);
  drawNeuralActivity();

  if (flyTarget) {
    const transition = flyTarget;
    transition.progress = Math.min(1, transition.progress + 0.025);
    const progress = transition.progress;
    const eased = progress < 0.5 ? 4 * progress ** 3 : 1 - (-2 * progress + 2) ** 3 / 2;
    camera.position.lerpVectors(transition.start, transition.end, eased);
    controls.target.set(0, 0, 0);
    if (progress >= 1) flyTarget = null;
  }

  controls.update();
  if (!document.hidden) {
    renderer.render(scene, camera);
    labelRenderer.render(scene, camera);
  }
}

window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
  labelRenderer.setSize(window.innerWidth, window.innerHeight);
});

document.addEventListener("visibilitychange", () => {
  if (document.hidden && "speechSynthesis" in window) window.speechSynthesis.pause();
  else if ("speechSynthesis" in window) window.speechSynthesis.resume();
});

updateTelemetry();
animate(performance.now());
window.setTimeout(() => {
  addMessage("Try a city, a population comparison, status, or teach jozi = Johannesburg.", "bot", "instruction registry ready");
  emitEcho("runtime online", window.innerWidth * 0.48, 120);
}, 450);

window.__jarvisX = Object.freeze({
  query: (text) => runtime.query(text),
  settle: () => runtime.settle(),
  optimize: () => runtime.refine(),
  snapshot: () => runtime.snapshot(),
  traces: () => trace.snapshot(),
});

console.info("JARVIS X Trace-Driven Neural Echo Chamber Online", runtime.snapshot());
