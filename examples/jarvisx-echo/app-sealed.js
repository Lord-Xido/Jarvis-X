import { encodeROM, JarvisXRuntime, TraceBus } from "./runtime-core.mjs";
import {
  DR_MOAGI_INSTRUCTION_MANIFEST,
  DR_MOAGI_RELEASE_SEAL,
  DR_MOAGI_TRUST_ANCHOR,
} from "./dr-moagi-seal.mjs";
import { verifyDrMoagiSeal } from "./provenance.mjs";

const RELEASE_CITIES = [
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

function renderSealStatus(result) {
  const status = byId("m-seal-status");
  const key = byId("m-seal-key");
  if (status) {
    status.textContent = result.authentic ? "AUTHENTIC" : "RECOVERY";
    status.className = result.authentic ? "seal-authentic" : "seal-recovery";
  }
  if (key) key.textContent = result.keyId;
}

function enterRecoveryMode(result) {
  document.documentElement.dataset.runtimeMode = "recovery";
  renderSealStatus(result);
  const input = byId("chat-input");
  const send = byId("chat-send");
  const refine = byId("refine-now");
  if (input) {
    input.disabled = true;
    input.placeholder = "Recovery mode: release verification failed";
  }
  if (send) send.disabled = true;
  if (refine) refine.disabled = true;

  const container = byId("chat-messages");
  if (container) {
    const message = document.createElement("div");
    message.className = "msg bot recovery-message";
    const strong = document.createElement("strong");
    strong.textContent = "ROM provenance verification failed.";
    const detail = document.createElement("span");
    detail.textContent = `Recovery mode engaged: ${result.reason || "unknown verification failure"}.`;
    message.append(strong, detail);
    container.appendChild(message);
  }
  console.error("JARVIS X recovery mode", result);
}

const verificationRuntime = new JarvisXRuntime(RELEASE_CITIES, { trace: new TraceBus(16) });
const runtimeInstructionSnapshot = verificationRuntime.snapshot().instructions;
const instructionManifest = Object.freeze({
  schema: DR_MOAGI_INSTRUCTION_MANIFEST.schema,
  version: runtimeInstructionSnapshot.version,
  order: Object.freeze(runtimeInstructionSnapshot.order.slice()),
});

const sealResult = await verifyDrMoagiSeal({
  seal: DR_MOAGI_RELEASE_SEAL,
  trustAnchor: DR_MOAGI_TRUST_ANCHOR,
  romBuffer: encodeROM(RELEASE_CITIES),
  instructionManifest,
});

window.__drMoagiSeal = Object.freeze({
  ...sealResult,
  trustScope: DR_MOAGI_TRUST_ANCHOR.scope,
  release: DR_MOAGI_RELEASE_SEAL.payload.release,
});

renderSealStatus(sealResult);
if (sealResult.authentic) {
  document.documentElement.dataset.runtimeMode = "authentic";
  await import("./app.js");
  console.info("Dr Moagi development provenance seal verified", window.__drMoagiSeal);
} else {
  enterRecoveryMode(sealResult);
}
