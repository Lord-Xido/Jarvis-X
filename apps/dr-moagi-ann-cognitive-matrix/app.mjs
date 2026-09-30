import {
  CognitiveMatrixEngine,
  OPCODES,
} from "./engine.mjs";

const engine = new CognitiveMatrixEngine({
  activeNodes: 1200,
  latentDim: 8,
  gamma: 0.72,
  rho: 0.86,
  learningRate: 0.045,
  fixedPointIterations: 8,
  ctrThreshold: 0.08,
});

const $ = (id) => document.getElementById(id);
const canvas = $("scene");
const ctx = canvas.getContext("2d");

const ui = {
  active: $("active"),
  pages: $("pages"),
  bytes: $("bytes"),
  latency: $("latency"),
  fixed: $("fixed"),
  residual: $("residual"),
  ctr: $("ctr"),
  omega: $("omega"),
  version: $("version"),
  decision: $("decision"),
  state: $("state"),
  log: $("log"),
};

const controls = {
  drive: $("drive"),
  gamma: $("gamma"),
  rho: $("rho"),
  eta: $("eta"),
  driveOut: $("driveOut"),
  gammaOut: $("gammaOut"),
  rhoOut: $("rhoOut"),
  etaOut: $("etaOut"),
};

const operators = [
  ["LOAD_BITSTREAM", OPCODES.LOAD_BITSTREAM, "#60a5fa", "Materialize the bounded input stream into sparse virtual memory.", "M_t ← load(X_t)"],
  ["AUTO_ENCODE", OPCODES.AUTO_ENCODE, "#60a5fa", "Project the current drive into the resident shared-weight latent field.", "Z_t = E_Θ(X_t, Ω_t)"],
  ["TENSOR_MAP", OPCODES.TENSOR_MAP, "#fbbf24", "Apply local shared-weight neighbour coupling across the active ring.", "Z′_i = tanh(W_s Z_i + W_n N_i)"],
  ["INWARD_FOLD", OPCODES.INWARD_FOLD, "#5eead4", "Damped fixed-point contraction of the candidate latent state.", "Z^(k+1) = (1−γ)Z^k + γ Φ_in(Z^k)"],
  ["DECODE_SPATIAL", OPCODES.DECODE_SPATIAL, "#5eead4", "Decode the contracted latent field into an observed scalar reconstruction.", "X̂_t = D_Θ(Z*)"],
  ["RESIDUAL", 0x70, "#fb7185", "Measure reconstruction discrepancy and preserve it as explicit evidence.", "E_t = X_t − X̂_t"],
  ["OMEGA_MEMORY", 0x71, "#fbbf24", "Integrate residual history into bounded temporal memory.", "Ω_(t+1) = ρΩ_t + (1−ρ)|E_t|"],
  ["CTR_VERIFY", 0x72, "#fb7185", "Gate candidate promotion on reconstruction, fixed-point and bound evidence.", "V_CTR = ||E||² + ||Φ(Z*)−Z*||² + C"],
  ["COMMIT_ROLLBACK", 0x73, "#5eead4", "Commit the verified candidate or restore the previously authoritative state.", "S_(t+1) = commit(S~) | rollback(S_t)"],
  ["EMIT_STREAM", OPCODES.EMIT_STREAM, "#60a5fa", "Expose bounded telemetry and reconstructed output.", "Y_t = emit(S_(t+1))"],
  ["RECUR", 0x74, "#5eead4", "Feed the accepted or restored state into the next macrocycle.", "S_(t+1) → S_(t+2)"],
];

const view = {
  yaw: 0.42,
  pitch: -0.18,
  zoom: 1,
  dragging: false,
  x: 0,
  y: 0,
  selected: 3,
};

const nodes = operators.map((op, i) => {
  const a = (i / operators.length) * Math.PI * 2;
  return {
    i,
    op,
    x: Math.cos(a) * 240,
    y: Math.sin(a * 2) * 70,
    z: Math.sin(a) * 240,
    sx: 0,
    sy: 0,
    depth: 0,
  };
});

function resize() {
  const dpr = Math.min(globalThis.devicePixelRatio || 1, 2);
  canvas.width = innerWidth * dpr;
  canvas.height = innerHeight * dpr;
  canvas.style.width = innerWidth + "px";
  canvas.style.height = innerHeight + "px";
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}
addEventListener("resize", resize);
resize();

function project(x, y, z) {
  const cy = Math.cos(view.yaw);
  const sy = Math.sin(view.yaw);
  const cp = Math.cos(view.pitch);
  const sp = Math.sin(view.pitch);
  const x1 = x * cy - z * sy;
  const z1 = x * sy + z * cy;
  const y2 = y * cp - z1 * sp;
  const z2 = y * sp + z1 * cp;
  const f = (620 * view.zoom) / (1020 + z2);
  return { x: innerWidth / 2 + x1 * f, y: innerHeight / 2 - y2 * f, z: z2 };
}

function rgba(hex, alpha) {
  const h = hex.slice(1);
  const n = Number.parseInt(h, 16);
  return \`rgba(\${(n >> 16) & 255},\${(n >> 8) & 255},\${n & 255},\${alpha})\`;
}

function draw() {
  ctx.clearRect(0, 0, innerWidth, innerHeight);
  const center = project(0, 0, 0);

  for (const n of nodes) {
    const p = project(n.x, n.y, n.z);
    n.sx = p.x;
    n.sy = p.y;
    n.depth = p.z;
  }

  ctx.lineWidth = 1.2;
  for (let i = 0; i < nodes.length; i += 1) {
    const a = nodes[i];
    const b = nodes[(i + 1) % nodes.length];
    ctx.strokeStyle = "rgba(96,165,250,.22)";
    ctx.beginPath();
    ctx.moveTo(a.sx, a.sy);
    ctx.lineTo(b.sx, b.sy);
    ctx.stroke();
  }

  const pulse = 24 + Math.sin(performance.now() * 0.002) * 3;
  const glow = ctx.createRadialGradient(center.x, center.y, 0, center.x, center.y, 82);
  glow.addColorStop(0, "rgba(94,234,212,.25)");
  glow.addColorStop(1, "rgba(94,234,212,0)");
  ctx.fillStyle = glow;
  ctx.beginPath();
  ctx.arc(center.x, center.y, 82, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = "rgba(94,234,212,.72)";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(center.x, center.y, pulse, 0, Math.PI * 2);
  ctx.stroke();

  for (const n of [...nodes].sort((a, b) => a.depth - b.depth)) {
    const selected = n.i === view.selected;
    const color = n.op[2];
    ctx.fillStyle = rgba(color, selected ? 1 : 0.82);
    ctx.beginPath();
    ctx.arc(n.sx, n.sy, selected ? 12 : 9, 0, Math.PI * 2);
    ctx.fill();
    if (selected) {
      ctx.strokeStyle = "rgba(255,255,255,.9)";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(n.sx, n.sy, 17, 0, Math.PI * 2);
      ctx.stroke();
    }
    ctx.fillStyle = "#e9f5ff";
    ctx.font = selected ? "700 11px ui-monospace" : "600 10px ui-monospace";
    ctx.textAlign = "center";
    ctx.fillText(n.op[0], n.sx, n.sy - 17);
  }

  view.yaw += 0.0012;
  requestAnimationFrame(draw);
}
requestAnimationFrame(draw);

function fmtBytes(value) {
  if (value < 1024) return value + " B";
  if (value < 1024 * 1024) return (value / 1024).toFixed(1) + " KiB";
  return (value / (1024 * 1024)).toFixed(2) + " MiB";
}

function log(text) {
  const d = document.createElement("div");
  d.textContent = text;
  ui.log.appendChild(d);
  while (ui.log.children.length > 9) ui.log.removeChild(ui.log.firstChild);
  ui.log.scrollTop = ui.log.scrollHeight;
}

function renderMetrics(m) {
  ui.active.textContent = m.activeNodes.toLocaleString();
  ui.pages.textContent = String(m.residentPages);
  ui.bytes.textContent = fmtBytes(m.residentBytes);
  ui.latency.textContent = m.macrocycleMs.toFixed(3) + " ms";
  ui.fixed.textContent = m.fixedPointResidual.toExponential(3);
  ui.residual.textContent = m.reconstructionResidual.toExponential(3);
  ui.ctr.textContent = m.ctrEnergy.toExponential(3);
  ui.omega.textContent = m.omega.toFixed(5);
  ui.version.textContent = String(m.version);
  ui.decision.textContent = m.accepted ? "COMMIT" : "ROLLBACK";
  ui.decision.style.color = m.accepted ? "#34d399" : "#fb7185";
  ui.state.textContent = m.accepted ? "REFERENCE COMMITTED" : "REFERENCE ROLLED BACK";
  ui.state.className = "pill " + (m.accepted ? "good" : "bad");
  log(
    \`v\${m.version} \${m.accepted ? "COMMIT" : "ROLLBACK"} · ΔZ=\${m.fixedPointResidual.toExponential(2)} · V_CTR=\${m.ctrEnergy.toExponential(2)} · \${m.macrocycleMs.toFixed(2)} ms\`,
  );
}

function step() {
  engine.setControls({
    gamma: Number(controls.gamma.value),
    rho: Number(controls.rho.value),
    learningRate: Number(controls.eta.value),
  });
  renderMetrics(engine.runMacrocycle(Number(controls.drive.value)));
}

for (const [input, out, digits] of [
  [controls.drive, controls.driveOut, 2],
  [controls.gamma, controls.gammaOut, 2],
  [controls.rho, controls.rhoOut, 2],
  [controls.eta, controls.etaOut, 3],
]) {
  input.addEventListener("input", () => {
    out.textContent = Number(input.value).toFixed(digits);
  });
}

$("step").addEventListener("click", step);
let auto = true;
$("auto").addEventListener("click", (event) => {
  auto = !auto;
  event.currentTarget.textContent = auto ? "Pause auto" : "Resume auto";
});
setInterval(() => {
  if (auto) step();
}, 850);

canvas.addEventListener("pointerdown", (event) => {
  view.dragging = true;
  view.x = event.clientX;
  view.y = event.clientY;
});
addEventListener("pointerup", () => {
  view.dragging = false;
});
addEventListener("pointermove", (event) => {
  if (!view.dragging) return;
  view.yaw += (event.clientX - view.x) * 0.006;
  view.pitch += (event.clientY - view.y) * 0.006;
  view.pitch = Math.max(-1.2, Math.min(1.2, view.pitch));
  view.x = event.clientX;
  view.y = event.clientY;
});
canvas.addEventListener("wheel", (event) => {
  event.preventDefault();
  view.zoom = Math.max(0.5, Math.min(2, view.zoom * (event.deltaY > 0 ? 0.92 : 1.08)));
}, { passive: false });
canvas.addEventListener("click", (event) => {
  let best = -1;
  let dist = 22;
  for (const n of nodes) {
    const d = Math.hypot(event.clientX - n.sx, event.clientY - n.sy);
    if (d < dist) {
      dist = d;
      best = n.i;
    }
  }
  if (best >= 0) {
    view.selected = best;
    const op = nodes[best].op;
    $("opTitle").textContent = \`\${op[0]} · 0x\${op[1].toString(16).toUpperCase().padStart(2, "0")}\`;
    $("opDesc").textContent = op[3];
    $("opEq").textContent = op[4];
  }
});

log("Bounded ANN Cognitive Matrix initialized.");
log("No external network, shell, filesystem or native SIMD authority.");
step();
