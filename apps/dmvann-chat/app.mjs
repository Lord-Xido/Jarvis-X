import { DEFAULT_FIELD, localControlPlaneResponse, stepField } from './core.mjs';

const $ = (selector) => document.querySelector(selector);
const chat = $('#chat');
const form = $('#composer');
const input = $('#prompt');
const sendButton = $('#send');
const runtimeBadge = $('#runtime-badge');
const metrics = {
  psi: $('#psi'), phi: $('#phi'), lambda: $('#lambda'), omega: $('#omega'), theta: $('#theta'),
};
const canvas = $('#field-canvas');
const ctx = canvas.getContext('2d');

let field = { ...DEFAULT_FIELD };
let messages = [];
let remoteAvailable = false;
let busy = false;
let phase = 0;

function addMessage(role, content) {
  const article = document.createElement('article');
  article.className = `message ${role}`;
  const header = document.createElement('div');
  header.className = 'message-role';
  header.textContent = role === 'user' ? 'YOU' : 'DMVANN';
  const body = document.createElement('pre');
  body.textContent = content;
  article.append(header, body);
  chat.append(article);
  chat.scrollTop = chat.scrollHeight;
}

function renderMetrics() {
  for (const [key, node] of Object.entries(metrics)) node.textContent = Number(field[key]).toFixed(4);
}

function resizeCanvas() {
  const rect = canvas.getBoundingClientRect();
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.max(1, Math.floor(rect.width * ratio));
  canvas.height = Math.max(1, Math.floor(rect.height * ratio));
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
}

function drawField() {
  const { width, height } = canvas.getBoundingClientRect();
  ctx.clearRect(0, 0, width, height);
  const cx = width / 2;
  const cy = height / 2;
  const scale = Math.min(width, height) * 0.28;
  const count = 180;
  for (let i = 0; i < count; i++) {
    const t = i / count;
    const a = t * Math.PI * 12 + phase;
    const z = Math.sin(a * 0.47 + field.theta * 4);
    const radius = (0.2 + 0.8 * t) * (0.68 + field.lambda * 0.32);
    const x3 = Math.cos(a) * radius;
    const y3 = Math.sin(a) * radius;
    const perspective = 1 / (1.7 - z * 0.35);
    const x = cx + x3 * scale * perspective;
    const y = cy + (y3 * 0.55 + z * 0.34) * scale * perspective;
    const size = 1.2 + 2.8 * perspective * (0.35 + field.omega);
    ctx.beginPath();
    ctx.arc(x, y, size, 0, Math.PI * 2);
    ctx.fillStyle = `hsla(${188 + field.phi * 85}, 92%, ${55 + z * 12}%, ${0.22 + perspective * 0.48})`;
    ctx.fill();
  }
  phase += 0.006 + field.psi * 0.012;
  requestAnimationFrame(drawField);
}

async function detectRuntime() {
  try {
    const response = await fetch('./healthz', { cache: 'no-store' });
    if (!response.ok) throw new Error('health unavailable');
    const health = await response.json();
    remoteAvailable = Boolean(health.upstreamConfigured);
    runtimeBadge.textContent = remoteAvailable ? 'REMOTE MODEL READY' : 'LOCAL FIELD MODE';
  } catch {
    remoteAvailable = false;
    runtimeBadge.textContent = 'STATIC / LOCAL FIELD MODE';
  }
}

async function requestAssistant() {
  if (!remoteAvailable) return localControlPlaneResponse(messages.at(-1)?.content || '', field);
  const response = await fetch('./api/chat', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ messages }),
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok || !payload?.message?.content) {
    throw new Error(payload?.message || payload?.error || `chat request failed (${response.status})`);
  }
  return payload.message.content;
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const text = input.value.trim();
  if (!text || busy) return;
  busy = true;
  sendButton.disabled = true;
  input.value = '';
  messages.push({ role: 'user', content: text });
  messages = messages.slice(-48);
  field = stepField(field, text);
  renderMetrics();
  addMessage('user', text);
  try {
    const content = await requestAssistant();
    messages.push({ role: 'assistant', content });
    field = stepField(field, content);
    renderMetrics();
    addMessage('assistant', content);
  } catch (error) {
    const content = `Runtime error: ${error.message}\nFalling back to deterministic local control-plane output.\n\n${localControlPlaneResponse(text, field)}`;
    messages.push({ role: 'assistant', content });
    addMessage('assistant', content);
    remoteAvailable = false;
    runtimeBadge.textContent = 'REMOTE DEGRADED / LOCAL MODE';
  } finally {
    busy = false;
    sendButton.disabled = false;
    input.focus();
  }
});

window.addEventListener('resize', resizeCanvas);
resizeCanvas();
renderMetrics();
detectRuntime();
drawField();
addMessage('assistant', 'DMVANN Chat initialized. The Ψ–Φ–Λ–Ω–Θ field is deterministic locally; generative chat activates only when a server-side upstream is configured.');
