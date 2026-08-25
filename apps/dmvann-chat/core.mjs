export const RUNTIME_VERSION = '0.1.0';

export const DEFAULT_FIELD = Object.freeze({
  psi: 0,
  phi: 0,
  lambda: 1,
  omega: 0,
  theta: 0,
  coupling: 1.8,
  memoryDecay: 0.82,
  learningRate: 0.12,
  tick: 0,
});

export function clamp(value, min, max) {
  return Math.min(max, Math.max(min, Number(value)));
}

export function normalizeText(value, maxLength = 16_384) {
  if (typeof value !== 'string') return '';
  return value.replace(/\u0000/g, '').slice(0, maxLength);
}

export function normalizeMessages(messages, { maxMessages = 64, maxLength = 16_384 } = {}) {
  if (!Array.isArray(messages)) throw new TypeError('messages must be an array');
  return messages
    .slice(-maxMessages)
    .map((message) => {
      const role = ['system', 'user', 'assistant'].includes(message?.role) ? message.role : 'user';
      return { role, content: normalizeText(message?.content, maxLength) };
    })
    .filter((message) => message.content.length > 0);
}

export function fingerprint(text) {
  const bytes = new TextEncoder().encode(normalizeText(text, 65_536));
  let hash = 0x811c9dc5;
  for (const byte of bytes) {
    hash ^= byte;
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  return hash.toString(16).padStart(8, '0');
}

export function stepField(previous = DEFAULT_FIELD, input = '') {
  const text = normalizeText(input, 16_384);
  const byteLength = new TextEncoder().encode(text).length;
  const psi = clamp(Math.log2(1 + byteLength) / 14, 0, 1);
  const coupling = clamp(previous.coupling ?? DEFAULT_FIELD.coupling, 0, 4);
  const memoryDecay = clamp(previous.memoryDecay ?? DEFAULT_FIELD.memoryDecay, 0, 0.999);
  const learningRate = clamp(previous.learningRate ?? DEFAULT_FIELD.learningRate, 0, 1);
  const thetaPrev = clamp(previous.theta ?? 0, -1, 1);
  const omegaPrev = clamp(previous.omega ?? 0, 0, 1);
  const phi = Math.tanh(coupling * psi + thetaPrev * 0.25);
  const residual = Math.abs(phi - psi);
  const omega = clamp(memoryDecay * omegaPrev + (1 - memoryDecay) * residual, 0, 1);
  const lambda = 1 / (1 + omega);
  const theta = clamp(thetaPrev + learningRate * (phi - psi), -1, 1);

  return {
    psi,
    phi,
    lambda,
    omega,
    theta,
    coupling,
    memoryDecay,
    learningRate,
    residual,
    tick: (previous.tick ?? 0) + 1,
    fingerprint: fingerprint(text),
  };
}

export function createChatPayload(messages, { model = 'dmvann-default', temperature = 0.2 } = {}) {
  const normalized = normalizeMessages(messages);
  if (!normalized.some((message) => message.role === 'user')) {
    throw new Error('at least one user message is required');
  }
  return {
    model: normalizeText(model, 256) || 'dmvann-default',
    temperature: clamp(temperature, 0, 2),
    messages: normalized,
    stream: false,
  };
}

export function localControlPlaneResponse(userText, field) {
  const text = normalizeText(userText, 16_384);
  const words = text.trim() ? text.trim().split(/\s+/).length : 0;
  return [
    'Local deterministic DMVANN control-plane fallback is active; no remote model response was used.',
    `Input: ${words} words, fingerprint ${field.fingerprint}.`,
    `Field state: Ψ=${field.psi.toFixed(4)}, Φ=${field.phi.toFixed(4)}, Λ=${field.lambda.toFixed(4)}, Ω=${field.omega.toFixed(4)}, Θ=${field.theta.toFixed(4)}.`,
    'Configure the server-side upstream runtime to enable generative chat.',
  ].join('\n');
}
