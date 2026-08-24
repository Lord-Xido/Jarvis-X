export const APP_SCHEMA_VERSION = 2;
export const Q16_48_SCALE = 1n << 48n;
export const Q16_48_MIN = -32768;
export const Q16_48_MAX = 32768 - 2 ** -48;

export const DEFAULT_CONTROL_STATE = Object.freeze({
  inversionMode: 'INWARD',
  fieldCoupling: 1.45,
  betaShift: 1.075,
  density: 24.2,
  densityLimit: 100,
});

export const CONTROL_BOUNDS = Object.freeze({
  fieldCoupling: [0, 4],
  betaShift: [0.5, 2],
  density: [0, 100],
});

export const QUALITY_PROFILES = Object.freeze([
  Object.freeze({name: 'LOW', particles: 8000, jets: 400, pixelRatio: 1}),
  Object.freeze({name: 'MEDIUM', particles: 18000, jets: 900, pixelRatio: 1.5}),
  Object.freeze({name: 'HIGH', particles: 32000, jets: 1600, pixelRatio: 2}),
]);

const KEY_ALIASES = Object.freeze({
  mode: 'inversionMode',
  inversion: 'inversionMode',
  inversion_mode: 'inversionMode',
  coupling: 'fieldCoupling',
  field_coupling: 'fieldCoupling',
  em: 'fieldCoupling',
  em_coupling: 'fieldCoupling',
  beta: 'betaShift',
  beta_shift: 'betaShift',
  density: 'density',
});

function finiteNumber(value, name) {
  const number = Number(value);
  if (!Number.isFinite(number)) throw new TypeError(`${name} must be finite`);
  return number;
}

function clamp(value, lo, hi) {
  return Math.min(hi, Math.max(lo, value));
}

export function normalizeControlState(input = {}) {
  const source = {...DEFAULT_CONTROL_STATE, ...input};
  const mode = String(source.inversionMode).toUpperCase();
  if (mode !== 'INWARD' && mode !== 'OUTWARD') {
    throw new RangeError('inversionMode must be INWARD or OUTWARD');
  }
  const coupling = finiteNumber(source.fieldCoupling, 'fieldCoupling');
  const beta = finiteNumber(source.betaShift, 'betaShift');
  const density = finiteNumber(source.density, 'density');
  const densityLimit = finiteNumber(source.densityLimit, 'densityLimit');
  if (densityLimit <= 0) throw new RangeError('densityLimit must be > 0');
  const [cLo, cHi] = CONTROL_BOUNDS.fieldCoupling;
  const [bLo, bHi] = CONTROL_BOUNDS.betaShift;
  const [dLo, dHi] = CONTROL_BOUNDS.density;
  if (coupling < cLo || coupling > cHi) throw new RangeError(`fieldCoupling must be in [${cLo}, ${cHi}]`);
  if (beta < bLo || beta > bHi) throw new RangeError(`betaShift must be in [${bLo}, ${bHi}]`);
  if (density < dLo || density > dHi) throw new RangeError(`density must be in [${dLo}, ${dHi}]`);
  return {inversionMode: mode, fieldCoupling: coupling, betaShift: beta, density, densityLimit};
}

function captureNumber(command, patterns) {
  for (const pattern of patterns) {
    const match = command.match(pattern);
    if (match) return Number(match[1]);
  }
  return null;
}

export function parseFieldCommand(command, incumbent = DEFAULT_CONTROL_STATE) {
  const text = String(command ?? '').trim();
  const lower = text.toLowerCase();
  const candidate = {...normalizeControlState(incumbent)};
  const changes = [];

  if (/\b(outward|decode|expand)\b/i.test(text)) {
    candidate.inversionMode = 'OUTWARD';
    changes.push('inversionMode=OUTWARD');
  } else if (/\b(inward|encode|implode|contract)\b/i.test(text)) {
    candidate.inversionMode = 'INWARD';
    changes.push('inversionMode=INWARD');
  }

  const coupling = captureNumber(text, [
    /(?:field\s+)?coupling(?:\s*(?:=|to))?\s*(-?\d+(?:\.\d+)?)/i,
    /\bem(?:\s*(?:=|to|boost))?\s*(-?\d+(?:\.\d+)?)/i,
  ]);
  if (coupling !== null) {
    candidate.fieldCoupling = coupling;
    changes.push(`fieldCoupling=${coupling}`);
  }

  const beta = captureNumber(text, [
    /\bbeta(?:\s*(?:=|to))?\s*(-?\d+(?:\.\d+)?)/i,
    /\bbeta[_\s-]?shift(?:\s*(?:=|to))?\s*(-?\d+(?:\.\d+)?)/i,
  ]);
  if (beta !== null) {
    candidate.betaShift = beta;
    changes.push(`betaShift=${beta}`);
  }

  const density = captureNumber(text, [/\bdensity(?:\s*(?:=|to))?\s*(-?\d+(?:\.\d+)?)/i]);
  if (density !== null) {
    candidate.density = density;
    changes.push(`density=${density}`);
  }

  return {command: text, lower, changes, candidate: normalizeControlState(candidate), changed: changes.length > 0};
}

export function compileParameterPatch(source, incumbent = DEFAULT_CONTROL_STATE) {
  const candidate = {...normalizeControlState(incumbent)};
  const assignments = [];
  const lines = String(source ?? '').split(/\r?\n/);

  for (let index = 0; index < lines.length; index += 1) {
    const stripped = lines[index].replace(/#.*/, '').replace(/\/\/.*/, '').trim();
    if (!stripped) continue;
    const match = stripped.match(/^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+?)\s*;?$/);
    if (!match) throw new SyntaxError(`line ${index + 1}: expected key = value`);
    const alias = match[1].toLowerCase();
    const key = KEY_ALIASES[alias];
    if (!key) throw new SyntaxError(`line ${index + 1}: unsupported parameter ${match[1]}`);
    const raw = match[2].replace(/^['"]|['"]$/g, '').trim();
    if (key === 'inversionMode') candidate.inversionMode = raw.toUpperCase();
    else candidate[key] = finiteNumber(raw, key);
    assignments.push(`${key}=${candidate[key]}`);
  }

  if (!assignments.length) throw new SyntaxError('parameter patch contains no assignments');
  return {candidate: normalizeControlState(candidate), assignments};
}

export function q16_48Encode(value) {
  const number = finiteNumber(value, 'Q16.48 value');
  if (number < Q16_48_MIN || number > Q16_48_MAX) throw new RangeError('Q16.48 value out of range');
  return BigInt(Math.round(number * 2 ** 48));
}

export function q16_48Decode(raw) {
  const signed = typeof raw === 'bigint' ? raw : BigInt(raw);
  return Number(signed) / 2 ** 48;
}

export function formatQ16_48(value) {
  const raw = q16_48Encode(value);
  const unsigned = raw < 0 ? (1n << 64n) + raw : raw;
  const hex = unsigned.toString(16).toUpperCase().padStart(16, '0');
  return `0x${hex.slice(0, 4)} ${hex.slice(4, 8)} ${hex.slice(8, 12)} ${hex.slice(12, 16)}`;
}

export function normalizedResidual(previous, current, epsilon = 1e-12) {
  if (!previous || !current || previous.length !== current.length || previous.length === 0) return 0;
  let delta2 = 0;
  let base2 = 0;
  for (let i = 0; i < current.length; i += 1) {
    const a = Number(previous[i]);
    const b = Number(current[i]);
    const d = b - a;
    delta2 += d * d;
    base2 += a * a;
  }
  return Math.sqrt(delta2) / (Math.sqrt(base2) + epsilon);
}

export function updateResidualMemory(previousMemory, residual, rho = 0.9) {
  const r = finiteNumber(residual, 'residual');
  const p = finiteNumber(previousMemory, 'previousMemory');
  const decay = clamp(finiteNumber(rho, 'rho'), 0, 1);
  return decay * p + (1 - decay) * r;
}

export function saturationRatio(state) {
  const normalized = normalizeControlState(state);
  return clamp(normalized.density / normalized.densityLimit, 0, 1);
}

export function fieldPointFromOriginal(x, y, z, timeSeconds, state) {
  const s = normalizeControlState(state);
  const ratio = saturationRatio(s);
  const inwardScale = s.inversionMode === 'INWARD' ? Math.max(0.1, 1 - ratio * 0.75) : 1 + ratio * 0.5;
  const radialScale = Math.pow(inwardScale, Math.max(0.1, s.betaShift));
  const baseAngle = Math.atan2(z, x);
  const angle = baseAngle + timeSeconds * s.fieldCoupling * 0.5;
  const radius = Math.hypot(x, z) * radialScale;
  const vertical = y * radialScale + Math.sin(timeSeconds * 3 + radius * 0.2) * (1.5 * s.fieldCoupling);
  return [Math.cos(angle) * radius, vertical, Math.sin(angle) * radius];
}

export function describeControlState(state, residual = 0, omegaMemory = 0, gate = 'ACCEPT') {
  const s = normalizeControlState(state);
  return {
    mode: s.inversionMode,
    fieldCoupling: s.fieldCoupling,
    betaShift: s.betaShift,
    density: s.density,
    saturation: saturationRatio(s),
    residual: Math.max(0, Number(residual) || 0),
    omegaMemory: Math.max(0, Number(omegaMemory) || 0),
    gate: String(gate),
  };
}

export function normalizeBackendBase(value) {
  const text = String(value ?? '').trim();
  if (!text) return '';
  let url;
  try { url = new URL(text); } catch { throw new TypeError('backend URL must be an absolute http(s) URL'); }
  if (url.protocol !== 'http:' && url.protocol !== 'https:') throw new TypeError('backend URL must use http or https');
  if (url.username || url.password) throw new TypeError('backend URL must not contain credentials');
  url.hash = '';
  url.search = '';
  return url.toString().replace(/\/$/, '');
}

export function boundedPush(items, value, limit = 200) {
  const max = Math.max(1, Math.floor(finiteNumber(limit, 'limit')));
  const next = Array.isArray(items) ? items.slice() : [];
  next.push(value);
  if (next.length > max) next.splice(0, next.length - max);
  return next;
}

export function adaptiveQualityDecision({fps, currentIndex = 1, lowThreshold = 42, highThreshold = 58}) {
  const measured = finiteNumber(fps, 'fps');
  const index = clamp(Math.floor(finiteNumber(currentIndex, 'currentIndex')), 0, QUALITY_PROFILES.length - 1);
  if (measured < lowThreshold && index > 0) return index - 1;
  if (measured > highThreshold && index < QUALITY_PROFILES.length - 1) return index + 1;
  return index;
}

export function convergenceState(residual, omegaMemory, epsilon = 1e-4) {
  const r = Math.max(0, finiteNumber(residual, 'residual'));
  const o = Math.max(0, finiteNumber(omegaMemory, 'omegaMemory'));
  const e = Math.max(0, finiteNumber(epsilon, 'epsilon'));
  return {converged: r <= e && o <= e, residual: r, omegaMemory: o, epsilon: e};
}

export function makeSessionSnapshot(input = {}) {
  const controlState = normalizeControlState(input.controlState ?? DEFAULT_CONTROL_STATE);
  const qualityMode = String(input.qualityMode ?? 'AUTO').toUpperCase();
  if (!['AUTO', 'LOW', 'MEDIUM', 'HIGH'].includes(qualityMode)) throw new RangeError('unsupported qualityMode');
  const commandHistory = (Array.isArray(input.commandHistory) ? input.commandHistory : [])
    .map(value => String(value).slice(0, 500))
    .slice(-100);
  return {
    schemaVersion: APP_SCHEMA_VERSION,
    createdAt: new Date(Number(input.createdAt) || Date.now()).toISOString(),
    controlState,
    backendBase: normalizeBackendBase(input.backendBase ?? ''),
    qualityMode,
    ttsEnabled: Boolean(input.ttsEnabled),
    paused: Boolean(input.paused),
    commandHistory,
  };
}

export function parseSessionSnapshot(value) {
  const source = typeof value === 'string' ? JSON.parse(value) : value;
  if (!source || typeof source !== 'object') throw new TypeError('session snapshot must be an object');
  if (Number(source.schemaVersion) !== APP_SCHEMA_VERSION) throw new RangeError(`unsupported session schema ${source.schemaVersion}`);
  return makeSessionSnapshot(source);
}
