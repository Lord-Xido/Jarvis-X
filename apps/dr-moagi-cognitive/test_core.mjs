import test from 'node:test';
import assert from 'node:assert/strict';

import {
  APP_SCHEMA_VERSION,
  DEFAULT_CONTROL_STATE,
  QUALITY_PROFILES,
  adaptiveQualityDecision,
  boundedPush,
  compileParameterPatch,
  convergenceState,
  fieldPointFromOriginal,
  formatQ16_48,
  makeSessionSnapshot,
  normalizeBackendBase,
  normalizedResidual,
  parseFieldCommand,
  parseSessionSnapshot,
  q16_48Decode,
  q16_48Encode,
  saturationRatio,
  updateResidualMemory,
} from './core.mjs';

test('Q16.48 encode/decode round-trips representative values', () => {
  for (const value of [0, 1, -1, 1.075, 24.2, 32767.5]) {
    const decoded = q16_48Decode(q16_48Encode(value));
    assert.ok(Math.abs(decoded - value) < 1e-10);
  }
  assert.equal(formatQ16_48(1), '0x0001 0000 0000 0000');
});

test('chat parser only changes supported bounded control parameters', () => {
  const report = parseFieldCommand('outward, set coupling to 1.8 beta 1.2 density 40', DEFAULT_CONTROL_STATE);
  assert.equal(report.candidate.inversionMode, 'OUTWARD');
  assert.equal(report.candidate.fieldCoupling, 1.8);
  assert.equal(report.candidate.betaShift, 1.2);
  assert.equal(report.candidate.density, 40);
  assert.equal(report.changed, true);
});

test('chat parser rejects out-of-bounds parameter commands', () => {
  assert.throws(() => parseFieldCommand('set coupling to 99', DEFAULT_CONTROL_STATE), /fieldCoupling must be in/);
});

test('bounded IDE patch compiler accepts only known assignments', () => {
  const compiled = compileParameterPatch(`
    mode = OUTWARD
    coupling = 1.7
    beta = 1.125
    density = 35
  `);
  assert.deepEqual(compiled.candidate, {
    inversionMode: 'OUTWARD',
    fieldCoupling: 1.7,
    betaShift: 1.125,
    density: 35,
    densityLimit: 100,
  });
  assert.throws(() => compileParameterPatch('eval = arbitrary_code'), /unsupported parameter/);
});

test('normalized fixed-point residual is zero for identical state', () => {
  const a = new Float64Array([1, 2, 3, 4]);
  assert.equal(normalizedResidual(a, a), 0);
  assert.ok(normalizedResidual(a, new Float64Array([1, 2, 3, 5])) > 0);
});

test('Omega residual memory is bounded exponentially weighted state', () => {
  const first = updateResidualMemory(0, 1, 0.9);
  const second = updateResidualMemory(first, 0, 0.9);
  assert.ok(first > 0 && first < 1);
  assert.ok(second < first);
});

test('field transform is deterministic and responds to inward/outward mode', () => {
  const inward = fieldPointFromOriginal(10, 2, 0, 1, DEFAULT_CONTROL_STATE);
  const outward = fieldPointFromOriginal(10, 2, 0, 1, {...DEFAULT_CONTROL_STATE, inversionMode: 'OUTWARD'});
  assert.deepEqual(inward, fieldPointFromOriginal(10, 2, 0, 1, DEFAULT_CONTROL_STATE));
  assert.ok(Math.hypot(outward[0], outward[2]) > Math.hypot(inward[0], inward[2]));
});

test('saturation is an explicit dimensionless ratio rather than a physical EM claim', () => {
  assert.equal(saturationRatio({...DEFAULT_CONTROL_STATE, density: 25, densityLimit: 100}), 0.25);
});

test('backend normalization rejects embedded credentials and non-http schemes', () => {
  assert.equal(normalizeBackendBase('https://example.com/'), 'https://example.com');
  assert.throws(() => normalizeBackendBase('ftp://example.com'), /http or https/);
  assert.throws(() => normalizeBackendBase('https://user:pass@example.com'), /must not contain credentials/);
});

test('bounded history retains only the newest records', () => {
  let items = [];
  for (let i = 0; i < 8; i += 1) items = boundedPush(items, i, 3);
  assert.deepEqual(items, [5, 6, 7]);
});

test('adaptive quality policy has bounded up/down transitions', () => {
  assert.equal(QUALITY_PROFILES.length, 3);
  assert.equal(adaptiveQualityDecision({fps: 20, currentIndex: 1}), 0);
  assert.equal(adaptiveQualityDecision({fps: 70, currentIndex: 1}), 2);
  assert.equal(adaptiveQualityDecision({fps: 50, currentIndex: 1}), 1);
  assert.equal(adaptiveQualityDecision({fps: 20, currentIndex: 0}), 0);
});

test('convergence requires both immediate and residual-memory tolerances', () => {
  assert.equal(convergenceState(1e-5, 1e-5, 1e-4).converged, true);
  assert.equal(convergenceState(1e-5, 1e-2, 1e-4).converged, false);
});

test('session snapshots are versioned, bounded and validated', () => {
  const snapshot = makeSessionSnapshot({
    controlState: {...DEFAULT_CONTROL_STATE, fieldCoupling: 1.8},
    backendBase: 'https://example.com/',
    qualityMode: 'HIGH',
    ttsEnabled: true,
    paused: true,
    commandHistory: Array.from({length: 150}, (_, index) => `command-${index}`),
  });
  assert.equal(snapshot.schemaVersion, APP_SCHEMA_VERSION);
  assert.equal(snapshot.backendBase, 'https://example.com');
  assert.equal(snapshot.commandHistory.length, 100);
  assert.equal(parseSessionSnapshot(JSON.stringify(snapshot)).controlState.fieldCoupling, 1.8);
  assert.throws(() => parseSessionSnapshot({...snapshot, schemaVersion: 999}), /unsupported session schema/);
});
