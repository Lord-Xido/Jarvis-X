import test from 'node:test';
import assert from 'node:assert/strict';
import {
  DEFAULT_PARAMS,
  FixedStepEngine,
  createInitialState,
  mulberry32,
  stepSystem,
} from './core.mjs';

test('density controls particle count exactly as artifact formula', () => {
  assert.equal(createInitialState({ rho: 1 }).particles.length, 310);
  assert.equal(createInitialState({ rho: 0.1 }).particles.length, 112);
  assert.equal(createInitialState({ rho: 2 }).particles.length, 530);
});

test('deterministic seeded simulation is reproducible', () => {
  const a = createInitialState({ seed: 42 });
  const b = createInitialState({ seed: 42 });
  const rngA = mulberry32(99);
  const rngB = mulberry32(99);
  let sa = a;
  let sb = b;
  for (let i = 0; i < 120; i += 1) {
    sa = stepSystem(sa, 1 / 120, { rng: rngA, regime: 'laminar' });
    sb = stepSystem(sb, 1 / 120, { rng: rngB, regime: 'laminar' });
  }
  assert.deepEqual(sa, sb);
});

test('state remains finite and bounded over sustained operation', () => {
  let state = createInitialState({ seed: 7 });
  const rng = mulberry32(123);
  for (let i = 0; i < 1200; i += 1) {
    state = stepSystem(state, 1 / 120, { rng, regime: 'turbulent', params: DEFAULT_PARAMS });
  }
  for (const p of state.particles) {
    assert.ok(Number.isFinite(p.r) && Number.isFinite(p.theta) && Number.isFinite(p.z));
    assert.ok(Number.isFinite(p.vr) && Number.isFinite(p.vtheta) && Number.isFinite(p.vz));
    assert.ok(p.r >= 0.05 && p.r <= 0.98);
    assert.ok(p.z >= -1 && p.z <= 1);
    assert.ok(p.shell >= 0 && p.shell <= 2);
  }
});

test('fixed-step engine decouples physics time from display refresh rate', () => {
  const run = (frameDt, frames) => {
    let state = createInitialState({ seed: 11 });
    const rng = mulberry32(111);
    const engine = new FixedStepEngine({ step: 1 / 120 });
    for (let i = 0; i < frames; i += 1) {
      ({ state } = engine.advance(state, frameDt, { rng, permeationEnabled: false }));
    }
    return state;
  };
  const at60 = run(1 / 60, 60);
  const at120 = run(1 / 120, 120);
  assert.ok(Math.abs(at60.time - 1) < 1e-10);
  assert.ok(Math.abs(at120.time - 1) < 1e-10);
  assert.deepEqual(at60, at120);
});

test('permeation gate can be disabled', () => {
  let state = createInitialState({ seed: 19 });
  const rng = () => 0;
  for (let i = 0; i < 200; i += 1) {
    state = stepSystem(state, 1 / 120, {
      rng,
      permeationEnabled: false,
      params: { ...DEFAULT_PARAMS, mu: 1 },
    });
  }
  assert.equal(state.permeations, 0);
});
