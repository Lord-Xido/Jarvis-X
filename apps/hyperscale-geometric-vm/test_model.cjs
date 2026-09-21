'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const {test} = require('node:test');
const html = fs.readFileSync(path.join(__dirname, 'index.html'), 'utf8');
const modelScript = html.match(/<script id="hyperscale-model">([\s\S]*?)<\/script>/)[1];
const Model = vm.runInNewContext(modelScript + '\nHyperscaleModel;');

test('both inline scripts parse', () => {
  const scripts = [...html.matchAll(/<script[^>]*>([\s\S]*?)<\/script>/g)];
  assert.equal(scripts.length, 2);
  scripts.forEach(script => new vm.Script(script[1]));
});

test('seeded shell contains exactly 64,000 bounded particles', () => {
  const model = new Model();
  const replay = new Model();
  assert.equal(model.positions.length, 192000);
  assert.deepEqual(model.positions, replay.positions);
  assert.equal(model.summary.length, 64);
  for (let i = 0; i < model.count; i++) {
    const radius = Math.hypot(...model.positions.subarray(i * 3, i * 3 + 3));
    assert.ok(radius >= 9 - 1e-6 && radius <= 15 + 1e-6);
  }
  assert.ok(model.summary.every(value => value >= 9 && value <= 15));
  assert.equal(model.advance(1), 0, 'idle rotation must not count as particle position updates');
});

test('contraction agrees across frame partitions and ends at the analytic target', () => {
  const a = new Model(256), b = new Model(256);
  a.start(); b.start();
  for (let i = 0; i < 15; i++) a.advance(1 / 30);
  for (let i = 0; i < 60; i++) b.advance(1 / 120);
  for (let i = 0; i < a.positions.length; i++) assert.ok(Math.abs(a.positions[i] - b.positions[i]) < 2e-6);
  a.advance(1); b.advance(1);
  assert.equal(a.state, 'contracted');
  assert.equal(b.state, 'contracted');
  assert.ok(Math.abs(a.angle - b.angle) < 1e-12, 'rotation stays continuous at burst completion');
  for (let i = 0; i < a.positions.length; i++) assert.ok(Math.abs(a.positions[i] - a.original[i] * 0.05) < 1e-6);
  assert.ok(a.measure() < 1e-6);
  assert.equal(a.start(), false, 'contracted state requires reset before replay');
  assert.equal(a.advance(1), 0);
});

test('summary and coordinate RMSE agree with independent calculations', () => {
  const model = new Model(65);
  model.start(); model.advance(0.6); model.measure();
  const radius = i => Math.hypot(...model.positions.subarray(3 * i, 3 * i + 3));
  assert.equal(model.summary[0], (radius(0) + radius(64)) / 2);
  assert.equal(model.summary[1], radius(1));
  const sum = [...model.positions].reduce((total, value, i) => total + (value - model.original[i] * 0.05) ** 2, 0);
  assert.equal(model.rmse, Math.sqrt(sum / 195));
});

test('mid-burst reset restores state and deterministically replays', () => {
  const model = new Model(128);
  const original = model.positions.slice();
  const initialSummary = model.summary.slice();
  const positionsReference = model.positions;
  model.start(); model.advance(0.4);
  const first = model.positions.slice();
  assert.equal(model.start(), false);
  assert.equal(model.updatedParticles, 128);
  model.reset();
  assert.strictEqual(model.positions, positionsReference, 'GPU-bound buffer identity must survive reset');
  assert.deepEqual(model.positions, original);
  assert.deepEqual(model.summary, initialSummary);
  assert.equal(model.angle, 0);
  assert.equal(model.elapsed, 0);
  assert.equal(model.updatedParticles, 0);
  model.start(); model.advance(0.4);
  assert.deepEqual(model.positions, first);
});

test('invalid counts, seeds and deltas fail before changing state', () => {
  for (const count of [0, -1, 64001, 1.5, NaN, Infinity]) assert.throws(() => new Model(count), {name: 'RangeError'});
  for (const seed of [-1, 2 ** 32, 1.5, NaN]) assert.throws(() => new Model(1, seed), {name: 'RangeError'});
  const model = new Model(1, 0);
  model.start();
  const before = model.positions.slice();
  for (const dt of [-1, NaN, Infinity]) assert.throws(() => model.advance(dt), {name: 'RangeError'});
  assert.equal(model.advance(0), 0);
  assert.deepEqual(model.positions, before);
  assert.equal(model.elapsed, 0);
  model.advance(2); model.measure();
  assert.ok(model.summary.every(Number.isFinite));
  assert.ok(model.rmse < 1e-6);
});
