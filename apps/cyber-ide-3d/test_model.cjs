'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const {test} = require('node:test');

const html = fs.readFileSync(path.join(__dirname, 'index.html'), 'utf8');
const modelMatch = html.match(/<script id="cyber-ide-model">([\s\S]*?)<\/script>/);
assert.ok(modelMatch, 'model script must exist');
const Model = vm.runInNewContext(modelMatch[1] + '\nCyberIDE3DModel;');

test('both inline scripts parse', () => {
  const scripts = [...html.matchAll(/<script[^>]*>([\s\S]*?)<\/script>/g)];
  assert.equal(scripts.length, 2);
  scripts.forEach(script => new vm.Script(script[1]));
});

test('one million LOC are partitioned exactly across 2,500 towers', () => {
  const model = new Model();
  assert.equal(model.towerCount, 2500);
  assert.equal(model.gridSize * model.gridSize, 2500);
  assert.equal(model.meanLOC, 400);
  assert.equal([...model.lineCounts].reduce((a,b) => a+b, 0), 1000000);
  assert.ok([...model.lineCounts].every(value => value > 0));
  assert.equal(model.lineStarts[0], 0);
  const last = model.towerCount - 1;
  assert.equal(model.lineStarts[last] + model.lineCounts[last], 1000000);
});

test('design arithmetic remains exact and explicitly virtual', () => {
  const model = new Model();
  assert.equal(model.targetThroughputLOCs, 1e12);
  assert.equal(model.targetPassSeconds, 1e-6);
  assert.equal(model.astBytesPerLOC, 32);
  assert.equal(model.logicalBandwidthBytesPerSecond, 32e12);
  assert.equal(model.logicalCharacters, 40e6);
  assert.equal(model.glyphPixelsPerCharacter, 128);
  assert.equal(model.logicalGlyphPixels, 5.12e9);
});

test('grid and torus embeddings satisfy their geometric contracts', () => {
  const model = new Model();
  const spacing = 1.45;
  assert.equal(model.gridPositions[0], -24.5 * spacing);
  assert.equal(model.gridPositions[1], 0);
  assert.equal(model.gridPositions[2], -24.5 * spacing);
  for (let i = 0; i < model.towerCount; i += 137) {
    const j = i * 3;
    const x = model.torusPositions[j];
    const y = model.torusPositions[j + 1];
    const z = model.torusPositions[j + 2];
    const radial = Math.hypot(x, z);
    const tubeRadius = Math.hypot(radial - model.majorRadius, y);
    assert.ok(Math.abs(tubeRadius - model.minorRadius) < 1e-9);
    const n = Math.hypot(model.torusNormals[j], model.torusNormals[j+1], model.torusNormals[j+2]);
    assert.ok(Math.abs(n - 1) < 1e-12);
  }
});

test('morph follows the exact exponential relaxation law', () => {
  const model = new Model();
  model.setTopology(1);
  model.advance(0.25, 1);
  const expected = 1 - Math.exp(-3.2 * 0.25);
  assert.ok(Math.abs(model.morph - expected) < 1e-12);
  model.advance(2, 1);
  assert.ok(model.morph > 0.999);
  model.setTopology(0);
  model.advance(2, 1);
  assert.ok(model.morph < 0.002);
});

test('runtime heat is bounded, deterministic, pausable and resettable', () => {
  const a = new Model(7), b = new Model(7);
  a.advance(0.5, 1); b.advance(0.5, 1);
  assert.deepEqual(a.heats, b.heats);
  assert.equal(a.processedLines, b.processedLines);
  assert.ok([...a.heats].every(v => v >= 0 && v <= 1));
  const before = a.processedLines;
  a.setRunning(false);
  a.advance(1, 1);
  assert.equal(a.processedLines, before);
  a.setOptimizer(true);
  a.advance(1, 1);
  assert.ok(a.syntheticObjective < 1 && a.syntheticObjective >= 0.08);
  a.reset();
  assert.equal(a.morph, 0);
  assert.equal(a.processedLines, 0);
  assert.equal(a.syntheticObjective, 1);
  assert.ok([...a.heats].every(v => v === 0));
});

test('invalid external state updates fail closed', () => {
  const model = new Model();
  assert.throws(() => model.setTopology(0.5), {name:'RangeError'});
  assert.throws(() => model.advance(-1), {name:'RangeError'});
  assert.throws(() => model.advance(NaN), {name:'RangeError'});
  assert.throws(() => model.advance(1, 0), {name:'RangeError'});
  assert.throws(() => model.towerState(-1), {name:'RangeError'});
  assert.throws(() => new Model(-1), {name:'RangeError'});
});
