import assert from 'node:assert/strict';
import test from 'node:test';

import {
  MODALITY,
  MultimodalROM3D,
  morton3D,
  unmorton3D,
} from './core.mjs';

test('million-cubed domain is represented as 10^18 logical cells', () => {
  const runtime = new MultimodalROM3D({ activeCells: 128, maxActiveCells: 256, latentDim: 32 });
  assert.equal(runtime.logicalExtent, 1_000_000);
  assert.equal(runtime.logicalCellCount, 1_000_000_000_000_000_000n);
  assert.ok(runtime.residentBytes() < 100_000);
});

test('60-bit Morton address round-trips legal million-cubed coordinates', () => {
  const samples = [
    [0, 0, 0],
    [1, 2, 3],
    [999_999, 500_000, 42],
    [2 ** 20 - 1, 2 ** 20 - 2, 2 ** 20 - 3],
  ];
  for (const [x, y, z] of samples) {
    assert.deepEqual(unmorton3D(morton3D(x, y, z)), { x, y, z });
  }
});

test('multimodal ingestion unifies modality-tagged numeric streams', () => {
  const runtime = new MultimodalROM3D({ activeCells: 12, maxActiveCells: 16, latentDim: 4 });
  const result = runtime.ingest([
    { modality: 'text', values: [0.1, 0.2] },
    { modality: 'audio', values: [0.3, 0.4] },
    { modality: 'image', values: [0.5, 0.6] },
    { modality: 'video', values: [0.7, 0.8] },
    { modality: 'tensor', values: [0.9, 1.0] },
    { modality: 'sensor', values: [0.25, 0.75] },
  ]);
  const all = MODALITY.text | MODALITY.audio | MODALITY.image | MODALITY.video | MODALITY.tensor | MODALITY.sensor;
  assert.equal(result.modalityMask, all);
  assert.equal(result.samples, 12);
});

test('end-to-end codec reaches bounded fixed point and commits transactionally', () => {
  const runtime = new MultimodalROM3D({
    activeCells: 256,
    maxActiveCells: 512,
    latentDim: 32,
    reconstructionTolerance: 0.5,
  });
  const receipt = runtime.tick({ timeSeconds: 1.25, drive: 0.65 });
  assert.equal(receipt.committed, true);
  assert.equal(receipt.version, 1);
  assert.ok(Number.isFinite(receipt.reconstructionDistance));
  assert.ok(receipt.reconstructionDistance <= 0.5);
  assert.ok(receipt.fixedPointResidual <= Math.max(runtime.fixedPointTolerance, 2 / 65536));
});

test('ROM round-trip preserves transactional state and rejects corruption', () => {
  const source = new MultimodalROM3D({ activeCells: 128, maxActiveCells: 256, latentDim: 16 });
  const receipt = source.tick({ timeSeconds: 0.75, drive: 0.4 });
  assert.equal(receipt.committed, true);

  const bytes = source.encodeROM();
  const restored = new MultimodalROM3D({ activeCells: 64, maxActiveCells: 256, latentDim: 16 });
  const result = restored.decodeROM(bytes);

  assert.equal(result.activeCells, source.activeCells);
  assert.equal(result.version, source.version);
  assert.equal(restored.x[17], source.x[17]);
  assert.equal(restored.y[17], source.y[17]);
  assert.equal(restored.z[17], source.z[17]);
  assert.equal(restored.modality[17], source.modality[17]);
  assert.ok(Math.abs(restored.committed[17] - source.committed[17]) <= 1 / 65536);

  const corrupted = bytes.slice();
  corrupted[corrupted.length - 1] ^= 0xff;
  assert.throws(() => restored.decodeROM(corrupted), /checksum mismatch/);
});

test('inward turn reduces resident active support without going below latent dimension', () => {
  const runtime = new MultimodalROM3D({ activeCells: 256, maxActiveCells: 256, latentDim: 32 });
  assert.deepEqual(runtime.turnInward(4), { changed: true, activeCells: 64 });
  assert.deepEqual(runtime.turnInward(4), { changed: true, activeCells: 32 });
  assert.deepEqual(runtime.turnInward(4), { changed: false, activeCells: 32 });
});

test('deterministic seeds yield deterministic transactional state', () => {
  const options = { activeCells: 128, maxActiveCells: 128, latentDim: 16, seed: 12345 };
  const left = new MultimodalROM3D(options);
  const right = new MultimodalROM3D(options);
  const a = left.tick({ timeSeconds: 2, drive: 0.3 });
  const b = right.tick({ timeSeconds: 2, drive: 0.3 });
  assert.equal(a.committed, b.committed);
  assert.equal(a.reconstructionDistance, b.reconstructionDistance);
  assert.deepEqual(Array.from(left.committed), Array.from(right.committed));
});
