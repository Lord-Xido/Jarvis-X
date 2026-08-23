import test from 'node:test';
import assert from 'node:assert/strict';
import {
  DEFAULT_GEOMETRY,
  TAU,
  candidateGeometry,
  drMoagiPoint,
  geometryMetrics,
  geometryVectorLattice,
  inwardRadius,
  normalizeGeometry,
  optimizeGeometry,
} from './core.mjs';

test('refined equation preserves the Möbius seam', () => {
  for (const t of [0, 0.5, 1.2]) {
    for (const v of [-2, -1, 0, 1, 2]) {
      const a = drMoagiPoint(0, v, t, DEFAULT_GEOMETRY);
      const b = drMoagiPoint(TAU, -v, t, DEFAULT_GEOMETRY);
      assert.ok(Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z) < 1e-10);
    }
  }
});

test('inward radius contracts monotonically toward Rmin', () => {
  const p = normalizeGeometry(DEFAULT_GEOMETRY);
  const values = [0, 0.5, 1, 2, 4].map((t) => inwardRadius(t, p));
  for (let i = 1; i < values.length; i += 1) assert.ok(values[i] <= values[i - 1]);
  assert.ok(values.at(-1) >= p.Rmin);
});

test('geometry metrics remain finite and non-degenerate', () => {
  const m = geometryMetrics(DEFAULT_GEOMETRY, { uSamples: 18, vSamples: 7, times: [0, 0.5] });
  for (const value of Object.values(m)) assert.ok(Number.isFinite(value));
  assert.ok(m.seamRms < 1e-9);
  assert.ok(m.minArea > 0);
});

test('geometry search is a 26-neighbour bounded lattice', () => {
  const vectors = geometryVectorLattice();
  assert.equal(vectors.length, 26);
  for (const v of vectors) {
    for (const x of [v.shape, v.kinetics, v.inward]) assert.ok([-1, 0, 1].includes(x));
  }
  const candidate = candidateGeometry(DEFAULT_GEOMETRY, { shape: 1, kinetics: -1, inward: 1 });
  assert.ok(candidate.Rmin < candidate.R);
});

test('optimizer is deterministic and never worsens authoritative score', () => {
  const options = { uSamples: 18, vSamples: 7, times: [0, 0.5, 1], maxCandidates: 8 };
  const a = optimizeGeometry(DEFAULT_GEOMETRY, options);
  const b = optimizeGeometry(DEFAULT_GEOMETRY, options);
  assert.deepEqual(a, b);
  assert.ok(a.best.score <= a.baseline.score + Number.EPSILON);
  assert.deepEqual(
    a.authoritativeParams,
    a.promoted ? a.best.params : normalizeGeometry(DEFAULT_GEOMETRY),
  );
  assert.equal(a.claimStatus, 'internal_geometry_improvement_only');
});
