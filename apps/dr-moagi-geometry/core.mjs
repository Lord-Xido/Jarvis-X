export const TAU = Math.PI * 2;

export const DEFAULT_GEOMETRY = Object.freeze({
  R: 5.0,
  Rmin: 3.6,
  chi: 0.16,
  omega: 0.50,
  alpha: 0.30,
  beta: 0.70,
  gamma: 0.30,
  delta: 0.50,
  kappa: 2,
  lambda: 0.50,
});

export const GEOMETRY_BOUNDS = Object.freeze({
  R: [3.0, 8.0],
  Rmin: [2.0, 7.5],
  chi: [0.0, 0.8],
  omega: [0.05, 1.5],
  alpha: [0.05, 1.2],
  beta: [0.05, 1.8],
  gamma: [0.0, 0.8],
  delta: [0.0, 1.2],
  kappa: [1, 4],
  lambda: [0.05, 1.5],
});

export function clamp(value, lo, hi) {
  return Math.max(lo, Math.min(hi, value));
}

export function normalizeGeometry(params = {}) {
  const p = { ...DEFAULT_GEOMETRY, ...params };
  for (const [name, [lo, hi]] of Object.entries(GEOMETRY_BOUNDS)) {
    p[name] = clamp(Number(p[name]), lo, hi);
  }
  p.kappa = Math.round(p.kappa);
  p.Rmin = Math.min(p.R - 0.25, p.Rmin);
  p.Rmin = Math.max(GEOMETRY_BOUNDS.Rmin[0], p.Rmin);
  return p;
}

export function inwardRadius(t, params = DEFAULT_GEOMETRY) {
  const p = normalizeGeometry(params);
  return p.Rmin + (p.R - p.Rmin) * Math.exp(-p.chi * Math.max(0, t));
}

// Topology-preserving refinement of the Dr Moagi kinetic geometry.
// The secondary perturbation uses a half-angle phase so the Möbius seam
// P(0,v,t) == P(2π,-v,t) remains continuous for integer kappa.
export function drMoagiPoint(u, v, t, params = DEFAULT_GEOMETRY) {
  const p = normalizeGeometry(params);
  const twist = 0.5 * u + p.omega * t;
  const orbit = u + p.alpha * t;
  const secondary = 0.5 * u + p.beta * t;
  const baseRadius = inwardRadius(t, p) + v * Math.cos(twist);
  const tangentPerturbation = p.gamma * v * Math.sin(secondary);
  const tangentX = -Math.sin(orbit);
  const tangentZ = Math.cos(orbit);

  const x = baseRadius * Math.cos(orbit) + tangentPerturbation * tangentX;
  const y = v * Math.sin(twist) + p.delta * Math.sin(p.kappa * u + p.lambda * t);
  const z = baseRadius * Math.sin(orbit) + tangentPerturbation * tangentZ;
  return { x, y, z };
}

function sub(a, b) {
  return { x: a.x - b.x, y: a.y - b.y, z: a.z - b.z };
}

function norm2(a) {
  return a.x * a.x + a.y * a.y + a.z * a.z;
}

function cross(a, b) {
  return {
    x: a.y * b.z - a.z * b.y,
    y: a.z * b.x - a.x * b.z,
    z: a.x * b.y - a.y * b.x,
  };
}

function mean(values) {
  if (!values.length) return 0;
  return values.reduce((a, b) => a + b, 0) / values.length;
}

export function geometryMetrics(params = DEFAULT_GEOMETRY, options = {}) {
  const p = normalizeGeometry(params);
  const uSamples = options.uSamples ?? 36;
  const vSamples = options.vSamples ?? 9;
  const times = options.times ?? [0.0, 0.5, 1.0, 1.5];
  const dt = options.dt ?? 1 / 60;
  const du = TAU / uSamples;
  const dv = 4 / Math.max(1, vSamples - 1);

  const seamSq = [];
  const speedSq = [];
  const accelSq = [];
  const areas = [];
  const radii = [];

  for (const t of times) {
    for (let j = 0; j < vSamples; j += 1) {
      const v = -2 + j * dv;
      const a = drMoagiPoint(0, v, t, p);
      const b = drMoagiPoint(TAU, -v, t, p);
      seamSq.push(norm2(sub(a, b)));
    }

    for (let i = 0; i < uSamples; i += 1) {
      const u = i * du;
      for (let j = 0; j < vSamples; j += 1) {
        const v = -2 + j * dv;
        const now = drMoagiPoint(u, v, t, p);
        const prev = drMoagiPoint(u, v, Math.max(0, t - dt), p);
        const next = drMoagiPoint(u, v, t + dt, p);
        const velocityDenom = Math.max(dt, t > 0 ? 2 * dt : dt);
        const velocity = {
          x: (next.x - prev.x) / velocityDenom,
          y: (next.y - prev.y) / velocityDenom,
          z: (next.z - prev.z) / velocityDenom,
        };
        const accel = {
          x: (next.x - 2 * now.x + prev.x) / (dt * dt),
          y: (next.y - 2 * now.y + prev.y) / (dt * dt),
          z: (next.z - 2 * now.z + prev.z) / (dt * dt),
        };
        speedSq.push(norm2(velocity));
        accelSq.push(norm2(accel));
        radii.push(Math.sqrt(now.x * now.x + now.z * now.z));

        const pu = drMoagiPoint(u + du * 0.5, v, t, p);
        const vProbe = v >= 2 - 1e-12 ? v - dv * 0.5 : v + dv * 0.5;
        const pv = drMoagiPoint(u, clamp(vProbe, -2, 2), t, p);
        const tu = sub(pu, now);
        const tv = sub(pv, now);
        areas.push(Math.sqrt(norm2(cross(tu, tv))));
      }
    }
  }

  const meanArea = mean(areas);
  const minArea = Math.min(...areas);
  const areaVariance = mean(areas.map((x) => (x - meanArea) ** 2));
  const areaCv = meanArea > 0 ? Math.sqrt(areaVariance) / meanArea : Infinity;
  const speedRms = Math.sqrt(mean(speedSq));
  const accelerationRms = Math.sqrt(mean(accelSq));
  const seamRms = Math.sqrt(mean(seamSq));
  const radialSpan = Math.max(...radii) - Math.min(...radii);
  const contraction = p.R - inwardRadius(times[times.length - 1], p);

  return {
    seamRms,
    minArea,
    meanArea,
    areaCv,
    speedRms,
    accelerationRms,
    radialSpan,
    contraction,
  };
}

export function geometryObjective(params = DEFAULT_GEOMETRY, options = {}) {
  const p = normalizeGeometry(params);
  const m = geometryMetrics(p, options);
  const collapsePenalty = m.minArea < 0.002 ? (0.002 - m.minArea) ** 2 * 1e6 : 0;
  const motionLow = Math.max(0, 0.35 - m.speedRms);
  const motionHigh = Math.max(0, m.speedRms - 3.0);
  const contractionLow = Math.max(0, 0.15 - m.contraction);
  const contractionHigh = Math.max(0, m.contraction - 1.8);
  const frequencyCost = p.omega ** 2 + p.alpha ** 2 + p.beta ** 2 + p.lambda ** 2;
  const expressivity = p.gamma ** 2 + p.delta ** 2;

  const score =
    1e6 * m.seamRms ** 2 +
    8.0 * m.areaCv ** 2 +
    collapsePenalty +
    0.015 * m.accelerationRms ** 2 +
    0.06 * motionHigh ** 2 +
    3.0 * motionLow ** 2 +
    2.5 * contractionLow ** 2 +
    0.8 * contractionHigh ** 2 +
    0.03 * frequencyCost +
    0.30 * Math.max(0, 0.18 - expressivity) ** 2;

  return { score, metrics: m };
}

export function geometryVectorLattice() {
  const values = [-1, 0, 1];
  const vectors = [];
  for (const shape of values) {
    for (const kinetics of values) {
      for (const inward of values) {
        if (shape || kinetics || inward) vectors.push({ shape, kinetics, inward });
      }
    }
  }
  vectors.sort((a, b) => {
    const ma = Math.abs(a.shape) + Math.abs(a.kinetics) + Math.abs(a.inward);
    const mb = Math.abs(b.shape) + Math.abs(b.kinetics) + Math.abs(b.inward);
    if (ma !== mb) return ma - mb;
    return a.shape - b.shape || a.kinetics - b.kinetics || a.inward - b.inward;
  });
  return vectors;
}

export function candidateGeometry(base, vector) {
  const p = normalizeGeometry(base);
  const q = { ...p };
  if (vector.shape) {
    const scale = vector.shape < 0 ? 0.88 : 1.12;
    q.gamma *= scale;
    q.delta *= scale;
  }
  if (vector.kinetics) {
    const scale = vector.kinetics < 0 ? 0.90 : 1.10;
    q.omega *= scale;
    q.alpha *= scale;
    q.beta *= scale;
    q.lambda *= scale;
  }
  if (vector.inward) {
    if (vector.inward < 0) {
      q.chi *= 0.82;
      q.Rmin += 0.18;
    } else {
      q.chi *= 1.18;
      q.Rmin -= 0.18;
    }
  }
  return normalizeGeometry(q);
}

export function optimizeGeometry(base = DEFAULT_GEOMETRY, options = {}) {
  const incumbent = normalizeGeometry(base);
  const baseline = geometryObjective(incumbent, options);
  const maxCandidates = Math.min(26, options.maxCandidates ?? 26);
  const minRelativeImprovement = options.minRelativeImprovement ?? 0.005;
  const maxAreaRegression = options.maxAreaRegression ?? 0.05;
  const seamTolerance = options.seamTolerance ?? 1e-9;

  const evaluations = [{
    vector: { shape: 0, kinetics: 0, inward: 0 },
    params: incumbent,
    ...baseline,
    stage: 'baseline',
  }];

  for (const vector of geometryVectorLattice().slice(0, maxCandidates)) {
    const params = candidateGeometry(incumbent, vector);
    const result = geometryObjective(params, options);
    evaluations.push({ vector, params, ...result, stage: 'candidate' });
  }

  const best = evaluations.reduce((a, b) => (b.score < a.score ? b : a));
  const relativeImprovement = baseline.score > 0
    ? Math.max(0, (baseline.score - best.score) / baseline.score)
    : 0;
  const areaFloor = baseline.metrics.minArea * (1 - maxAreaRegression);
  const promoted =
    best.stage !== 'baseline' &&
    relativeImprovement >= minRelativeImprovement &&
    best.metrics.seamRms <= seamTolerance &&
    best.metrics.minArea >= areaFloor &&
    Number.isFinite(best.score);

  return {
    baseline: evaluations[0],
    best,
    promoted,
    relativeImprovement,
    promotedParams: promoted ? best.params : null,
    authoritativeParams: promoted ? best.params : incumbent,
    evaluations,
    claimStatus: 'internal_geometry_improvement_only',
  };
}
