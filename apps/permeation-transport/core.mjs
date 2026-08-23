export const SHELL_BOUNDARIES = Object.freeze([0.33, 0.66, 1.0]);
export const SHELL_EQUILIBRIA = Object.freeze([0.165, 0.495, 0.83]);
export const DEFAULT_PARAMS = Object.freeze({
  omega: 1.8,
  kr: 0.85,
  Ares: 0.62,
  c: 0.12,
  mu: 0.38,
  D: 0.28,
  rho: 1.0,
});
export const REGIMES = Object.freeze({
  laminar: Object.freeze({ alpha: 0.15, gamma: 0.22, beta: 1.7 }),
  turbulent: Object.freeze({ alpha: 1.35, gamma: 0.85, beta: 3.2 }),
  quantum: Object.freeze({ alpha: 0.02, gamma: 0.03, beta: 0.6 }),
});

export function mulberry32(seed = 0x5eed1234) {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function createParticles(rho = 1, rng = Math.random) {
  const count = Math.floor(90 + rho * 220);
  const particles = [];
  for (let i = 0; i < count; i += 1) {
    const shell = Math.floor(rng() * 3);
    const r = SHELL_EQUILIBRIA[shell] + (rng() - 0.5) * 0.18;
    particles.push({
      r: Math.max(0.06, r),
      theta: rng() * Math.PI * 2,
      z: (rng() - 0.5) * 1.6,
      vr: (rng() - 0.5) * 0.1,
      vtheta: 0,
      vz: 0,
      shell,
      q: 0.15 + rng() * 0.35,
    });
  }
  return particles;
}

function densityBins(particles, bins = 24) {
  const occupancy = Array(bins).fill(0);
  for (const p of particles) {
    const idx = Math.min(bins - 1, Math.max(0, Math.floor(p.r * bins)));
    occupancy[idx] += 1;
  }
  return occupancy;
}

function wrapTheta(theta) {
  const tau = Math.PI * 2;
  theta %= tau;
  return theta < 0 ? theta + tau : theta;
}

export function stepSystem(state, dt, options = {}) {
  const params = { ...DEFAULT_PARAMS, ...(options.params || {}) };
  const regime = REGIMES[options.regime || 'laminar'];
  if (!regime) throw new Error(`Unknown regime: ${options.regime}`);
  const rng = options.rng || Math.random;
  const permeationEnabled = options.permeationEnabled !== false;
  const bins = options.bins || 24;
  const particles = state.particles;
  const occupancy = densityBins(particles, bins);
  const baseline = particles.length / bins || 1;
  const nextTime = state.time + dt;

  let velocitySquaredSum = 0;
  let radialVelocitySum = 0;
  let zMean = 0;
  let permeations = state.permeations || 0;

  for (const p of particles) {
    const bin = Math.min(bins - 1, Math.max(0, Math.floor(p.r * bins)));
    const densityDeviation = (occupancy[bin] - baseline) / baseline;
    const equilibrium = SHELL_EQUILIBRIA[p.shell];
    const angularNoise = (rng() - 0.5) * regime.alpha;
    const targetVTheta = params.omega * (0.4 + p.r) + angularNoise;
    p.vtheta += (targetVTheta - p.vtheta) * 0.12;

    const resonance = params.Ares * Math.sin(regime.beta * nextTime + 0.7 * p.theta);
    p.vr += (-params.kr * (p.r - equilibrium) + resonance - params.D * densityDeviation) * dt;
    p.vz += (regime.gamma * Math.cos(params.omega * nextTime + 4 * p.r) - 0.3 * p.vz) * dt;

    const damping = Math.max(0, 1 - params.c * dt);
    p.vr *= damping;
    p.vtheta *= damping;
    p.vz *= damping;

    if (permeationEnabled) {
      const probability = params.mu * (1 + p.q * Math.abs(p.vtheta)) * 0.12;
      const upperBoundary = p.shell < 2 ? SHELL_BOUNDARIES[p.shell] : 1;
      const lowerBoundary = p.shell > 0 ? SHELL_BOUNDARIES[p.shell - 1] : 0;
      const nearUpper = Math.abs(p.r - upperBoundary) < 0.07;
      const nearLower = Math.abs(p.r - lowerBoundary) < 0.07;
      if ((nearUpper || nearLower) && rng() < probability) {
        if (p.r > equilibrium && p.shell < 2 && rng() > 0.35) {
          p.shell += 1;
          p.vr += 0.18;
          permeations += 1;
        } else if (p.r < equilibrium && p.shell > 0 && rng() > 0.35) {
          p.shell -= 1;
          p.vr -= 0.18;
          permeations += 1;
        }
      }
    }

    p.r += p.vr * dt;
    p.theta = wrapTheta(p.theta + (p.vtheta / Math.max(0.15, p.r)) * dt);
    p.z += p.vz * dt;

    if (p.r < 0.05) { p.r = 0.05; p.vr *= -0.5; }
    if (p.r > 0.98) { p.r = 0.98; p.vr *= -0.6; }
    if (p.z > 1) { p.z = 1; p.vz *= -0.7; }
    if (p.z < -1) { p.z = -1; p.vz *= -0.7; }

    velocitySquaredSum += p.vr ** 2 + p.vtheta ** 2 + p.vz ** 2;
    radialVelocitySum += p.vr;
    zMean += p.z;
  }

  zMean /= particles.length || 1;
  let zVariance = 0;
  for (const p of particles) zVariance += (p.z - zMean) ** 2;
  const zSigma = Math.sqrt(zVariance / (particles.length || 1));

  return {
    ...state,
    time: nextTime,
    permeations,
    density: occupancy,
    diagnostics: {
      velocitySquaredSum,
      radialVelocitySum,
      zSigma,
    },
  };
}

export class FixedStepEngine {
  constructor({ step = 1 / 120, maxFrame = 0.05, maxSubsteps = 16 } = {}) {
    this.step = step;
    this.maxFrame = maxFrame;
    this.maxSubsteps = maxSubsteps;
    this.accumulator = 0;
  }

  advance(state, frameDeltaSeconds, options = {}) {
    const frame = Math.min(this.maxFrame, Math.max(0, frameDeltaSeconds));
    this.accumulator += frame;
    let substeps = 0;
    while (this.accumulator + Number.EPSILON >= this.step && substeps < this.maxSubsteps) {
      state = stepSystem(state, this.step, options);
      this.accumulator -= this.step;
      substeps += 1;
    }
    if (substeps === this.maxSubsteps) this.accumulator = 0;
    return { state, substeps, alpha: this.accumulator / this.step };
  }
}

export function createInitialState({ rho = DEFAULT_PARAMS.rho, seed = 0x5eed1234 } = {}) {
  const rng = mulberry32(seed);
  return {
    time: 0,
    permeations: 0,
    particles: createParticles(rho, rng),
    density: Array(24).fill(0),
    diagnostics: { velocitySquaredSum: 0, radialVelocitySum: 0, zSigma: 0 },
  };
}
