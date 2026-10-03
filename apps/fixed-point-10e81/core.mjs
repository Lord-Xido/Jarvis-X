// Jarvis-X sparse virtual fixed-point engine.
// Logical geometry: (10^9)^3 = 10^27 coordinates per axis.
// Total logical cardinality: (10^27)^3 = 10^81 sites.
// Physical execution is bounded to activeSupport samples.

export const AXIS_EXTENT = 10n ** 27n;
export const LOGICAL_CARDINALITY = 10n ** 81n;
export const BITS_PER_AXIS = 90;
export const LOGICAL_ADDRESS_BITS = 270;
export const DEFAULT_ACTIVE_SUPPORT = 1024;

const MASK64 = (1n << 64n) - 1n;

function splitmix64(x) {
  let z = BigInt.asUintN(64, x + 0x9e3779b97f4a7c15n);
  z = BigInt.asUintN(64, (z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n);
  z = BigInt.asUintN(64, (z ^ (z >> 27n)) * 0x94d049bb133111ebn);
  return BigInt.asUintN(64, z ^ (z >> 31n));
}

function word128(seed, lane) {
  const a = splitmix64(seed + BigInt(lane) * 0x9e3779b97f4a7c15n);
  const b = splitmix64(a ^ 0xd1b54a32d192ed03n);
  return ((a & MASK64) << 64n) | (b & MASK64);
}

export function packCoordinate(x, y, z) {
  for (const entry of [['x', x], ['y', y], ['z', z]]) {
    const name = entry[0];
    const value = entry[1];
    if (typeof value !== 'bigint' || value < 0n || value >= AXIS_EXTENT) {
      throw new RangeError(name + ' outside [0, 10^27)');
    }
  }
  return x + AXIS_EXTENT * (y + AXIS_EXTENT * z);
}

export function unpackCoordinate(address) {
  if (typeof address !== 'bigint' || address < 0n || address >= LOGICAL_CARDINALITY) {
    throw new RangeError('address outside [0, 10^81)');
  }
  const plane = AXIS_EXTENT * AXIS_EXTENT;
  const z = address / plane;
  const rem = address % plane;
  const y = rem / AXIS_EXTENT;
  const x = rem % AXIS_EXTENT;
  return {x, y, z};
}

export class FixedPoint10E81 {
  constructor({
    activeSupport = DEFAULT_ACTIVE_SUPPORT,
    seed = 1n,
    eta = 0.24,
    rho = 0.92,
  } = {}) {
    if (!Number.isInteger(activeSupport) || activeSupport < 1 || activeSupport > 1_000_000) {
      throw new RangeError('activeSupport must be an integer in [1, 1_000_000]');
    }
    if (!(eta > 0) || !Number.isFinite(eta)) throw new RangeError('eta must be finite and > 0');
    if (!(rho >= 0 && rho < 1) || !Number.isFinite(rho)) throw new RangeError('rho must be in [0,1)');

    this.activeSupport = activeSupport;
    this.seed = BigInt(seed);
    this.initialEta = eta;
    this.rho = rho;

    this.coordinates = new Array(activeSupport);
    this.target = new Float64Array(activeSupport);
    this.latent = new Float64Array(activeSupport);
    this.omega = new Float64Array(activeSupport);
    this.residual = new Float64Array(activeSupport);

    this.reset();
  }

  coordinateFor(index, epoch = 0n) {
    if (!Number.isInteger(index) || index < 0 || index >= this.activeSupport) {
      throw new RangeError('active-support index out of range');
    }
    const base = this.seed
      ^ (BigInt(index + 1) * 0x94d049bb133111ebn)
      ^ (BigInt(epoch) * 0xbf58476d1ce4e5b9n);

    return [
      word128(base + 11n, 0) % AXIS_EXTENT,
      word128(base + 29n, 1) % AXIS_EXTENT,
      word128(base + 47n, 2) % AXIS_EXTENT,
    ];
  }

  targetFromCoordinate(coord) {
    const x = coord[0];
    const y = coord[1];
    const z = coord[2];
    const m = 1_000_003n;
    const a = Number(x % m) / Number(m);
    const b = Number(y % m) / Number(m);
    const c = Number(z % m) / Number(m);

    return 0.55 * Math.sin(2 * Math.PI * a)
         + 0.30 * Math.cos(4 * Math.PI * b)
         + 0.15 * Math.sin(6 * Math.PI * c);
  }

  remap(epoch = this.epoch + 1n) {
    this.epoch = BigInt(epoch);
    for (let i = 0; i < this.activeSupport; ++i) {
      const coord = this.coordinateFor(i, this.epoch);
      this.coordinates[i] = coord;
      this.target[i] = this.targetFromCoordinate(coord);
      this.latent[i] = 0;
      this.omega[i] = 0;
      this.residual[i] = this.target[i];
    }

    this.iteration = 0;
    this.eta = this.initialEta;
    this.commits = 0;
    this.rollbacks = 0;
    this.lastDecision = 'REMAPPED';
    this.lastCandidateLoss = this.loss();
    this.lastAcceptedLoss = this.lastCandidateLoss;
    return this.telemetry();
  }

  reset() {
    this.epoch = -1n;
    return this.remap(0n);
  }

  decodedValue(index, source = this.latent) {
    return Math.tanh(source[index]);
  }

  loss(source = this.latent) {
    let sum = 0;
    for (let i = 0; i < this.activeSupport; ++i) {
      const error = this.target[i] - Math.tanh(source[i]);
      sum += error * error;
    }
    return sum / this.activeSupport;
  }

  metrics() {
    let sq = 0;
    let maxAbs = 0;

    for (let i = 0; i < this.activeSupport; ++i) {
      const r = this.target[i] - Math.tanh(this.latent[i]);
      this.residual[i] = r;
      sq += r * r;
      maxAbs = Math.max(maxAbs, Math.abs(r));
    }

    return {
      mse: sq / this.activeSupport,
      residualL2: Math.sqrt(sq),
      residualMaxAbs: maxAbs,
    };
  }

  step() {
    const before = this.loss();
    const candidate = new Float64Array(this.activeSupport);

    for (let i = 0; i < this.activeSupport; ++i) {
      const decoded = Math.tanh(this.latent[i]);
      const r = this.target[i] - decoded;
      candidate[i] = this.latent[i]
        + this.eta * (0.80 * r + 0.20 * this.omega[i]);
    }

    const candidateLoss = this.loss(candidate);
    const accepted = Number.isFinite(candidateLoss)
      && candidateLoss <= before + 1e-15;

    if (accepted) {
      this.latent.set(candidate);
      this.commits += 1;
      this.lastDecision = 'COMMIT';
      this.eta = Math.min(0.42, this.eta * 1.015);
      this.lastAcceptedLoss = candidateLoss;
    } else {
      this.rollbacks += 1;
      this.lastDecision = 'ROLLBACK';
      this.eta = Math.max(0.001, this.eta * 0.5);
      this.lastAcceptedLoss = before;
    }

    this.lastCandidateLoss = candidateLoss;

    const metrics = this.metrics();
    for (let i = 0; i < this.activeSupport; ++i) {
      this.omega[i] = this.rho * this.omega[i]
        + (1 - this.rho) * this.residual[i];
    }

    this.iteration += 1;
    return {
      accepted,
      before,
      candidateLoss,
      ...metrics,
      ...this.telemetry(),
    };
  }

  converged({mse = 1e-8, residualMaxAbs = 1e-4} = {}) {
    const m = this.metrics();
    return m.mse <= mse && m.residualMaxAbs <= residualMaxAbs;
  }

  telemetry() {
    const m = this.metrics();
    return {
      axisExtent: AXIS_EXTENT.toString(),
      logicalCardinality: LOGICAL_CARDINALITY.toString(),
      bitsPerAxis: BITS_PER_AXIS,
      logicalAddressBits: LOGICAL_ADDRESS_BITS,
      activeSupport: this.activeSupport,
      epoch: this.epoch.toString(),
      iteration: this.iteration,
      eta: this.eta,
      rho: this.rho,
      commits: this.commits,
      rollbacks: this.rollbacks,
      decision: this.lastDecision,
      mse: m.mse,
      residualL2: m.residualL2,
      residualMaxAbs: m.residualMaxAbs,
      candidateLoss: this.lastCandidateLoss,
      acceptedLoss: this.lastAcceptedLoss,
    };
  }
}
