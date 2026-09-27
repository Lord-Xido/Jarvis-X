const Q16_SCALE = 65536;
const Q16_MIN = -2147483648;
const Q16_MAX = 2147483647;
const ROM_MAGIC = 0x52334d44;
const ROM_VERSION = 2;
const ROM_HEADER_BYTES = 64;
const CELL_BYTES = 24;

export const MODALITY = Object.freeze({
  text: 1 << 0,
  audio: 1 << 1,
  image: 1 << 2,
  video: 1 << 3,
  tensor: 1 << 4,
  sensor: 1 << 5,
});

export function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

export function toQ16(value) {
  if (!Number.isFinite(value)) throw new TypeError('Q16.16 input must be finite');
  return clamp(Math.round(value * Q16_SCALE), Q16_MIN, Q16_MAX);
}

export function fromQ16(value) {
  return value / Q16_SCALE;
}

export function mulberry32(seed) {
  let state = seed >>> 0;
  return function random() {
    state = (state + 0x6d2b79f5) >>> 0;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

export function crc32(bytes) {
  let crc = 0xffffffff;
  for (const byte of bytes) {
    crc ^= byte;
    for (let bit = 0; bit < 8; bit += 1) {
      const mask = -(crc & 1);
      crc = (crc >>> 1) ^ (0xedb88320 & mask);
    }
  }
  return (crc ^ 0xffffffff) >>> 0;
}

export function rmse(reference, candidate, length = reference.length) {
  if (length <= 0 || length > reference.length || length > candidate.length) {
    throw new RangeError('Invalid RMSE length');
  }
  let sum = 0;
  for (let index = 0; index < length; index += 1) {
    const delta = reference[index] - candidate[index];
    sum += delta * delta;
  }
  return Math.sqrt(sum / length);
}

export function morton3D(x, y, z) {
  for (const [name, value] of [['x', x], ['y', y], ['z', z]]) {
    if (!Number.isInteger(value) || value < 0 || value >= 2 ** 20) {
      throw new RangeError(name + ' must be an integer in [0, 2^20)');
    }
  }
  let code = 0n;
  for (let bit = 0n; bit < 20n; bit += 1n) {
    const shift = Number(bit);
    code |= BigInt((x >>> shift) & 1) << (3n * bit);
    code |= BigInt((y >>> shift) & 1) << (3n * bit + 1n);
    code |= BigInt((z >>> shift) & 1) << (3n * bit + 2n);
  }
  return code;
}

export function unmorton3D(code) {
  if (typeof code !== 'bigint' || code < 0n || code >= (1n << 60n)) {
    throw new RangeError('Morton code must be an unsigned 60-bit bigint');
  }
  let x = 0;
  let y = 0;
  let z = 0;
  for (let bit = 0n; bit < 20n; bit += 1n) {
    const shift = Number(bit);
    x |= Number((code >> (3n * bit)) & 1n) << shift;
    y |= Number((code >> (3n * bit + 1n)) & 1n) << shift;
    z |= Number((code >> (3n * bit + 2n)) & 1n) << shift;
  }
  return { x: x >>> 0, y: y >>> 0, z: z >>> 0 };
}

function hashBin(index, latentDim, seed) {
  let value = (index ^ seed) >>> 0;
  value = Math.imul(value ^ (value >>> 16), 0x7feb352d);
  value = Math.imul(value ^ (value >>> 15), 0x846ca68b);
  value = (value ^ (value >>> 16)) >>> 0;
  return value % latentDim;
}

function modalityUnion(flags, length) {
  let mask = 0;
  for (let index = 0; index < length; index += 1) mask |= flags[index];
  return mask >>> 0;
}

function modalityNameToFlag(name) {
  const flag = MODALITY[name];
  if (!flag) throw new RangeError('Unsupported modality: ' + name);
  return flag;
}

export class MultimodalROM3D {
  constructor({
    logicalExtent = 1_000_000,
    maxActiveCells = 4096,
    activeCells = 3072,
    latentDim = 96,
    seed = 0x4d4f4147,
    reconstructionTolerance = 0.42,
    fixedPointTolerance = 5e-4,
    fixedPointIterations = 10,
    contraction = 0.35,
    rho = 0.90,
    omegaGain = 0.04,
  } = {}) {
    if (!Number.isInteger(logicalExtent) || logicalExtent <= 0 || logicalExtent >= 2 ** 20) {
      throw new RangeError('logicalExtent must be a positive integer below 2^20');
    }
    if (!Number.isInteger(maxActiveCells) || maxActiveCells <= 0) {
      throw new RangeError('maxActiveCells must be positive');
    }
    if (!Number.isInteger(activeCells) || activeCells <= 0 || activeCells > maxActiveCells) {
      throw new RangeError('activeCells must be within maxActiveCells');
    }
    if (!Number.isInteger(latentDim) || latentDim <= 0 || latentDim > activeCells) {
      throw new RangeError('latentDim must be positive and no larger than activeCells');
    }
    if (!Number.isFinite(contraction) || contraction < 0 || contraction >= 1) {
      throw new RangeError('contraction must satisfy 0 <= contraction < 1');
    }

    this.logicalExtent = logicalExtent;
    this.logicalCellCount = BigInt(logicalExtent) ** 3n;
    this.maxActiveCells = maxActiveCells;
    this.activeCells = activeCells;
    this.latentDim = latentDim;
    this.seed = seed >>> 0;
    this.reconstructionTolerance = reconstructionTolerance;
    this.fixedPointTolerance = fixedPointTolerance;
    this.fixedPointIterations = fixedPointIterations;
    this.contraction = contraction;
    this.rho = rho;
    this.omegaGain = omegaGain;
    this.step = 0;
    this.version = 0;

    this.x = new Uint32Array(maxActiveCells);
    this.y = new Uint32Array(maxActiveCells);
    this.z = new Uint32Array(maxActiveCells);
    this.modality = new Uint8Array(maxActiveCells);
    this.observed = new Float32Array(maxActiveCells);
    this.committed = new Float32Array(maxActiveCells);
    this.candidate = new Float32Array(maxActiveCells);
    this.decoded = new Float32Array(maxActiveCells);
    this.omega = new Float32Array(maxActiveCells);
    this.residual = new Float32Array(maxActiveCells);
    this.latentTargetQ16 = new Int32Array(latentDim);
    this.latentQ16 = new Int32Array(latentDim);
    this.lastReceipt = null;

    const random = mulberry32(this.seed);
    for (let index = 0; index < maxActiveCells; index += 1) {
      this.x[index] = Math.floor(random() * logicalExtent);
      this.y[index] = Math.floor(random() * logicalExtent);
      this.z[index] = Math.floor(random() * logicalExtent);
    }

    this.senseSynthetic(0, 0.5);
    this.committed.set(this.observed);
    this.encode(this.observed);
    this.latentQ16.set(this.latentTargetQ16);
  }

  normalizedPosition(index) {
    if (!Number.isInteger(index) || index < 0 || index >= this.activeCells) {
      throw new RangeError('cell index out of range');
    }
    const d = Math.max(1, this.logicalExtent - 1);
    return {
      x: this.x[index] / d * 2 - 1,
      y: this.y[index] / d * 2 - 1,
      z: this.z[index] / d * 2 - 1,
    };
  }

  mortonAt(index) {
    if (!Number.isInteger(index) || index < 0 || index >= this.activeCells) {
      throw new RangeError('cell index out of range');
    }
    return morton3D(this.x[index], this.y[index], this.z[index]);
  }

  senseSynthetic(timeSeconds = 0, drive = 0.5) {
    const boundedDrive = clamp(Number(drive), 0, 1);
    const modalityCycle = [
      MODALITY.text,
      MODALITY.audio,
      MODALITY.image,
      MODALITY.video,
      MODALITY.tensor,
      MODALITY.sensor,
    ];

    for (let index = 0; index < this.activeCells; index += 1) {
      const p = this.normalizedPosition(index);
      const flag = modalityCycle[index % modalityCycle.length];
      this.modality[index] = flag;
      const modalPhase = (index % modalityCycle.length) * 0.37;
      const wave =
        0.50 +
        0.17 * Math.sin(2.7 * p.x + timeSeconds * (0.35 + boundedDrive) + modalPhase) +
        0.13 * Math.cos(2.2 * p.y - timeSeconds * 0.29) +
        0.10 * Math.sin(3.1 * p.z + boundedDrive * 1.9);
      this.observed[index] = clamp(wave, 0, 1);
    }
    return this.observed;
  }

  ingest(frames) {
    if (!Array.isArray(frames) || frames.length === 0) {
      throw new TypeError('frames must be a non-empty array');
    }
    this.observed.fill(0, 0, this.activeCells);
    this.modality.fill(0, 0, this.activeCells);

    let cursor = 0;
    for (const frame of frames) {
      if (!frame || typeof frame !== 'object') throw new TypeError('frame must be an object');
      const flag = modalityNameToFlag(frame.modality);
      if (!Array.isArray(frame.values) && !ArrayBuffer.isView(frame.values)) {
        throw new TypeError('frame.values must be an Array or TypedArray');
      }
      for (const raw of frame.values) {
        if (cursor >= this.activeCells) break;
        const value = Number(raw);
        if (!Number.isFinite(value)) throw new TypeError('modality values must be finite');
        this.observed[cursor] = clamp(value, 0, 1);
        this.modality[cursor] = flag;
        cursor += 1;
      }
      if (cursor >= this.activeCells) break;
    }

    if (cursor === 0) throw new RangeError('frames contain no values');
    for (let index = cursor; index < this.activeCells; index += 1) {
      this.observed[index] = this.observed[index % cursor];
      this.modality[index] = this.modality[index % cursor];
    }

    return {
      samples: this.activeCells,
      modalityMask: modalityUnion(this.modality, this.activeCells),
    };
  }

  encode(input = this.observed) {
    const sums = new Float64Array(this.latentDim);
    const weights = new Float64Array(this.latentDim);

    for (let index = 0; index < this.activeCells; index += 1) {
      const bin = hashBin(index, this.latentDim, this.seed);
      const flag = this.modality[index] || MODALITY.tensor;
      const gain = 1 + (Math.log2(flag) % 6) * 0.015;
      sums[bin] += input[index] * gain;
      weights[bin] += gain;
    }

    for (let bin = 0; bin < this.latentDim; bin += 1) {
      const value = weights[bin] > 0 ? sums[bin] / weights[bin] : 0;
      this.latentTargetQ16[bin] = toQ16(value);
    }
    return this.latentTargetQ16;
  }

  refineLatent() {
    let maxResidual = Infinity;
    for (let iteration = 0; iteration < this.fixedPointIterations; iteration += 1) {
      maxResidual = 0;
      for (let bin = 0; bin < this.latentDim; bin += 1) {
        const current = fromQ16(this.latentQ16[bin]);
        const target = fromQ16(this.latentTargetQ16[bin]);
        const next = target + this.contraction * (current - target);
        maxResidual = Math.max(maxResidual, Math.abs(next - current));
        this.latentQ16[bin] = toQ16(next);
      }
      if (maxResidual <= this.fixedPointTolerance) break;
    }
    return maxResidual;
  }

  decode(latent = this.latentQ16) {
    for (let index = 0; index < this.activeCells; index += 1) {
      const bin = hashBin(index, this.latentDim, this.seed);
      const base = fromQ16(latent[bin]);
      const p = this.normalizedPosition(index);
      const spatial = 0.022 * (p.x + p.y + p.z);
      this.decoded[index] = clamp(base + spatial, 0, 1);
    }
    return this.decoded;
  }

  propose({ timeSeconds = 0, drive = 0.5, frames = null } = {}) {
    if (frames) this.ingest(frames);
    else this.senseSynthetic(timeSeconds, drive);

    const started = globalThis.performance?.now?.() ?? Date.now();
    this.encode(this.observed);
    const fixedPointResidual = this.refineLatent();
    this.decode(this.latentQ16);

    let finite = true;
    let boundsValid = true;
    for (let index = 0; index < this.activeCells; index += 1) {
      const error = this.observed[index] - this.decoded[index];
      this.residual[index] = error;
      const value = this.decoded[index] + this.omega[index];
      this.candidate[index] = clamp(value, 0, 1);
      finite &&= Number.isFinite(value);
      boundsValid &&= value >= -0.25 && value <= 1.25;
    }

    const reconstructionDistance = rmse(this.observed, this.candidate, this.activeCells);
    const elapsedMs = (globalThis.performance?.now?.() ?? Date.now()) - started;
    const budgetValid = this.activeCells <= this.maxActiveCells;
    const fixedPointValid = fixedPointResidual <= Math.max(this.fixedPointTolerance, 2 / Q16_SCALE);
    const valid =
      finite &&
      boundsValid &&
      budgetValid &&
      fixedPointValid &&
      reconstructionDistance <= this.reconstructionTolerance;

    return {
      valid,
      finite,
      boundsValid,
      budgetValid,
      fixedPointValid,
      fixedPointResidual,
      reconstructionDistance,
      elapsedMs,
      activeCells: this.activeCells,
      latentDim: this.latentDim,
      modalityMask: modalityUnion(this.modality, this.activeCells),
    };
  }

  commit(proposal) {
    if (!proposal || typeof proposal.valid !== 'boolean') {
      throw new TypeError('Invalid proposal');
    }
    const previousVersion = this.version;

    if (proposal.valid) {
      this.committed.set(this.candidate.subarray(0, this.activeCells), 0);
      for (let index = 0; index < this.activeCells; index += 1) {
        this.omega[index] = clamp(
          this.rho * this.omega[index] + this.omegaGain * this.residual[index],
          -0.25,
          0.25,
        );
      }
      this.version += 1;
    }
    this.step += 1;

    this.lastReceipt = {
      step: this.step,
      previousVersion,
      version: this.version,
      committed: proposal.valid,
      reconstructionDistance: proposal.reconstructionDistance,
      fixedPointResidual: proposal.fixedPointResidual,
      elapsedMs: proposal.elapsedMs,
      activeCells: this.activeCells,
      latentDim: this.latentDim,
      modalityMask: proposal.modalityMask,
      residentBytes: this.residentBytes(),
    };
    return this.lastReceipt;
  }

  tick(options = {}) {
    return this.commit(this.propose(options));
  }

  turnInward(factor = 2) {
    if (!Number.isFinite(factor) || factor <= 1) {
      throw new RangeError('Inward factor must be greater than one');
    }
    const next = Math.max(this.latentDim, Math.ceil(this.activeCells / factor));
    const changed = next < this.activeCells;
    this.activeCells = next;
    return { changed, activeCells: next };
  }

  residentBytes() {
    return (
      this.x.byteLength +
      this.y.byteLength +
      this.z.byteLength +
      this.modality.byteLength +
      this.observed.byteLength +
      this.committed.byteLength +
      this.candidate.byteLength +
      this.decoded.byteLength +
      this.omega.byteLength +
      this.residual.byteLength +
      this.latentTargetQ16.byteLength +
      this.latentQ16.byteLength
    );
  }

  logicalCompressionRatio() {
    return Number(this.logicalCellCount / BigInt(this.activeCells));
  }

  encodeROM() {
    const payloadBytes = this.activeCells * CELL_BYTES + this.latentDim * 4;
    const bytes = new Uint8Array(ROM_HEADER_BYTES + payloadBytes);
    const view = new DataView(bytes.buffer);
    const mask = modalityUnion(this.modality, this.activeCells);

    view.setUint32(0, ROM_MAGIC, true);
    view.setUint16(4, ROM_VERSION, true);
    view.setUint16(6, 0, true);
    view.setUint32(8, this.logicalExtent, true);
    view.setUint32(12, this.activeCells, true);
    view.setUint32(16, this.latentDim, true);
    view.setUint32(20, this.maxActiveCells, true);
    view.setUint32(24, this.seed, true);
    view.setUint32(28, this.step, true);
    view.setUint32(32, this.version, true);
    view.setUint32(36, mask, true);
    view.setUint32(40, payloadBytes, true);
    view.setUint32(44, 0, true);

    let offset = ROM_HEADER_BYTES;
    for (let index = 0; index < this.activeCells; index += 1) {
      view.setUint32(offset, this.x[index], true);
      view.setUint32(offset + 4, this.y[index], true);
      view.setUint32(offset + 8, this.z[index], true);
      view.setUint8(offset + 12, this.modality[index]);
      view.setInt32(offset + 16, toQ16(this.committed[index]), true);
      view.setInt32(offset + 20, toQ16(this.omega[index]), true);
      offset += CELL_BYTES;
    }
    for (let bin = 0; bin < this.latentDim; bin += 1, offset += 4) {
      view.setInt32(offset, this.latentQ16[bin], true);
    }

    const checksum = crc32(bytes.subarray(ROM_HEADER_BYTES));
    view.setUint32(44, checksum, true);
    return bytes;
  }

  decodeROM(bytes) {
    if (!(bytes instanceof Uint8Array)) throw new TypeError('ROM must be a Uint8Array');
    if (bytes.byteLength < ROM_HEADER_BYTES) throw new RangeError('ROM image is too short');

    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const magic = view.getUint32(0, true);
    const version = view.getUint16(4, true);
    const logicalExtent = view.getUint32(8, true);
    const activeCells = view.getUint32(12, true);
    const latentDim = view.getUint32(16, true);
    const seed = view.getUint32(24, true);
    const step = view.getUint32(28, true);
    const stateVersion = view.getUint32(32, true);
    const payloadBytes = view.getUint32(40, true);
    const expectedChecksum = view.getUint32(44, true);

    if (magic !== ROM_MAGIC) throw new Error('ROM magic mismatch');
    if (version !== ROM_VERSION) throw new Error('Unsupported ROM version ' + version);
    if (logicalExtent !== this.logicalExtent) throw new Error('ROM logical extent mismatch');
    if (activeCells <= 0 || activeCells > this.maxActiveCells) throw new Error('ROM active-cell budget invalid');
    if (latentDim !== this.latentDim) throw new Error('ROM latent dimension mismatch');
    if (seed !== this.seed) throw new Error('ROM seed mismatch');
    if (payloadBytes !== bytes.byteLength - ROM_HEADER_BYTES) throw new Error('ROM payload length mismatch');
    if (payloadBytes !== activeCells * CELL_BYTES + latentDim * 4) throw new Error('ROM payload schema mismatch');
    if (crc32(bytes.subarray(ROM_HEADER_BYTES)) !== expectedChecksum) throw new Error('ROM checksum mismatch');

    const xCandidate = new Uint32Array(this.maxActiveCells);
    const yCandidate = new Uint32Array(this.maxActiveCells);
    const zCandidate = new Uint32Array(this.maxActiveCells);
    const modalityCandidate = new Uint8Array(this.maxActiveCells);
    const committedCandidate = new Float32Array(this.maxActiveCells);
    const omegaCandidate = new Float32Array(this.maxActiveCells);
    const latentCandidate = new Int32Array(this.latentDim);

    let offset = ROM_HEADER_BYTES;
    for (let index = 0; index < activeCells; index += 1) {
      xCandidate[index] = view.getUint32(offset, true);
      yCandidate[index] = view.getUint32(offset + 4, true);
      zCandidate[index] = view.getUint32(offset + 8, true);
      modalityCandidate[index] = view.getUint8(offset + 12);
      committedCandidate[index] = fromQ16(view.getInt32(offset + 16, true));
      omegaCandidate[index] = fromQ16(view.getInt32(offset + 20, true));
      offset += CELL_BYTES;

      if (
        xCandidate[index] >= this.logicalExtent ||
        yCandidate[index] >= this.logicalExtent ||
        zCandidate[index] >= this.logicalExtent
      ) throw new Error('ROM coordinate bounds invalid');
      if (!Number.isFinite(committedCandidate[index]) || committedCandidate[index] < 0 || committedCandidate[index] > 1) {
        throw new Error('ROM committed-state bounds invalid');
      }
      if (!Number.isFinite(omegaCandidate[index]) || omegaCandidate[index] < -0.25 || omegaCandidate[index] > 0.25) {
        throw new Error('ROM memory bounds invalid');
      }
    }
    for (let bin = 0; bin < latentDim; bin += 1, offset += 4) {
      latentCandidate[bin] = view.getInt32(offset, true);
    }

    this.activeCells = activeCells;
    this.x.set(xCandidate);
    this.y.set(yCandidate);
    this.z.set(zCandidate);
    this.modality.set(modalityCandidate);
    this.committed.set(committedCandidate);
    this.omega.set(omegaCandidate);
    this.latentQ16.set(latentCandidate);
    this.latentTargetQ16.set(latentCandidate);
    this.step = step;
    this.version = stateVersion;

    return {
      activeCells,
      latentDim,
      step,
      version: stateVersion,
      bytes: bytes.byteLength,
      checksum: expectedChecksum,
      modalityMask: modalityUnion(this.modality, activeCells),
    };
  }
}

export function bytesToBase64(bytes) {
  if (typeof Buffer !== 'undefined') return Buffer.from(bytes).toString('base64');
  let binary = '';
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary);
}

export function base64ToBytes(value) {
  if (typeof Buffer !== 'undefined') return new Uint8Array(Buffer.from(value, 'base64'));
  const binary = atob(value);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index);
  return bytes;
}
