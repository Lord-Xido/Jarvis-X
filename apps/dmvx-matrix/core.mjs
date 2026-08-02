const Q16_SCALE = 65536;
const Q16_MIN = -2147483648;
const Q16_MAX = 2147483647;
const ROM_MAGIC = 0x584d5644; // "DMVX" in little endian
const ROM_VERSION = 1;
const ROM_HEADER_BYTES = 36;

export function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

export function toQ16(value) {
  if (!Number.isFinite(value)) {
    throw new TypeError('Q16.16 input must be finite');
  }
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

function makePositions(count, random) {
  const positions = new Float32Array(count * 3);
  for (let index = 0; index < count; index += 1) {
    const radius = Math.cbrt(random());
    const theta = random() * Math.PI * 2;
    const phi = Math.acos(random() * 2 - 1);
    positions[index * 3] = radius * Math.sin(phi) * Math.cos(theta);
    positions[index * 3 + 1] = radius * Math.sin(phi) * Math.sin(theta);
    positions[index * 3 + 2] = radius * Math.cos(phi);
  }
  return positions;
}

function hashBin(index, latentDim, seed) {
  let value = (index ^ seed) >>> 0;
  value = Math.imul(value ^ (value >>> 16), 0x7feb352d);
  value = Math.imul(value ^ (value >>> 15), 0x846ca68b);
  value = (value ^ (value >>> 16)) >>> 0;
  return value % latentDim;
}

export class DMVXMatrixRuntime {
  constructor({
    logicalExtent = 1000,
    maxActiveCells = 4096,
    activeCells = 2048,
    latentDim = 64,
    seed = 0x4d4f4147,
    reconstructionTolerance = 0.42,
    rho = 0.92,
    omegaGain = 0.08,
  } = {}) {
    if (!Number.isInteger(logicalExtent) || logicalExtent <= 0) {
      throw new RangeError('logicalExtent must be a positive integer');
    }
    if (!Number.isInteger(maxActiveCells) || maxActiveCells <= 0) {
      throw new RangeError('maxActiveCells must be a positive integer');
    }
    if (!Number.isInteger(activeCells) || activeCells <= 0 || activeCells > maxActiveCells) {
      throw new RangeError('activeCells must be within maxActiveCells');
    }
    if (!Number.isInteger(latentDim) || latentDim <= 0 || latentDim > activeCells) {
      throw new RangeError('latentDim must be positive and no larger than activeCells');
    }

    this.logicalExtent = logicalExtent;
    this.logicalCellCount = logicalExtent ** 3;
    this.maxActiveCells = maxActiveCells;
    this.activeCells = activeCells;
    this.latentDim = latentDim;
    this.seed = seed >>> 0;
    this.reconstructionTolerance = reconstructionTolerance;
    this.rho = rho;
    this.omegaGain = omegaGain;
    this.step = 0;
    this.version = 0;

    const random = mulberry32(this.seed);
    this.positions = makePositions(maxActiveCells, random);
    this.raw = new Float32Array(maxActiveCells);
    this.committed = new Float32Array(maxActiveCells);
    this.candidate = new Float32Array(maxActiveCells);
    this.decoded = new Float32Array(maxActiveCells);
    this.omega = new Float32Array(maxActiveCells);
    this.latentQ16 = new Int32Array(latentDim);
    this.lastResidual = new Float32Array(maxActiveCells);
    this.lastReceipt = null;

    this.sense(0, 0.5);
    this.committed.set(this.raw);
  }

  sense(timeSeconds, drive = 0.5) {
    const boundedDrive = clamp(Number(drive), 0, 1);
    for (let index = 0; index < this.activeCells; index += 1) {
      const x = this.positions[index * 3];
      const y = this.positions[index * 3 + 1];
      const z = this.positions[index * 3 + 2];
      const wave =
        0.5 +
        0.19 * Math.sin(2.4 * x + timeSeconds * (0.4 + boundedDrive)) +
        0.14 * Math.cos(2.1 * y - timeSeconds * 0.31) +
        0.11 * Math.sin(2.8 * z + boundedDrive * 2.0);
      this.raw[index] = clamp(wave, 0, 1);
    }
    return this.raw;
  }

  encode(input = this.raw) {
    const sums = new Float64Array(this.latentDim);
    const counts = new Uint32Array(this.latentDim);
    for (let index = 0; index < this.activeCells; index += 1) {
      const bin = hashBin(index, this.latentDim, this.seed);
      sums[bin] += input[index];
      counts[bin] += 1;
    }
    for (let bin = 0; bin < this.latentDim; bin += 1) {
      const average = counts[bin] ? sums[bin] / counts[bin] : 0;
      this.latentQ16[bin] = toQ16(average);
    }
    return this.latentQ16;
  }

  decode(latentQ16 = this.latentQ16) {
    for (let index = 0; index < this.activeCells; index += 1) {
      const bin = hashBin(index, this.latentDim, this.seed);
      const base = fromQ16(latentQ16[bin]);
      const x = this.positions[index * 3];
      const y = this.positions[index * 3 + 1];
      const z = this.positions[index * 3 + 2];
      const spatialCorrection = 0.025 * (x + y + z);
      this.decoded[index] = clamp(base + spatialCorrection, 0, 1);
    }
    return this.decoded;
  }

  propose({ timeSeconds = 0, drive = 0.5 } = {}) {
    this.sense(timeSeconds, drive);
    const started = globalThis.performance?.now?.() ?? Date.now();
    this.encode(this.raw);
    this.decode(this.latentQ16);

    let finite = true;
    let boundsValid = true;
    for (let index = 0; index < this.activeCells; index += 1) {
      const residual = this.raw[index] - this.decoded[index];
      this.lastResidual[index] = residual;
      const value = this.decoded[index] + this.omega[index];
      this.candidate[index] = clamp(value, 0, 1);
      finite &&= Number.isFinite(value);
      boundsValid &&= value >= -0.25 && value <= 1.25;
    }

    const reconstructionDistance = rmse(this.raw, this.candidate, this.activeCells);
    const elapsedMs = (globalThis.performance?.now?.() ?? Date.now()) - started;
    const budgetValid = this.activeCells <= this.maxActiveCells;
    const valid =
      finite &&
      boundsValid &&
      budgetValid &&
      reconstructionDistance <= this.reconstructionTolerance;

    return {
      valid,
      finite,
      boundsValid,
      budgetValid,
      reconstructionDistance,
      elapsedMs,
      activeCells: this.activeCells,
      candidate: this.candidate,
      latentQ16: this.latentQ16,
    };
  }

  commit(proposal) {
    if (!proposal || typeof proposal.valid !== 'boolean') {
      throw new TypeError('Invalid proposal');
    }

    const previousVersion = this.version;
    if (proposal.valid) {
      this.committed.set(this.candidate);
      for (let index = 0; index < this.activeCells; index += 1) {
        this.omega[index] = clamp(
          this.rho * this.omega[index] + this.omegaGain * this.lastResidual[index],
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
      elapsedMs: proposal.elapsedMs,
      activeCells: this.activeCells,
      latentDim: this.latentDim,
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
    return { changed, activeCells: this.activeCells };
  }

  residentBytes() {
    return (
      this.positions.byteLength +
      this.raw.byteLength +
      this.committed.byteLength +
      this.candidate.byteLength +
      this.decoded.byteLength +
      this.omega.byteLength +
      this.latentQ16.byteLength +
      this.lastResidual.byteLength
    );
  }

  logicalCompressionRatio() {
    return this.logicalCellCount / this.activeCells;
  }

  encodeROM() {
    const payloadBytes = (this.activeCells * 2 + this.latentDim) * 4;
    const bytes = new Uint8Array(ROM_HEADER_BYTES + payloadBytes);
    const view = new DataView(bytes.buffer);
    view.setUint32(0, ROM_MAGIC, true);
    view.setUint16(4, ROM_VERSION, true);
    view.setUint16(6, 0, true);
    view.setUint32(8, this.activeCells, true);
    view.setUint32(12, this.latentDim, true);
    view.setUint32(16, this.seed, true);
    view.setUint32(20, this.step, true);
    view.setUint32(24, this.version, true);
    view.setUint32(28, payloadBytes, true);
    view.setUint32(32, 0, true);

    let offset = ROM_HEADER_BYTES;
    for (let index = 0; index < this.activeCells; index += 1, offset += 4) {
      view.setInt32(offset, toQ16(this.committed[index]), true);
    }
    for (let index = 0; index < this.activeCells; index += 1, offset += 4) {
      view.setInt32(offset, toQ16(this.omega[index]), true);
    }
    for (let index = 0; index < this.latentDim; index += 1, offset += 4) {
      view.setInt32(offset, this.latentQ16[index], true);
    }

    const checksum = crc32(bytes.subarray(ROM_HEADER_BYTES));
    view.setUint32(32, checksum, true);
    return bytes;
  }

  decodeROM(bytes) {
    if (!(bytes instanceof Uint8Array)) {
      throw new TypeError('ROM must be a Uint8Array');
    }
    if (bytes.byteLength < ROM_HEADER_BYTES) {
      throw new RangeError('ROM image is too short');
    }

    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const magic = view.getUint32(0, true);
    const version = view.getUint16(4, true);
    const activeCells = view.getUint32(8, true);
    const latentDim = view.getUint32(12, true);
    const seed = view.getUint32(16, true);
    const step = view.getUint32(20, true);
    const stateVersion = view.getUint32(24, true);
    const payloadBytes = view.getUint32(28, true);
    const expectedChecksum = view.getUint32(32, true);

    if (magic !== ROM_MAGIC) throw new Error('ROM magic mismatch');
    if (version !== ROM_VERSION) throw new Error(`Unsupported ROM version ${version}`);
    if (activeCells <= 0 || activeCells > this.maxActiveCells) throw new Error('ROM active-cell budget invalid');
    if (latentDim !== this.latentDim) throw new Error('ROM latent dimension mismatch');
    if (seed !== this.seed) throw new Error('ROM seed mismatch');
    if (payloadBytes !== bytes.byteLength - ROM_HEADER_BYTES) throw new Error('ROM payload length mismatch');
    if (crc32(bytes.subarray(ROM_HEADER_BYTES)) !== expectedChecksum) throw new Error('ROM checksum mismatch');

    const expectedPayload = (activeCells * 2 + latentDim) * 4;
    if (payloadBytes !== expectedPayload) throw new Error('ROM payload schema mismatch');

    const committedCandidate = new Float32Array(this.maxActiveCells);
    const omegaCandidate = new Float32Array(this.maxActiveCells);
    const latentCandidate = new Int32Array(this.latentDim);
    let offset = ROM_HEADER_BYTES;

    for (let index = 0; index < activeCells; index += 1, offset += 4) {
      committedCandidate[index] = fromQ16(view.getInt32(offset, true));
    }
    for (let index = 0; index < activeCells; index += 1, offset += 4) {
      omegaCandidate[index] = fromQ16(view.getInt32(offset, true));
    }
    for (let index = 0; index < latentDim; index += 1, offset += 4) {
      latentCandidate[index] = view.getInt32(offset, true);
    }

    for (let index = 0; index < activeCells; index += 1) {
      if (!Number.isFinite(committedCandidate[index]) || committedCandidate[index] < 0 || committedCandidate[index] > 1) {
        throw new Error('ROM committed-state bounds invalid');
      }
      if (!Number.isFinite(omegaCandidate[index]) || omegaCandidate[index] < -0.25 || omegaCandidate[index] > 0.25) {
        throw new Error('ROM memory bounds invalid');
      }
    }

    this.activeCells = activeCells;
    this.committed.set(committedCandidate);
    this.omega.set(omegaCandidate);
    this.latentQ16.set(latentCandidate);
    this.step = step;
    this.version = stateVersion;
    return {
      activeCells,
      latentDim,
      step,
      version: stateVersion,
      bytes: bytes.byteLength,
      checksum: expectedChecksum,
    };
  }
}

export function bytesToBase64(bytes) {
  if (typeof Buffer !== 'undefined') {
    return Buffer.from(bytes).toString('base64');
  }
  let binary = '';
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary);
}

export function base64ToBytes(value) {
  if (typeof Buffer !== 'undefined') {
    return new Uint8Array(Buffer.from(value, 'base64'));
  }
  const binary = atob(value);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }
  return bytes;
}
