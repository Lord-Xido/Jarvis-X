(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.AE3D = Object.assign(root.AE3D || {}, api);
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';

  const POPCOUNT8 = new Uint8Array(256);
  for (let i = 0; i < 256; i++) {
    let x = i, c = 0;
    while (x) { x &= x - 1; c++; }
    POPCOUNT8[i] = c;
  }

  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a |= 0;
      a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function clamp(x, lo, hi) { return x < lo ? lo : x > hi ? hi : x; }

  function bytesFrom(input) {
    if (input instanceof Uint8Array) return input;
    if (ArrayBuffer.isView(input)) return new Uint8Array(input.buffer, input.byteOffset, input.byteLength);
    if (input instanceof ArrayBuffer) return new Uint8Array(input);
    if (typeof input === 'string') return new TextEncoder().encode(input);
    if (Array.isArray(input)) return Uint8Array.from(input);
    throw new TypeError('Input must be string, Uint8Array, ArrayBuffer, or number[]');
  }

  class RecursiveByteANN3D {
    constructor(options = {}) {
      this.grid = options.grid || 6;
      this.voxels = this.grid ** 3;
      this.dim = options.dim || 32;
      this.beta = options.beta ?? 0.85;
      this.recursionDepth = options.recursionDepth || 4;
      this.foldGain = options.foldGain ?? 0.08;
      this.learningRate = options.learningRate ?? 0.025;
      this.seed = options.seed ?? 1337;
      this.maxFrame = options.maxFrame || this.voxels;
      this.quantMode = options.quantMode || 'int8';
      this.rng = mulberry32(this.seed);
      this.clock = 0;

      this.embedding = new Float32Array(256 * this.dim);
      this.decoder = new Float32Array(256 * this.dim);
      this.latent = new Float32Array(this.voxels * this.dim);
      this.prevLatent = new Float32Array(this.voxels * this.dim);
      this.dequant = new Float32Array(this.voxels * this.dim);
      this.qLatent = new Int8Array(this.voxels * this.dim);
      this.scales = new Float32Array(this.voxels);
      this.activeMask = new Uint8Array(this.voxels);
      this.voxelEnergy = new Float32Array(this.voxels);
      this.lastInput = new Uint8Array(0);
      this.lastOutput = new Uint8Array(0);
      this.lastMetrics = this._emptyMetrics();
      this._scratch = new Float32Array(this.latent.length);
      this._scores = new Float32Array(256);
      this._initWeights();
    }

    _emptyMetrics() {
      return {
        byteAccuracy: 0,
        ber: 1,
        bitErrors: 0,
        mse: 0,
        psnr: 0,
        activeVoxels: 0,
        latentDelta: 0,
        stability: 0,
        quantMSE: 0,
        iterations: 0,
        elapsedMs: 0,
        bytesPerSecond: 0
      };
    }

    _initWeights() {
      for (let b = 0; b < 256; b++) {
        let norm = 0;
        const off = b * this.dim;
        for (let j = 0; j < this.dim; j++) {
          const v = (this.rng() * 2 - 1);
          this.embedding[off + j] = v;
          norm += v * v;
        }
        norm = Math.sqrt(norm) || 1;
        for (let j = 0; j < this.dim; j++) {
          const v = this.embedding[off + j] / norm;
          this.embedding[off + j] = v;
          this.decoder[off + j] = v;
        }
      }
    }

    reset(seed = this.seed) {
      this.seed = seed;
      this.rng = mulberry32(seed);
      this.clock = 0;
      this.latent.fill(0);
      this.prevLatent.fill(0);
      this.dequant.fill(0);
      this.qLatent.fill(0);
      this.scales.fill(0);
      this.activeMask.fill(0);
      this.voxelEnergy.fill(0);
      this.lastInput = new Uint8Array(0);
      this.lastOutput = new Uint8Array(0);
      this.lastMetrics = this._emptyMetrics();
      this._initWeights();
      return this;
    }

    setConfig(patch = {}) {
      if (patch.beta != null) this.beta = clamp(Number(patch.beta), 0, 1);
      if (patch.recursionDepth != null) this.recursionDepth = clamp(Math.floor(patch.recursionDepth), 1, 4096);
      if (patch.foldGain != null) this.foldGain = clamp(Number(patch.foldGain), 0, 1);
      if (patch.learningRate != null) this.learningRate = Math.max(1e-6, Number(patch.learningRate));
      return this.getConfig();
    }

    getConfig() {
      return {
        grid: this.grid,
        voxels: this.voxels,
        dim: this.dim,
        beta: this.beta,
        recursionDepth: this.recursionDepth,
        foldGain: this.foldGain,
        learningRate: this.learningRate,
        quantMode: this.quantMode,
        maxFrame: this.maxFrame,
        seed: this.seed
      };
    }

    index(x, y, z) {
      const g = this.grid;
      x = (x % g + g) % g;
      y = (y % g + g) % g;
      z = (z % g + g) % g;
      return x + g * y + g * g * z;
    }

    coords(p) {
      const g = this.grid;
      const z = Math.floor(p / (g * g));
      const rem = p - z * g * g;
      const y = Math.floor(rem / g);
      const x = rem - y * g;
      return [x, y, z];
    }

    _positional(i, j) {
      const a = (i + 1) * (j + 1);
      return 0.025 * Math.sin(a * 0.173) + 0.0125 * Math.cos(a * 0.071);
    }

    encode(input) {
      const bytes = bytesFrom(input).slice(0, this.maxFrame);
      this.lastInput = bytes;
      this.prevLatent.set(this.latent);
      this.latent.fill(0);
      this.activeMask.fill(0);
      this.voxelEnergy.fill(0);

      for (let i = 0; i < bytes.length; i++) {
        const p = i % this.voxels;
        const lo = p * this.dim;
        const eo = bytes[i] * this.dim;
        this.activeMask[p] = 1;
        let energy = 0;
        for (let j = 0; j < this.dim; j++) {
          const v = this.embedding[eo + j] + this._positional(i, j);
          this.latent[lo + j] = v;
          energy += v * v;
        }
        this.voxelEnergy[p] = energy;
      }
      return this.latent;
    }

    _foldOnce() {
      const d = this.dim;
      const gain = this.foldGain;
      if (gain <= 0) return;
      this._scratch.set(this.latent);
      const src = this._scratch;
      const dst = this.latent;
      const neighbors = [[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]];

      for (let p = 0; p < this.voxels; p++) {
        if (!this.activeMask[p]) continue;
        const [x,y,z] = this.coords(p);
        const po = p * d;
        for (let j = 0; j < d; j++) {
          let sum = 0, n = 0;
          for (const [dx,dy,dz] of neighbors) {
            const q = this.index(x + dx, y + dy, z + dz);
            if (!this.activeMask[q]) continue;
            sum += src[q * d + j];
            n++;
          }
          const mean = n ? sum / n : src[po + j];
          dst[po + j] = src[po + j] + gain * (mean - src[po + j]);
        }
      }
    }

    quantize() {
      const d = this.dim;
      let qerr = 0, qcount = 0;
      for (let p = 0; p < this.voxels; p++) {
        const off = p * d;
        if (!this.activeMask[p]) {
          this.scales[p] = 0;
          for (let j = 0; j < d; j++) {
            this.qLatent[off+j] = 0;
            this.dequant[off+j] = 0;
          }
          continue;
        }
        let maxAbs = 0;
        for (let j = 0; j < d; j++) maxAbs = Math.max(maxAbs, Math.abs(this.latent[off+j]));
        const scale = maxAbs > 0 ? maxAbs / 127 : 1 / 127;
        this.scales[p] = scale;
        for (let j = 0; j < d; j++) {
          const q = clamp(Math.round(this.latent[off+j] / scale), -127, 127);
          const dq = q * scale;
          this.qLatent[off+j] = q;
          this.dequant[off+j] = dq;
          const e = this.latent[off+j] - dq;
          qerr += e * e;
          qcount++;
        }
      }
      return qcount ? qerr / qcount : 0;
    }

    _decodeVector(vec, offset = 0) {
      let best = -Infinity, bestByte = 0;
      for (let b = 0; b < 256; b++) {
        const wo = b * this.dim;
        let score = 0;
        for (let j = 0; j < this.dim; j++) score += vec[offset+j] * this.decoder[wo+j];
        if (score > best) { best = score; bestByte = b; }
      }
      return bestByte;
    }

    decode(length = this.lastInput.length) {
      const n = Math.min(length, this.maxFrame);
      const out = new Uint8Array(n);
      for (let i = 0; i < n; i++) {
        const p = i % this.voxels;
        out[i] = this._decodeVector(this.dequant, p * this.dim);
      }
      this.lastOutput = out;
      return out;
    }

    _feedback(original, decoded) {
      const n = Math.min(original.length, decoded.length);
      const mixed = new Uint8Array(n);
      for (let i = 0; i < n; i++) {
        const xo = original[i] / 127.5 - 1;
        const xd = decoded[i] / 127.5 - 1;
        const xn = (1 - this.beta) * xo + this.beta * xd;
        mixed[i] = clamp(Math.round((xn + 1) * 127.5), 0, 255);
      }
      return mixed;
    }

    verify(original = this.lastInput, decoded = this.lastOutput, quantMSE = 0, elapsedMs = 0, iterations = 0) {
      const n = Math.min(original.length, decoded.length);
      let correct = 0, bitErrors = 0, mse = 0;
      for (let i = 0; i < n; i++) {
        const a = original[i], b = decoded[i];
        if (a === b) correct++;
        bitErrors += POPCOUNT8[a ^ b];
        const e = a - b;
        mse += e * e;
      }
      mse = n ? mse / n : 0;
      const psnr = mse === 0 ? Infinity : 10 * Math.log10((255 * 255) / mse);

      let delta2 = 0, base2 = 0, active = 0;
      for (let p = 0; p < this.voxels; p++) if (this.activeMask[p]) active++;
      for (let i = 0; i < this.latent.length; i++) {
        const d = this.latent[i] - this.prevLatent[i];
        delta2 += d*d;
        base2 += this.prevLatent[i]*this.prevLatent[i];
      }
      const latentDelta = Math.sqrt(delta2) / (Math.sqrt(base2) + 1e-9);
      const stability = clamp(1 - latentDelta, 0, 1);
      const bytesPerSecond = elapsedMs > 0 ? (n * 1000 / elapsedMs) : 0;

      this.lastMetrics = {
        byteAccuracy: n ? correct / n : 0,
        ber: n ? bitErrors / (8*n) : 0,
        bitErrors,
        mse,
        psnr,
        activeVoxels: active,
        latentDelta,
        stability,
        quantMSE,
        iterations,
        elapsedMs,
        bytesPerSecond
      };
      return this.lastMetrics;
    }

    run(input, options = {}) {
      const original = bytesFrom(input).slice(0, this.maxFrame);
      const depth = options.recursionDepth ?? this.recursionDepth;
      const t0 = typeof performance !== 'undefined' ? performance.now() : Date.now();
      let state = original;
      let qMSE = 0;
      let decoded = original;
      for (let k = 0; k < depth; k++) {
        this.encode(state);
        this._foldOnce();
        qMSE = this.quantize();
        decoded = this.decode(original.length);
        state = this._feedback(original, decoded);
        this.clock++;
      }
      this.lastInput = original;
      this.lastOutput = decoded;
      const t1 = typeof performance !== 'undefined' ? performance.now() : Date.now();
      const metrics = this.verify(original, decoded, qMSE, t1 - t0, depth);
      return { input: original, output: decoded, metrics, clock: this.clock };
    }

    step(input) { return this.run(input, { recursionDepth: 1 }); }

    train(input, options = {}) {
      const bytes = bytesFrom(input).slice(0, this.maxFrame);
      const epochs = options.epochs ?? 10;
      const lr = options.learningRate ?? this.learningRate;
      const t0 = typeof performance !== 'undefined' ? performance.now() : Date.now();
      let loss = 0;
      const gradH = new Float32Array(this.dim);

      for (let epoch = 0; epoch < epochs; epoch++) {
        loss = 0;
        for (let i = 0; i < bytes.length; i++) {
          const target = bytes[i];
          const eo = target * this.dim;
          let maxScore = -Infinity;
          for (let b = 0; b < 256; b++) {
            const wo = b * this.dim;
            let s = 0;
            for (let j = 0; j < this.dim; j++) {
              const h = this.embedding[eo+j] + this._positional(i,j);
              s += h * this.decoder[wo+j];
            }
            this._scores[b] = s;
            if (s > maxScore) maxScore = s;
          }
          let denom = 0;
          for (let b = 0; b < 256; b++) {
            const e = Math.exp(this._scores[b] - maxScore);
            this._scores[b] = e;
            denom += e;
          }
          const pTarget = this._scores[target] / denom;
          loss += -Math.log(Math.max(1e-12, pTarget));
          gradH.fill(0);

          for (let b = 0; b < 256; b++) {
            const p = this._scores[b] / denom;
            const g = p - (b === target ? 1 : 0);
            const wo = b * this.dim;
            for (let j = 0; j < this.dim; j++) gradH[j] += g * this.decoder[wo+j];
          }

          for (let b = 0; b < 256; b++) {
            const p = this._scores[b] / denom;
            const g = p - (b === target ? 1 : 0);
            const wo = b * this.dim;
            for (let j = 0; j < this.dim; j++) {
              const h = this.embedding[eo+j] + this._positional(i,j);
              this.decoder[wo+j] -= lr * g * h;
            }
          }
          for (let j = 0; j < this.dim; j++) this.embedding[eo+j] -= lr * gradH[j];
        }
        loss /= Math.max(1, bytes.length);
      }
      const t1 = typeof performance !== 'undefined' ? performance.now() : Date.now();
      return { epochs, loss, elapsedMs: t1 - t0 };
    }

    exportModel() {
      return {
        format: 'AE3D-SDK-1',
        config: this.getConfig(),
        clock: this.clock,
        embedding: Array.from(this.embedding),
        decoder: Array.from(this.decoder)
      };
    }

    importModel(model) {
      if (!model || model.format !== 'AE3D-SDK-1') throw new Error('Unsupported model format');
      if (!model.config || model.config.dim !== this.dim) throw new Error('Model dimension mismatch');
      if (model.embedding.length !== this.embedding.length || model.decoder.length !== this.decoder.length) {
        throw new Error('Model weight shape mismatch');
      }
      this.embedding.set(model.embedding);
      this.decoder.set(model.decoder);
      this.clock = model.clock || 0;
      this.setConfig(model.config);
      return this;
    }

    snapshot() {
      return {
        config: this.getConfig(),
        clock: this.clock,
        input: Array.from(this.lastInput),
        output: Array.from(this.lastOutput),
        activeMask: Array.from(this.activeMask),
        voxelEnergy: Array.from(this.voxelEnergy),
        metrics: { ...this.lastMetrics }
      };
    }
  }

  return { RecursiveByteANN3D, bytesFrom };
});
