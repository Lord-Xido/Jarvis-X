const PERF = globalThis.performance ?? { now: () => Date.now() };

export const LOGICAL_NODE_COUNT = 1_200_000;
export const VIRTUAL_BYTES = 1024 * 1024 * 1024;
export const PAGE_SIZE = 64 * 1024;

export const OPCODES = Object.freeze({
  LOAD_BITSTREAM: 0x10,
  AUTO_ENCODE: 0x20,
  DECODE_SPATIAL: 0x30,
  TENSOR_MAP: 0x40,
  INWARD_FOLD: 0x50,
  EMIT_STREAM: 0x60,
  HALT_SYNC: 0xFF,
});

export function encodeInstruction(opcode, operand = 0) {
  if (!Number.isInteger(opcode) || opcode < 0 || opcode > 0xFF) {
    throw new RangeError("opcode must fit in 8 bits");
  }
  if (!Number.isInteger(operand) || operand < 0 || operand > 0xFFFFFF) {
    throw new RangeError("operand must fit in 24 bits");
  }
  return (((opcode & 0xFF) << 24) | (operand & 0xFFFFFF)) >>> 0;
}

export function decodeInstruction(word) {
  const value = Number(word) >>> 0;
  return Object.freeze({
    opcode: (value >>> 24) & 0xFF,
    operand: value & 0xFFFFFF,
  });
}

export class SparseRingMemory {
  constructor({
    capacity = VIRTUAL_BYTES,
    pageSize = PAGE_SIZE,
    maxResidentPages = 64,
  } = {}) {
    if (!Number.isSafeInteger(capacity) || capacity <= 0) {
      throw new RangeError("capacity must be a positive safe integer");
    }
    if (!Number.isSafeInteger(pageSize) || pageSize <= 0) {
      throw new RangeError("pageSize must be a positive safe integer");
    }
    if (!Number.isSafeInteger(maxResidentPages) || maxResidentPages <= 0) {
      throw new RangeError("maxResidentPages must be positive");
    }
    this.capacity = capacity;
    this.pageSize = pageSize;
    this.maxResidentPages = maxResidentPages;
    this.pages = new Map();
    this.head = 0;
  }

  _pageIndex(address) {
    return Math.floor(address / this.pageSize);
  }

  _offset(address) {
    return address % this.pageSize;
  }

  _page(index, create = false) {
    let page = this.pages.get(index);
    if (!page && create) {
      if (this.pages.size >= this.maxResidentPages) {
        throw new RangeError("resident page budget exceeded");
      }
      page = new Uint8Array(this.pageSize);
      this.pages.set(index, page);
    }
    return page;
  }

  write(bytes) {
    if (!(bytes instanceof Uint8Array)) {
      throw new TypeError("write expects Uint8Array");
    }
    for (let i = 0; i < bytes.length; i += 1) {
      const address = this.head;
      const page = this._page(this._pageIndex(address), true);
      page[this._offset(address)] = bytes[i];
      this.head = (this.head + 1) % this.capacity;
    }
    return bytes.length;
  }

  read(address, length) {
    if (!Number.isSafeInteger(address) || address < 0 || address >= this.capacity) {
      throw new RangeError("address out of range");
    }
    if (!Number.isSafeInteger(length) || length < 0) {
      throw new RangeError("length must be non-negative");
    }
    const out = new Uint8Array(length);
    for (let i = 0; i < length; i += 1) {
      const current = (address + i) % this.capacity;
      const page = this._page(this._pageIndex(current), false);
      out[i] = page ? page[this._offset(current)] : 0;
    }
    return out;
  }

  clear() {
    this.pages.clear();
    this.head = 0;
  }

  stats() {
    return Object.freeze({
      virtualBytes: this.capacity,
      pageSize: this.pageSize,
      residentPages: this.pages.size,
      residentBytes: this.pages.size * this.pageSize,
      maxResidentPages: this.maxResidentPages,
      head: this.head,
    });
  }
}

function xorshift32(seed) {
  let x = seed >>> 0 || 0x9E3779B9;
  return () => {
    x ^= x << 13;
    x ^= x >>> 17;
    x ^= x << 5;
    x >>>= 0;
    return x / 0x100000000;
  };
}

function clamp(value, lo, hi) {
  return Math.max(lo, Math.min(hi, value));
}

export class CognitiveMatrixEngine {
  constructor({
    activeNodes = 1200,
    latentDim = 8,
    gamma = 0.72,
    rho = 0.86,
    learningRate = 0.045,
    fixedPointIterations = 8,
    ctrThreshold = 0.08,
    seed = 0x4D4F4147,
    memory = new SparseRingMemory(),
  } = {}) {
    if (!Number.isInteger(activeNodes) || activeNodes < 8 || activeNodes > 100_000) {
      throw new RangeError("activeNodes must be between 8 and 100000");
    }
    if (!Number.isInteger(latentDim) || latentDim < 2 || latentDim > 128) {
      throw new RangeError("latentDim must be between 2 and 128");
    }
    this.logicalNodes = LOGICAL_NODE_COUNT;
    this.activeNodes = activeNodes;
    this.latentDim = latentDim;
    this.gamma = clamp(gamma, 0.01, 0.99);
    this.rho = clamp(rho, 0.0, 0.999);
    this.learningRate = clamp(learningRate, 0.0001, 0.2);
    this.fixedPointIterations = Math.max(1, Math.min(64, fixedPointIterations | 0));
    this.ctrThreshold = Math.max(1e-8, Number(ctrThreshold));
    this.memory = memory;

    const rand = xorshift32(seed);
    const size = activeNodes * latentDim;
    this.latent = new Float64Array(size);
    this.committed = new Float64Array(size);
    this.candidate = new Float64Array(size);
    this.weights = new Float64Array(latentDim);
    for (let d = 0; d < latentDim; d += 1) {
      this.weights[d] = (rand() - 0.5) * 0.24;
    }
    for (let i = 0; i < size; i += 1) {
      this.latent[i] = (rand() - 0.5) * 0.08;
    }
    this.committed.set(this.latent);
    this.omega = 0;
    this.version = 0;
    this.last = null;
    this.program = new Uint32Array([
      encodeInstruction(OPCODES.LOAD_BITSTREAM),
      encodeInstruction(OPCODES.AUTO_ENCODE),
      encodeInstruction(OPCODES.TENSOR_MAP),
      encodeInstruction(OPCODES.INWARD_FOLD),
      encodeInstruction(OPCODES.DECODE_SPATIAL),
      encodeInstruction(OPCODES.EMIT_STREAM),
      encodeInstruction(OPCODES.HALT_SYNC),
    ]);
  }

  setControls({ gamma, rho, learningRate, fixedPointIterations } = {}) {
    if (gamma !== undefined) this.gamma = clamp(Number(gamma), 0.01, 0.99);
    if (rho !== undefined) this.rho = clamp(Number(rho), 0, 0.999);
    if (learningRate !== undefined) {
      this.learningRate = clamp(Number(learningRate), 0.0001, 0.2);
    }
    if (fixedPointIterations !== undefined) {
      this.fixedPointIterations = Math.max(
        1,
        Math.min(64, Number(fixedPointIterations) | 0),
      );
    }
  }

  _encode(input) {
    const x = clamp(Number(input), -4, 4);
    const width = this.latentDim;
    for (let n = 0; n < this.activeNodes; n += 1) {
      const phase = ((n * 0.61803398875) % 1) * Math.PI * 2;
      for (let d = 0; d < width; d += 1) {
        const i = n * width + d;
        const drive = Math.sin(phase + d * 0.37) * x * 0.08;
        this.candidate[i] =
          this.committed[i] * 0.78 +
          drive +
          this.weights[d] * 0.14 +
          this.omega * 0.025;
      }
    }
  }

  _tensorMap() {
    const width = this.latentDim;
    const nodes = this.activeNodes;
    const tmp = new Float64Array(this.candidate.length);
    for (let n = 0; n < nodes; n += 1) {
      const prev = (n + nodes - 1) % nodes;
      const next = (n + 1) % nodes;
      for (let d = 0; d < width; d += 1) {
        const i = n * width + d;
        const neighbour =
          0.5 *
          (this.candidate[prev * width + d] +
            this.candidate[next * width + d]);
        tmp[i] = Math.tanh(
          this.candidate[i] * 0.82 +
            neighbour * 0.16 +
            this.weights[d] * 0.02,
        );
      }
    }
    this.candidate.set(tmp);
  }

  _inwardFold() {
    const width = this.latentDim;
    let residual = Infinity;
    const tmp = new Float64Array(this.candidate.length);
    for (let k = 0; k < this.fixedPointIterations; k += 1) {
      let sq = 0;
      for (let n = 0; n < this.activeNodes; n += 1) {
        const prev = (n + this.activeNodes - 1) % this.activeNodes;
        const next = (n + 1) % this.activeNodes;
        for (let d = 0; d < width; d += 1) {
          const i = n * width + d;
          const local = this.candidate[i];
          const neighbour =
            0.5 *
            (this.candidate[prev * width + d] +
              this.candidate[next * width + d]);
          const phi = Math.tanh(
            local * 0.72 +
              neighbour * 0.22 +
              this.omega * 0.04 +
              this.weights[d] * 0.02,
          );
          const nextValue = (1 - this.gamma) * local + this.gamma * phi;
          tmp[i] = nextValue;
          const delta = nextValue - local;
          sq += delta * delta;
        }
      }
      residual = Math.sqrt(sq / tmp.length);
      this.candidate.set(tmp);
    }
    return residual;
  }

  _decode() {
    let sum = 0;
    let energy = 0;
    const width = this.latentDim;
    for (let n = 0; n < this.activeNodes; n += 1) {
      let local = 0;
      for (let d = 0; d < width; d += 1) {
        local += this.candidate[n * width + d] * (d + 1) / width;
      }
      local /= width;
      sum += local;
      energy += local * local;
    }
    return {
      reconstruction: sum / this.activeNodes,
      latentEnergy: energy / this.activeNodes,
    };
  }

  _adapt(residual) {
    const direction = clamp(residual, -1, 1);
    for (let d = 0; d < this.latentDim; d += 1) {
      const gradient = direction * (d + 1) / this.latentDim;
      this.weights[d] = clamp(
        this.weights[d] + this.learningRate * gradient * 0.02,
        -0.5,
        0.5,
      );
    }
  }

  runMacrocycle(input = 0.5) {
    const start = PERF.now();
    const target = clamp(Number(input), -4, 4);
    const bytes = new TextEncoder().encode(
      JSON.stringify({ version: this.version, target }),
    );

    let decoded = { reconstruction: 0, latentEnergy: 0 };
    let fixedPointResidual = Infinity;
    const trace = [];

    for (let pc = 0; pc < this.program.length; pc += 1) {
      const { opcode } = decodeInstruction(this.program[pc]);
      if (opcode === OPCODES.LOAD_BITSTREAM) {
        this.memory.write(bytes);
        trace.push("LOAD_BITSTREAM");
      } else if (opcode === OPCODES.AUTO_ENCODE) {
        this._encode(target);
        trace.push("AUTO_ENCODE");
      } else if (opcode === OPCODES.TENSOR_MAP) {
        this._tensorMap();
        trace.push("TENSOR_MAP");
      } else if (opcode === OPCODES.INWARD_FOLD) {
        fixedPointResidual = this._inwardFold();
        trace.push("INWARD_FOLD");
      } else if (opcode === OPCODES.DECODE_SPATIAL) {
        decoded = this._decode();
        trace.push("DECODE_SPATIAL");
      } else if (opcode === OPCODES.EMIT_STREAM) {
        trace.push("EMIT_STREAM");
      } else if (opcode === OPCODES.HALT_SYNC) {
        trace.push("HALT_SYNC");
        break;
      } else {
        throw new Error(\`unknown opcode 0x\${opcode.toString(16)}\`);
      }
    }

    const reconstructionResidual = target * 0.08 - decoded.reconstruction;
    this.omega =
      this.rho * this.omega +
      (1 - this.rho) * Math.abs(reconstructionResidual);

    const ctrEnergy =
      reconstructionResidual * reconstructionResidual +
      fixedPointResidual * fixedPointResidual +
      Math.max(0, decoded.latentEnergy - 1) ** 2;

    const accepted =
      Number.isFinite(ctrEnergy) &&
      Number.isFinite(fixedPointResidual) &&
      ctrEnergy <= this.ctrThreshold;

    if (accepted) {
      this.committed.set(this.candidate);
      this.version += 1;
      this._adapt(reconstructionResidual);
    } else {
      this.candidate.set(this.committed);
    }

    this.latent.set(this.committed);
    const elapsedMs = PERF.now() - start;
    const stats = this.memory.stats();
    this.last = Object.freeze({
      accepted,
      version: this.version,
      input: target,
      reconstruction: decoded.reconstruction,
      reconstructionResidual,
      fixedPointResidual,
      ctrEnergy,
      omega: this.omega,
      latentEnergy: decoded.latentEnergy,
      macrocycleMs: elapsedMs,
      logicalNodes: this.logicalNodes,
      activeNodes: this.activeNodes,
      residentPages: stats.residentPages,
      residentBytes: stats.residentBytes,
      trace: Object.freeze(trace),
    });
    return this.last;
  }

  sampleNodes(count = 128) {
    const n = Math.max(1, Math.min(this.activeNodes, count | 0));
    const out = [];
    const stride = Math.max(1, Math.floor(this.activeNodes / n));
    for (let node = 0; node < this.activeNodes && out.length < n; node += stride) {
      const base = node * this.latentDim;
      let magnitude = 0;
      for (let d = 0; d < this.latentDim; d += 1) {
        const v = this.committed[base + d];
        magnitude += v * v;
      }
      out.push(
        Object.freeze({
          logicalIndex: Math.floor(
            (node / this.activeNodes) * this.logicalNodes,
          ),
          activeIndex: node,
          magnitude: Math.sqrt(magnitude),
        }),
      );
    }
    return Object.freeze(out);
  }
}
