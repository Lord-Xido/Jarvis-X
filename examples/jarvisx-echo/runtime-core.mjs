const textEncoder = new TextEncoder();
const textDecoder = new TextDecoder();

export const RUNTIME_STATES = Object.freeze({
  IDLE: "IDLE",
  PROCESSING: "PROCESSING",
  ECHOING: "ECHOING",
  ERROR: "ERROR",
});

const ALLOWED_TRANSITIONS = Object.freeze({
  IDLE: new Set(["PROCESSING"]),
  PROCESSING: new Set(["ECHOING", "ERROR"]),
  ECHOING: new Set(["IDLE", "PROCESSING", "ERROR"]),
  ERROR: new Set(["IDLE"]),
});

function assertFiniteNumber(value, name) {
  if (!Number.isFinite(value)) throw new TypeError(`${name} must be finite`);
}

function fnv1a(bytes) {
  let hash = 0x811c9dc5;
  for (const byte of bytes) {
    hash ^= byte;
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  return hash >>> 0;
}

function stableHash(value) {
  const bytes = textEncoder.encode(JSON.stringify(value));
  return fnv1a(bytes).toString(16).padStart(8, "0");
}

function requireBytes(view, offset, length) {
  if (offset < 0 || length < 0 || offset + length > view.byteLength) {
    throw new RangeError(`ROM read outside buffer at ${offset} for ${length} bytes`);
  }
}

export function encodeROM(cities) {
  if (!Array.isArray(cities)) throw new TypeError("cities must be an array");
  const normalized = cities.map((city) => {
    const record = {
      name: String(city.name),
      country: String(city.country),
      lat: Number(city.lat),
      lon: Number(city.lon),
      pop: Number(city.pop),
      elev: Number(city.elev),
      tz: Number(city.tz),
    };
    assertFiniteNumber(record.lat, "lat");
    assertFiniteNumber(record.lon, "lon");
    assertFiniteNumber(record.pop, "pop");
    assertFiniteNumber(record.elev, "elev");
    assertFiniteNumber(record.tz, "tz");
    if (record.pop < 0 || record.pop > 0xffffffff) throw new RangeError("population outside uint32");
    if (record.elev < -32768 || record.elev > 32767) throw new RangeError("elevation outside int16");
    if (record.tz < -32768 || record.tz > 32767) throw new RangeError("timezone outside int16");
    return record;
  });

  const stringList = Array.from(new Set(normalized.flatMap((city) => [city.name, city.country])));
  const stringMap = new Map(stringList.map((value, index) => [value, index]));
  const encodedStrings = stringList.map((value) => {
    const bytes = textEncoder.encode(value);
    if (bytes.length > 255) throw new RangeError(`ROM string exceeds 255 bytes: ${value}`);
    return bytes;
  });

  const headerSize = 18;
  const stringTableSize = encodedStrings.reduce((total, bytes) => total + 1 + bytes.length, 0);
  const cityRecordSize = 20;
  const totalSize = headerSize + stringTableSize + normalized.length * cityRecordSize;
  const buffer = new ArrayBuffer(totalSize);
  const view = new DataView(buffer);
  let offset = 0;

  view.setUint32(offset, 0x4a58524d, false); offset += 4;
  view.setUint8(offset, 0x03); offset += 1;
  view.setUint8(offset, 0x07); offset += 1;
  view.setUint16(offset, normalized.length, false); offset += 2;
  view.setUint16(offset, stringList.length, false); offset += 2;
  view.setUint32(offset, headerSize, false); offset += 4;
  view.setUint32(offset, 0, false); offset += 4;

  for (const bytes of encodedStrings) {
    view.setUint8(offset, bytes.length); offset += 1;
    new Uint8Array(buffer, offset, bytes.length).set(bytes);
    offset += bytes.length;
  }

  for (const city of normalized) {
    view.setFloat32(offset, city.lat, false); offset += 4;
    view.setFloat32(offset, city.lon, false); offset += 4;
    view.setUint32(offset, city.pop, false); offset += 4;
    view.setInt16(offset, city.elev, false); offset += 2;
    view.setInt16(offset, city.tz, false); offset += 2;
    view.setUint16(offset, stringMap.get(city.name), false); offset += 2;
    view.setUint16(offset, stringMap.get(city.country), false); offset += 2;
  }

  const checksum = fnv1a(new Uint8Array(buffer, headerSize));
  view.setUint32(14, checksum, false);
  return buffer;
}

export function decodeROM(buffer) {
  if (!(buffer instanceof ArrayBuffer) || buffer.byteLength < 18) {
    throw new TypeError("ROM must be an ArrayBuffer with a complete header");
  }
  const view = new DataView(buffer);
  requireBytes(view, 0, 18);
  let offset = 0;
  const magic = view.getUint32(offset, false); offset += 4;
  if (magic !== 0x4a58524d) throw new Error("Invalid JARVIS X ROM magic");
  const version = view.getUint8(offset); offset += 1;
  const flags = view.getUint8(offset); offset += 1;
  const cityCount = view.getUint16(offset, false); offset += 2;
  const stringCount = view.getUint16(offset, false); offset += 2;
  const tableOffset = view.getUint32(offset, false); offset += 4;
  const checksum = view.getUint32(offset, false); offset += 4;
  if (tableOffset !== 18 || tableOffset >= buffer.byteLength) throw new RangeError("Invalid ROM table offset");
  const actualChecksum = fnv1a(new Uint8Array(buffer, tableOffset));
  if (actualChecksum !== checksum) throw new Error("ROM checksum mismatch");

  offset = tableOffset;
  const strings = [];
  for (let index = 0; index < stringCount; index += 1) {
    requireBytes(view, offset, 1);
    const length = view.getUint8(offset); offset += 1;
    requireBytes(view, offset, length);
    strings.push(textDecoder.decode(new Uint8Array(buffer, offset, length)));
    offset += length;
  }

  const cities = [];
  for (let index = 0; index < cityCount; index += 1) {
    requireBytes(view, offset, 20);
    const lat = view.getFloat32(offset, false); offset += 4;
    const lon = view.getFloat32(offset, false); offset += 4;
    const pop = view.getUint32(offset, false); offset += 4;
    const elev = view.getInt16(offset, false); offset += 2;
    const tz = view.getInt16(offset, false); offset += 2;
    const nameIndex = view.getUint16(offset, false); offset += 2;
    const countryIndex = view.getUint16(offset, false); offset += 2;
    if (nameIndex >= strings.length || countryIndex >= strings.length) {
      throw new RangeError("ROM record references an invalid string index");
    }
    cities.push({ lat, lon, pop, elev, tz, name: strings[nameIndex], country: strings[countryIndex] });
  }
  if (offset !== buffer.byteLength) throw new Error("ROM contains trailing or malformed bytes");
  return Object.freeze({ version, flags, checksum, cities: Object.freeze(cities), bufferSize: buffer.byteLength });
}

export class TraceBus {
  constructor(maxEvents = 512) {
    this.maxEvents = maxEvents;
    this.sequence = 0;
    this.events = [];
    this.listeners = new Set();
  }

  emit(stage, payload = {}) {
    const event = Object.freeze({
      sequence: ++this.sequence,
      stage,
      timestamp: performance.now(),
      payload: Object.freeze({ ...payload }),
    });
    this.events.push(event);
    if (this.events.length > this.maxEvents) this.events.splice(0, this.events.length - this.maxEvents);
    for (const listener of this.listeners) listener(event);
    return event;
  }

  subscribe(listener) {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  snapshot() {
    return this.events.slice();
  }
}

export class RuntimeFSM {
  constructor(trace) {
    this.state = RUNTIME_STATES.IDLE;
    this.trace = trace;
  }

  transition(next, reason = "") {
    if (next === this.state) return this.state;
    if (!ALLOWED_TRANSITIONS[this.state]?.has(next)) {
      throw new Error(`Invalid runtime transition ${this.state} -> ${next}`);
    }
    const previous = this.state;
    this.state = next;
    this.trace?.emit("STATE_TRANSITION", { previous, next, reason });
    return this.state;
  }
}

export class BoundedTTLCache {
  constructor({ maxEntries = 128, ttlMs = 300_000 } = {}) {
    this.maxEntries = maxEntries;
    this.ttlMs = ttlMs;
    this.map = new Map();
    this.hits = 0;
    this.misses = 0;
    this.evictions = 0;
  }

  get(key, now = Date.now()) {
    const entry = this.map.get(key);
    if (!entry) {
      this.misses += 1;
      return null;
    }
    if (entry.expiresAt <= now) {
      this.map.delete(key);
      this.misses += 1;
      return null;
    }
    this.map.delete(key);
    this.map.set(key, entry);
    this.hits += 1;
    return entry.value;
  }

  set(key, value, ttlMs = this.ttlMs, now = Date.now()) {
    if (this.map.has(key)) this.map.delete(key);
    this.map.set(key, { value, expiresAt: now + ttlMs });
    while (this.map.size > this.maxEntries) {
      this.map.delete(this.map.keys().next().value);
      this.evictions += 1;
    }
  }

  clear() {
    this.map.clear();
  }

  stats() {
    const total = this.hits + this.misses;
    return Object.freeze({
      entries: this.map.size,
      hits: this.hits,
      misses: this.misses,
      evictions: this.evictions,
      hitRate: total ? this.hits / total : 0,
    });
  }
}

export class InstructionRegistry {
  constructor(trace) {
    this.trace = trace;
    this.instructions = new Map();
    this.order = [];
    this.version = 1;
    this.mutationCount = 0;
    this.manifestHash = stableHash([]);
  }

  register(definition) {
    const required = ["name", "match", "execute"];
    for (const field of required) {
      if (!(field in definition)) throw new TypeError(`Instruction missing ${field}`);
    }
    if (this.instructions.has(definition.name)) throw new Error(`Duplicate instruction ${definition.name}`);
    const instruction = {
      name: definition.name,
      match: definition.match,
      execute: definition.execute,
      priority: Number(definition.priority ?? 0),
      usage: 0,
      totalLatency: 0,
      core: Boolean(definition.core),
    };
    this.instructions.set(instruction.name, instruction);
    this.order.push(instruction.name);
    this._sortAndSeal();
    return this;
  }

  resolve(query, context) {
    for (const name of this.order) {
      const instruction = this.instructions.get(name);
      const match = instruction.match(query, context);
      if (match) return { instruction, match };
    }
    throw new Error("Instruction registry has no matching fallback");
  }

  execute(query, context) {
    const { instruction, match } = this.resolve(query, context);
    const start = performance.now();
    this.trace?.emit("INSTRUCTION_RESOLVED", { name: instruction.name, version: this.version });
    const result = instruction.execute(match, context);
    const latency = performance.now() - start;
    instruction.usage += 1;
    instruction.totalLatency += latency;
    this.trace?.emit("INSTRUCTION_EXECUTED", { name: instruction.name, latency, type: result.type });
    return { ...result, instruction: instruction.name, latency };
  }

  proposePriorityMutation() {
    const currentOrder = this.order.slice();
    const candidateOrder = currentOrder.slice().sort((leftName, rightName) => {
      const left = this.instructions.get(leftName);
      const right = this.instructions.get(rightName);
      const leftScore = left.usage / Math.max(1, left.totalLatency);
      const rightScore = right.usage / Math.max(1, right.totalLatency);
      if (left.name === "UNKNOWN") return 1;
      if (right.name === "UNKNOWN") return -1;
      return rightScore - leftScore || right.priority - left.priority || left.name.localeCompare(right.name);
    });
    const objective = (order) => order.reduce((sum, name, index) => {
      const instruction = this.instructions.get(name);
      return sum + instruction.usage * (index + 1);
    }, 0);
    return Object.freeze({
      currentOrder,
      candidateOrder,
      currentCost: objective(currentOrder),
      candidateCost: objective(candidateOrder),
    });
  }

  commitMutation(proposal) {
    if (!proposal || !Array.isArray(proposal.candidateOrder)) throw new TypeError("Invalid mutation proposal");
    const names = new Set(proposal.candidateOrder);
    if (names.size !== this.instructions.size || [...this.instructions.keys()].some((name) => !names.has(name))) {
      throw new Error("Mutation changes the instruction set membership");
    }
    const changed = proposal.candidateOrder.some((name, index) => name !== this.order[index]);
    if (proposal.candidateCost > proposal.currentCost || !changed) return false;
    this.order = proposal.candidateOrder.slice();
    this.version += 1;
    this.mutationCount += 1;
    this._sortAndSeal(false);
    this.trace?.emit("INSTRUCTION_MUTATION_COMMITTED", {
      version: this.version,
      currentCost: proposal.currentCost,
      candidateCost: proposal.candidateCost,
      manifestHash: this.manifestHash,
    });
    return true;
  }

  _sortAndSeal(sort = true) {
    if (sort) {
      this.order.sort((leftName, rightName) => {
        const left = this.instructions.get(leftName);
        const right = this.instructions.get(rightName);
        return right.priority - left.priority || left.name.localeCompare(right.name);
      });
    }
    this.manifestHash = stableHash({ version: this.version, order: this.order });
  }

  snapshot() {
    return Object.freeze({
      version: this.version,
      mutationCount: this.mutationCount,
      manifestHash: this.manifestHash,
      order: this.order.slice(),
      instructions: this.order.map((name) => {
        const instruction = this.instructions.get(name);
        return Object.freeze({
          name,
          usage: instruction.usage,
          averageLatency: instruction.usage ? instruction.totalLatency / instruction.usage : 0,
        });
      }),
    });
  }
}

function seededRandom(seed) {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(1664525, state) + 1013904223) >>> 0;
    return state / 0x100000000;
  };
}

export class TraceNeuralCore {
  constructor({ inputSize = 12, hiddenSize = 16, outputSize = 4, learningRate = 0.025, seed = 0x4a58524d } = {}) {
    this.inputSize = inputSize;
    this.hiddenSize = hiddenSize;
    this.outputSize = outputSize;
    this.learningRate = learningRate;
    this.steps = 0;
    this.loss = 0;
    this.lastHidden = new Array(hiddenSize).fill(0);
    const random = seededRandom(seed);
    const init = (rows, cols) => Array.from({ length: rows }, () =>
      Array.from({ length: cols }, () => (random() - 0.5) * 0.3));
    this.w1 = init(hiddenSize, inputSize);
    this.b1 = new Array(hiddenSize).fill(0);
    this.w2 = init(outputSize, hiddenSize);
    this.b2 = new Array(outputSize).fill(0);
  }

  forward(input) {
    if (!Array.isArray(input) || input.length !== this.inputSize) throw new TypeError(`Expected ${this.inputSize} inputs`);
    const hidden = this.w1.map((row, i) => Math.tanh(row.reduce((sum, weight, j) => sum + weight * input[j], this.b1[i])));
    const logits = this.w2.map((row, i) => row.reduce((sum, weight, j) => sum + weight * hidden[j], this.b2[i]));
    const max = Math.max(...logits);
    const exponentials = logits.map((value) => Math.exp(value - max));
    const denominator = exponentials.reduce((sum, value) => sum + value, 0);
    const probabilities = exponentials.map((value) => value / denominator);
    this.lastHidden = hidden;
    return { hidden, probabilities };
  }

  train(input, targetIndex) {
    if (!Number.isInteger(targetIndex) || targetIndex < 0 || targetIndex >= this.outputSize) {
      throw new RangeError("Invalid neural target index");
    }
    const { hidden, probabilities } = this.forward(input);
    const outputGradient = probabilities.slice();
    outputGradient[targetIndex] -= 1;
    const hiddenGradient = new Array(this.hiddenSize).fill(0);
    for (let output = 0; output < this.outputSize; output += 1) {
      for (let hiddenIndex = 0; hiddenIndex < this.hiddenSize; hiddenIndex += 1) {
        hiddenGradient[hiddenIndex] += this.w2[output][hiddenIndex] * outputGradient[output];
      }
    }
    for (let output = 0; output < this.outputSize; output += 1) {
      for (let hiddenIndex = 0; hiddenIndex < this.hiddenSize; hiddenIndex += 1) {
        this.w2[output][hiddenIndex] -= this.learningRate * outputGradient[output] * hidden[hiddenIndex];
      }
      this.b2[output] -= this.learningRate * outputGradient[output];
    }
    for (let hiddenIndex = 0; hiddenIndex < this.hiddenSize; hiddenIndex += 1) {
      const localGradient = hiddenGradient[hiddenIndex] * (1 - hidden[hiddenIndex] ** 2);
      for (let inputIndex = 0; inputIndex < this.inputSize; inputIndex += 1) {
        this.w1[hiddenIndex][inputIndex] -= this.learningRate * localGradient * input[inputIndex];
      }
      this.b1[hiddenIndex] -= this.learningRate * localGradient;
    }
    this.steps += 1;
    this.loss = -Math.log(Math.max(probabilities[targetIndex], 1e-9));
    return Object.freeze({ loss: this.loss, probabilities, hidden });
  }

  snapshot() {
    const activation = this.lastHidden.reduce((sum, value) => sum + Math.abs(value), 0) / this.hiddenSize;
    return Object.freeze({ steps: this.steps, loss: this.loss, activation, hidden: this.lastHidden.slice() });
  }
}

function normalizeQuery(text) {
  return String(text).toLowerCase().trim().replace(/\s+/g, " ");
}

function formatUTCOffset(minutes) {
  const sign = minutes >= 0 ? "+" : "−";
  const absolute = Math.abs(minutes);
  const hours = Math.floor(absolute / 60);
  const remainder = absolute % 60;
  return `UTC${sign}${String(hours).padStart(2, "0")}:${String(remainder).padStart(2, "0")}`;
}

function comparePopulation(population, operator, threshold) {
  if (operator === "<") return population < threshold;
  if (operator === "<=") return population <= threshold;
  if (operator === ">=") return population >= threshold;
  if (operator === "=") return population === threshold;
  return population > threshold;
}

function resultClass(type) {
  return type === "fly" ? 0 : type === "time" ? 1 : type === "list" ? 2 : 3;
}

function buildNeuralInput({ query, cacheHit, result, latency, runtime }) {
  const instructionNames = runtime.registry.snapshot().order;
  const instructionIndex = Math.max(0, instructionNames.indexOf(result.instruction));
  const resultCount = result.cities?.length ?? (result.city ? 1 : 0);
  return [
    Math.min(1, query.length / 80),
    cacheHit ? 1 : 0,
    Math.min(1, instructionIndex / Math.max(1, instructionNames.length - 1)),
    Math.min(1, resultCount / 16),
    Math.min(1, latency / 10),
    runtime.fsm.state === RUNTIME_STATES.PROCESSING ? 1 : 0,
    runtime.cache.stats().hitRate,
    Math.min(1, runtime.queries / 100),
    Math.min(1, runtime.registry.mutationCount / 20),
    Math.min(1, runtime.romRefinements / 20),
    result.city ? Math.min(1, result.city.pop / 30_000_000) : 0,
    result.type === "unknown" ? 1 : 0,
  ];
}

export class JarvisXRuntime {
  constructor(cities, options = {}) {
    this.trace = options.trace ?? new TraceBus();
    this.fsm = new RuntimeFSM(this.trace);
    this.cache = new BoundedTTLCache(options.cache);
    this.neural = new TraceNeuralCore(options.neural);
    this.registry = new InstructionRegistry(this.trace);
    this.aliases = new Map();
    this.queryCounts = new Map();
    this.queries = 0;
    this.unknowns = 0;
    this.romRefinements = 0;
    this.romBuffer = encodeROM(cities);
    this.rom = decodeROM(this.romBuffer);
    this.cities = [...this.rom.cities];
    this._installInstructions();
    this.trace.emit("RUNTIME_BOOT", { cities: this.cities.length, romBytes: this.romBuffer.byteLength });
  }

  _findCity(query) {
    let aliasTarget = this.aliases.get(query);
    if (!aliasTarget) {
      for (const [alias, target] of this.aliases) {
        if (query.includes(alias)) { aliasTarget = target; break; }
      }
    }
    if (aliasTarget) return this.cities.find((city) => city.name.toLowerCase() === aliasTarget) ?? null;
    const exact = this.cities.find((city) => city.name.toLowerCase() === query);
    if (exact) return exact;
    return this.cities.find((city) => query.includes(city.name.toLowerCase())) ?? null;
  }

  _recordUsage(city) {
    if (!city) return;
    const count = (this.queryCounts.get(city.name) ?? 0) + 1;
    this.queryCounts.set(city.name, count);
  }

  _installInstructions() {
    this.registry
      .register({
        name: "TEACH_ALIAS",
        priority: 100,
        core: true,
        match: (query) => query.match(/^teach\s+(.+?)\s*=\s*(.+)$/i),
        execute: (match) => {
          const alias = normalizeQuery(match[1]);
          const targetQuery = normalizeQuery(match[2]);
          const city = this._findCity(targetQuery);
          if (!city) return { type: "unknown", msg: `Cannot teach alias “${alias}”: target city was not found.` };
          if (!alias || alias.length > 40) return { type: "unknown", msg: "Alias must contain 1–40 characters." };
          this.aliases.set(alias, city.name.toLowerCase());
          this.trace.emit("ALIAS_LEARNED", { alias, city: city.name });
          return { type: "text", city, msg: `Learned alias “${alias}” → ${city.name}.` };
        },
      })
      .register({
        name: "RUNTIME_STATUS",
        priority: 95,
        core: true,
        match: (query) => /^(status|runtime status|neural status|system status)$/.test(query),
        execute: () => {
          const snapshot = this.snapshot();
          return {
            type: "text",
            msg: `Runtime v${snapshot.instructions.version}: ${snapshot.queries} queries, ${(snapshot.cache.hitRate * 100).toFixed(0)}% cache hit, neural loss ${snapshot.neural.loss.toFixed(3)}, ${snapshot.romRefinements} ROM refinements.`,
          };
        },
      })
      .register({
        name: "LOCAL_TIME",
        priority: 80,
        match: (query) => (query.includes("time") || query.includes("clock") || query.includes("hour")) ? { query } : null,
        execute: ({ query }) => {
          const city = this._findCity(query);
          if (!city) return { type: "unknown", msg: "Specify an encoded city, for example: Time in London." };
          const localTime = new Date(Date.now() + city.tz * 60_000);
          const hours = String(localTime.getUTCHours()).padStart(2, "0");
          const minutes = String(localTime.getUTCMinutes()).padStart(2, "0");
          return { type: "time", city, msg: `Local time in ${city.name}: ${hours}:${minutes}`, metaTag: formatUTCOffset(city.tz) };
        },
      })
      .register({
        name: "POPULATION_FILTER",
        priority: 70,
        match: (query) => query.match(/(<=|>=|=|<|>)?\s*(\d+(?:\.\d+)?)\s*(m|million|k|thousand)\b/i),
        execute: (match) => {
          const operator = match[1] || ">";
          let threshold = Number.parseFloat(match[2]);
          const unit = match[3].toLowerCase();
          threshold *= unit === "m" || unit === "million" ? 1_000_000 : 1_000;
          const cities = this.cities.filter((city) => comparePopulation(city.pop, operator, threshold));
          return {
            type: "list",
            cities,
            msg: cities.length
              ? `Cities ${operator} ${(threshold / 1_000_000).toFixed(1)}M: ${cities.map((city) => city.name).join(", ")}.`
              : `No encoded cities satisfy ${operator} ${(threshold / 1_000_000).toFixed(1)}M.`,
          };
        },
      })
      .register({
        name: "CITY_LOOKUP",
        priority: 60,
        match: (query) => {
          const city = this._findCity(query);
          return city ? { city } : null;
        },
        execute: ({ city }) => ({
          type: "fly",
          city,
          msg: `${city.name}, ${city.country}. Population ${(city.pop / 1_000_000).toFixed(1)}M, elevation ${city.elev}m, ${formatUTCOffset(city.tz)}.`,
        }),
      })
      .register({
        name: "UNKNOWN",
        priority: -1000,
        core: true,
        match: () => ({}),
        execute: () => {
          this.unknowns += 1;
          return { type: "unknown", msg: "No matching instruction or ROM record. Teach an alias with: teach jozi = Johannesburg." };
        },
      });
  }

  refine() {
    const before = this._weightedSearchCost(this.cities);
    const candidate = this.cities.slice().sort((left, right) => {
      const delta = (this.queryCounts.get(right.name) ?? 0) - (this.queryCounts.get(left.name) ?? 0);
      return delta || left.name.localeCompare(right.name);
    });
    const after = this._weightedSearchCost(candidate);
    const instructionProposal = this.registry.proposePriorityMutation();
    const instructionCommitted = this.registry.commitMutation(instructionProposal);
    let romCommitted = false;
    if (after <= before) {
      const candidateBuffer = encodeROM(candidate);
      const decoded = decodeROM(candidateBuffer);
      this.romBuffer = candidateBuffer;
      this.rom = decoded;
      this.cities = [...decoded.cities];
      this.romRefinements += 1;
      this.cache.clear();
      romCommitted = true;
      this.trace.emit("ROM_REFINEMENT_COMMITTED", { before, after, checksum: decoded.checksum });
    }
    return Object.freeze({ romCommitted, instructionCommitted, before, after, instructionProposal });
  }

  _weightedSearchCost(order) {
    return order.reduce((sum, city, index) => sum + (this.queryCounts.get(city.name) ?? 0) * (index + 1), 0);
  }

  async query(text) {
    const query = normalizeQuery(text);
    if (!query) throw new TypeError("Query must not be empty");
    if (this.fsm.state === RUNTIME_STATES.ERROR) this.fsm.transition(RUNTIME_STATES.IDLE, "recovery");
    if (this.fsm.state === RUNTIME_STATES.ECHOING) this.fsm.transition(RUNTIME_STATES.PROCESSING, "new query");
    else this.fsm.transition(RUNTIME_STATES.PROCESSING, "query received");
    this.queries += 1;
    const started = performance.now();
    this.trace.emit("QUERY_RECEIVED", { queryLength: query.length });

    try {
      const cached = this.cache.get(query);
      if (cached) {
        this._recordUsage(cached.city);
        this.trace.emit("CACHE_HIT", { query, instruction: cached.instruction });
        const latency = performance.now() - started;
        const input = buildNeuralInput({ query, cacheHit: true, result: cached, latency, runtime: this });
        const neural = this.neural.train(input, resultClass(cached.type));
        this.fsm.transition(RUNTIME_STATES.ECHOING, "cached result");
        return Object.freeze({ ...cached, cacheHit: true, neural, latency });
      }

      this.trace.emit("CACHE_MISS", { query });
      const result = this.registry.execute(query, this);
      this._recordUsage(result.city);
      const ttl = result.type === "time" ? 10_000 : 300_000;
      this.cache.set(query, result, ttl);
      const latency = performance.now() - started;
      const input = buildNeuralInput({ query, cacheHit: false, result, latency, runtime: this });
      const neural = this.neural.train(input, resultClass(result.type));
      this.trace.emit("NEURAL_BACKPROP", { loss: neural.loss, activation: neural.hidden.reduce((sum, value) => sum + Math.abs(value), 0) / neural.hidden.length });
      this.fsm.transition(RUNTIME_STATES.ECHOING, "result ready");
      return Object.freeze({ ...result, cacheHit: false, neural, latency });
    } catch (error) {
      this.fsm.transition(RUNTIME_STATES.ERROR, error.message);
      this.trace.emit("QUERY_ERROR", { message: error.message });
      throw error;
    }
  }

  settle() {
    if (this.fsm.state === RUNTIME_STATES.ECHOING) this.fsm.transition(RUNTIME_STATES.IDLE, "echo complete");
  }

  hotCities(limit = 5) {
    return this.cities
      .slice()
      .sort((left, right) => (this.queryCounts.get(right.name) ?? 0) - (this.queryCounts.get(left.name) ?? 0) || left.name.localeCompare(right.name))
      .filter((city) => (this.queryCounts.get(city.name) ?? 0) > 0)
      .slice(0, limit)
      .map((city) => city.name);
  }

  snapshot() {
    return Object.freeze({
      state: this.fsm.state,
      queries: this.queries,
      unknowns: this.unknowns,
      aliases: this.aliases.size,
      romBytes: this.romBuffer.byteLength,
      romChecksum: this.rom.checksum.toString(16).padStart(8, "0"),
      romRefinements: this.romRefinements,
      cache: this.cache.stats(),
      instructions: this.registry.snapshot(),
      neural: this.neural.snapshot(),
      hotCities: this.hotCities(),
      traceEvents: this.trace.snapshot().length,
    });
  }
}
