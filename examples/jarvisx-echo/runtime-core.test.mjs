import assert from "node:assert/strict";
import test from "node:test";
import {
  BoundedTTLCache,
  InstructionRegistry,
  JarvisXRuntime,
  RUNTIME_STATES,
  RuntimeFSM,
  TraceBus,
  TraceNeuralCore,
  decodeROM,
  encodeROM,
} from "./runtime-core.mjs";

const CITIES = [
  { name: "Tokyo", country: "Japan", lat: 35.7, lon: 139.7, pop: 13_960_000, elev: 40, tz: 540 },
  { name: "London", country: "UK", lat: 51.5, lon: -0.1, pop: 8_982_000, elev: 11, tz: 0 },
  { name: "Johannesburg", country: "South Africa", lat: -26.2, lon: 28.0, pop: 5_635_000, elev: 1753, tz: 120 },
];

test("ROM round-trips and detects corruption", () => {
  const buffer = encodeROM(CITIES);
  const decoded = decodeROM(buffer);
  assert.equal(decoded.version, 3);
  assert.deepEqual(decoded.cities.map((city) => city.name), CITIES.map((city) => city.name));
  const corrupted = buffer.slice(0);
  new Uint8Array(corrupted)[corrupted.byteLength - 1] ^= 1;
  assert.throws(() => decodeROM(corrupted), /checksum mismatch/);
});

test("bounded TTL cache evicts least recently used entry", () => {
  const cache = new BoundedTTLCache({ maxEntries: 2, ttlMs: 100 });
  cache.set("a", 1, 100, 0);
  cache.set("b", 2, 100, 0);
  assert.equal(cache.get("a", 1), 1);
  cache.set("c", 3, 100, 1);
  assert.equal(cache.get("b", 2), null);
  assert.equal(cache.get("a", 2), 1);
  assert.equal(cache.stats().evictions, 1);
});

test("finite-state machine rejects illegal transitions", () => {
  const fsm = new RuntimeFSM(new TraceBus());
  assert.throws(() => fsm.transition(RUNTIME_STATES.ECHOING), /Invalid runtime transition/);
  fsm.transition(RUNTIME_STATES.PROCESSING);
  fsm.transition(RUNTIME_STATES.ECHOING);
  fsm.transition(RUNTIME_STATES.IDLE);
  assert.equal(fsm.state, RUNTIME_STATES.IDLE);
});

test("instruction mutation is versioned and cannot change membership", () => {
  const registry = new InstructionRegistry(new TraceBus());
  registry.register({ name: "A", priority: 1, match: () => true, execute: () => ({ type: "text" }) });
  registry.register({ name: "B", priority: 2, match: () => false, execute: () => ({ type: "text" }) });
  registry.execute("x", {});
  const proposal = registry.proposePriorityMutation();
  assert.equal(registry.commitMutation(proposal), true);
  assert.equal(registry.snapshot().version, 2);
  assert.throws(() => registry.commitMutation({ ...proposal, candidateOrder: ["A"] }), /membership/);
});

test("neural core performs real online backpropagation", () => {
  const core = new TraceNeuralCore({ seed: 7 });
  const input = new Array(12).fill(0);
  input[0] = 1;
  const before = core.forward(input).probabilities[2];
  for (let index = 0; index < 50; index += 1) core.train(input, 2);
  const after = core.forward(input).probabilities[2];
  assert.ok(after > before);
  assert.equal(core.snapshot().steps, 50);
  assert.ok(Number.isFinite(core.snapshot().loss));
});

test("runtime executes, caches, learns aliases, refines, and settles", async () => {
  const runtime = new JarvisXRuntime(CITIES, { cache: { maxEntries: 8, ttlMs: 1000 } });
  const first = await runtime.query("Where is Tokyo?");
  assert.equal(first.type, "fly");
  assert.equal(first.cacheHit, false);
  assert.equal(runtime.snapshot().state, RUNTIME_STATES.ECHOING);
  runtime.settle();
  assert.equal(runtime.snapshot().state, RUNTIME_STATES.IDLE);

  const second = await runtime.query("Where is Tokyo?");
  assert.equal(second.cacheHit, true);
  runtime.settle();

  const taught = await runtime.query("teach jozi = Johannesburg");
  assert.match(taught.msg, /Learned alias/);
  runtime.settle();

  const alias = await runtime.query("jozi");
  assert.equal(alias.city.name, "Johannesburg");
  runtime.settle();

  const filtered = await runtime.query("cities >= 8.9m");
  assert.equal(filtered.type, "list");
  assert.deepEqual(filtered.cities.map((city) => city.name).sort(), ["London", "Tokyo"]);
  runtime.settle();

  const refinement = runtime.refine();
  assert.equal(refinement.romCommitted, true);
  const snapshot = runtime.snapshot();
  assert.ok(snapshot.neural.steps >= 5);
  assert.ok(snapshot.instructions.version >= 2);
  assert.ok(snapshot.traceEvents > 0);
  assert.equal(snapshot.cache.entries, 0);
});
