import test from "node:test";
import assert from "node:assert/strict";

import {
  CognitiveMatrixEngine,
  LOGICAL_NODE_COUNT,
  OPCODES,
  SparseRingMemory,
  decodeInstruction,
  encodeInstruction,
} from "./engine.mjs";

test("32-bit instruction encoding is deterministic", () => {
  const word = encodeInstruction(OPCODES.INWARD_FOLD, 0x123456);
  assert.equal(word >>> 0, 0x50123456);
  assert.deepEqual(decodeInstruction(word), {
    opcode: OPCODES.INWARD_FOLD,
    operand: 0x123456,
  });
});

test("sparse virtual memory allocates only touched pages", () => {
  const memory = new SparseRingMemory({ maxResidentPages: 4 });
  memory.write(new Uint8Array([1, 2, 3, 4]));
  const stats = memory.stats();
  assert.equal(stats.virtualBytes, 1024 * 1024 * 1024);
  assert.equal(stats.residentPages, 1);
  assert.equal(stats.residentBytes, 64 * 1024);
  assert.deepEqual([...memory.read(0, 4)], [1, 2, 3, 4]);
});

test("resident page budget fails closed", () => {
  const memory = new SparseRingMemory({
    capacity: 1024,
    pageSize: 8,
    maxResidentPages: 1,
  });
  memory.write(new Uint8Array(8));
  assert.throws(() => memory.write(new Uint8Array([1])), /budget exceeded/);
});

test("cognitive matrix uses bounded resident active set", () => {
  const engine = new CognitiveMatrixEngine({ activeNodes: 64, latentDim: 4 });
  assert.equal(engine.logicalNodes, LOGICAL_NODE_COUNT);
  assert.equal(engine.activeNodes, 64);
  assert.equal(engine.committed.length, 64 * 4);
});

test("macrocycle executes VM trace and returns finite verification evidence", () => {
  const engine = new CognitiveMatrixEngine({
    activeNodes: 96,
    latentDim: 4,
    fixedPointIterations: 5,
    ctrThreshold: 2,
  });
  const result = engine.runMacrocycle(0.25);
  assert.equal(result.trace[0], "LOAD_BITSTREAM");
  assert.equal(result.trace.at(-1), "HALT_SYNC");
  assert.ok(Number.isFinite(result.fixedPointResidual));
  assert.ok(Number.isFinite(result.ctrEnergy));
  assert.ok(Number.isFinite(result.macrocycleMs));
  assert.ok(result.residentBytes > 0);
});

test("failed CTR verification rolls candidate back", () => {
  const engine = new CognitiveMatrixEngine({
    activeNodes: 64,
    latentDim: 4,
    ctrThreshold: 1e-14,
  });
  const before = [...engine.committed];
  const result = engine.runMacrocycle(1);
  assert.equal(result.accepted, false);
  assert.deepEqual([...engine.candidate], [...engine.committed]);
  assert.deepEqual([...engine.committed], before);
});
