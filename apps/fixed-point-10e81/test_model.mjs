import test from "node:test";
import assert from "node:assert/strict";
import {
  AXIS_EXTENT,
  LOGICAL_CARDINALITY,
  BITS_PER_AXIS,
  LOGICAL_ADDRESS_BITS,
  FixedPoint10E81,
  packCoordinate,
  unpackCoordinate,
} from "./core.mjs";

test("logical geometry is exact", () => {
  assert.equal(AXIS_EXTENT, 10n ** 27n);
  assert.equal(LOGICAL_CARDINALITY, 10n ** 81n);
  assert.equal(BITS_PER_AXIS, 90);
  assert.equal(LOGICAL_ADDRESS_BITS, 270);
  assert.ok((1n << 89n) < AXIS_EXTENT);
  assert.ok(AXIS_EXTENT < (1n << 90n));
});

test("270-bit logical address round trips", () => {
  const probes = [
    [0n, 0n, 0n],
    [AXIS_EXTENT - 1n, 0n, 0n],
    [0n, AXIS_EXTENT - 1n, 0n],
    [0n, 0n, AXIS_EXTENT - 1n],
    [AXIS_EXTENT - 1n, AXIS_EXTENT - 1n, AXIS_EXTENT - 1n],
    [123456789012345678901234567n, 1n, 999999999999999999999999999n],
  ];
  for (const [x,y,z] of probes) {
    const a = packCoordinate(x,y,z);
    const p = unpackCoordinate(a);
    assert.deepEqual([p.x,p.y,p.z],[x,y,z]);
  }
  assert.equal(
    packCoordinate(AXIS_EXTENT-1n,AXIS_EXTENT-1n,AXIS_EXTENT-1n),
    LOGICAL_CARDINALITY-1n
  );
});

test("active support stays bounded and deterministic", () => {
  const a = new FixedPoint10E81({activeSupport:64, seed:7n});
  const b = new FixedPoint10E81({activeSupport:64, seed:7n});
  assert.equal(a.coordinates.length,64);
  assert.deepEqual(a.coordinates,b.coordinates);
  for (const c of a.coordinates) {
    for (const axis of c) assert.ok(axis >= 0n && axis < AXIS_EXTENT);
  }
});

test("accepted recurrent state never increases MSE", () => {
  const engine = new FixedPoint10E81({activeSupport:256, seed:9n});
  let acceptedLoss = engine.loss();
  for (let i=0;i<120;i++) {
    const receipt = engine.step();
    assert.ok(Number.isFinite(receipt.mse));
    assert.ok(Number.isFinite(receipt.residualL2));
    assert.ok(engine.loss() <= acceptedLoss + 1e-14);
    acceptedLoss = engine.loss();
  }
  assert.ok(engine.commits > 0);
  assert.ok(engine.loss() < 1e-4);
});

test("verification gate can roll back a destabilizing proposal", () => {
  const engine = new FixedPoint10E81({activeSupport:128, seed:13n, eta:0.24});
  for (let i=0;i<8;i++) engine.step();
  const before = engine.loss();
  engine.eta = 100;
  const receipt = engine.step();
  assert.equal(receipt.accepted,false);
  assert.equal(engine.lastDecision,"ROLLBACK");
  assert.ok(engine.loss() <= before + 1e-14);
  assert.ok(engine.rollbacks >= 1);
});

test("remap preserves virtual geometry but resets recurrent authority state", () => {
  const engine = new FixedPoint10E81({activeSupport:32, seed:21n});
  engine.step();
  const before = engine.coordinates.map(c => c.join(":"));
  engine.remap();
  const after = engine.coordinates.map(c => c.join(":"));
  assert.notDeepEqual(after,before);
  assert.equal(engine.iteration,0);
  assert.equal(engine.commits,0);
  assert.equal(engine.rollbacks,0);
});
