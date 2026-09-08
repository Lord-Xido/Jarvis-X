import assert from 'node:assert/strict';
import {
  SIDE,
  CAPACITY_BI,
  virtualAddress,
  addressToXYZ,
  makeVirtualSample,
  encodeSwarm,
  decodeSwarm,
  reconstructionMSE,
  optimize3D,
} from './core.mjs';

assert.equal(SIDE, 1_000_000);
assert.equal(CAPACITY_BI, 1_000_000_000_000_000_000n);

for (const p of [
  {x:0,y:0,z:0},
  {x:3,y:2,z:1},
  {x:999_999,y:999_999,z:999_999},
]) {
  const n = virtualAddress(p.x,p.y,p.z);
  assert.deepEqual(addressToXYZ(n), p);
}

const a = makeVirtualSample(128, 1234);
const b = makeVirtualSample(128, 1234);
assert.deepEqual(a, b, 'sampling must be deterministic for the same seed');

const cfg = {alpha:0.28,bits:8,omega:0.18};
const encoded = encodeSwarm(a, cfg);
const decoded = decodeSwarm(encoded.latent, encoded.centroid, cfg);
const mse = reconstructionMSE(a, decoded);
assert.ok(Number.isFinite(mse));
assert.ok(mse >= 0);

const result = optimize3D(a, cfg);
assert.equal(result.tested, 125);
assert.equal(result.direction.length, 3);
assert.ok(result.config.alpha >= 0.02 && result.config.alpha <= 0.80);
assert.ok(result.config.bits >= 3 && result.config.bits <= 16);
assert.ok(result.config.omega >= 0 && result.config.omega <= 0.60);
assert.ok(Number.isFinite(result.metrics.score));

assert.throws(() => virtualAddress(-1,0,0), RangeError);
assert.throws(() => virtualAddress(1_000_000,0,0), RangeError);
assert.throws(() => addressToXYZ(CAPACITY_BI), RangeError);

console.log('dr-moagi-1e18-swarm core tests passed');
