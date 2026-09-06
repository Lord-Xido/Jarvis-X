export const SIDE = 1_000_000;
export const SIDE_BI = 1_000_000n;
export const CAPACITY_BI = SIDE_BI ** 3n;

export function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

export function virtualAddress(x, y, z) {
  for (const [name, v] of [["x", x], ["y", y], ["z", z]]) {
    if (!Number.isInteger(v) || v < 0 || v >= SIDE) {
      throw new RangeError(`${name} must be an integer in [0, ${SIDE - 1}]`);
    }
  }
  return BigInt(x) + SIDE_BI * (BigInt(y) + SIDE_BI * BigInt(z));
}

export function addressToXYZ(n) {
  const a = typeof n === "bigint" ? n : BigInt(n);
  if (a < 0n || a >= CAPACITY_BI) throw new RangeError("address outside virtual lattice");
  const x = Number(a % SIDE_BI);
  const yz = a / SIDE_BI;
  const y = Number(yz % SIDE_BI);
  const z = Number(yz / SIDE_BI);
  return { x, y, z };
}

export function xorshift32(seed = 0x5eed1234) {
  let state = seed >>> 0;
  return () => {
    state ^= state << 13;
    state ^= state >>> 17;
    state ^= state << 5;
    return (state >>> 0) / 4294967296;
  };
}

export function makeVirtualSample(count, seed = 0x5eed1234) {
  if (!Number.isInteger(count) || count < 1 || count > 100_000) {
    throw new RangeError("count must be an integer in [1, 100000]");
  }
  const rnd = xorshift32(seed);
  const points = new Array(count);
  for (let i = 0; i < count; i++) {
    const vx = Math.floor(rnd() * SIDE);
    const vy = Math.floor(rnd() * SIDE);
    const vz = Math.floor(rnd() * SIDE);
    points[i] = {
      x: (vx / (SIDE - 1)) * 2 - 1,
      y: (vy / (SIDE - 1)) * 2 - 1,
      z: (vz / (SIDE - 1)) * 2 - 1,
      vx,
      vy,
      vz,
      address: virtualAddress(vx, vy, vz),
    };
  }
  return points;
}

export function rotateForward(p) {
  const a = 0.47, b = -0.31, c = 0.22;
  let { x, y, z } = p;
  let co = Math.cos(a), si = Math.sin(a);
  [y, z] = [y * co - z * si, y * si + z * co];
  co = Math.cos(b); si = Math.sin(b);
  [x, z] = [x * co + z * si, -x * si + z * co];
  co = Math.cos(c); si = Math.sin(c);
  [x, y] = [x * co - y * si, x * si + y * co];
  return { x, y, z };
}

export function rotateInverse(p) {
  const a = 0.47, b = -0.31, c = 0.22;
  let { x, y, z } = p;
  let co = Math.cos(-c), si = Math.sin(-c);
  [x, y] = [x * co - y * si, x * si + y * co];
  co = Math.cos(-b); si = Math.sin(-b);
  [x, z] = [x * co + z * si, -x * si + z * co];
  co = Math.cos(-a); si = Math.sin(-a);
  [y, z] = [y * co - z * si, y * si + z * co];
  return { x, y, z };
}

export function rotatedCentroid(points) {
  if (!points.length) return { x: 0, y: 0, z: 0 };
  let x = 0, y = 0, z = 0;
  for (const p of points) {
    const r = rotateForward(p);
    x += r.x; y += r.y; z += r.z;
  }
  return { x: x / points.length, y: y / points.length, z: z / points.length };
}

export function encodePoint(p, { alpha = 0.28, bits = 8, omega = 0.18, centroid = {x:0,y:0,z:0} } = {}) {
  alpha = clamp(alpha, 0.001, 0.95);
  bits = Math.round(clamp(bits, 3, 16));
  omega = clamp(omega, 0, 0.60);
  const r0 = rotateForward(p);
  let x = (1 - omega) * r0.x + omega * centroid.x;
  let y = (1 - omega) * r0.y + omega * centroid.y;
  let z = (1 - omega) * r0.z + omega * centroid.z;
  const radius = Math.hypot(x, y, z) || 1e-12;
  const foldedRadius = radius / (1 + alpha * radius);
  const scale = foldedRadius / radius;
  x *= scale; y *= scale; z *= scale;

  const maxRadius = Math.sqrt(3) / (1 + alpha * Math.sqrt(3));
  const levels = 2 ** bits - 1;
  const quantize = (v) => {
    const u = clamp((v / maxRadius + 1) / 2, 0, 1);
    return ((Math.round(u * levels) / levels) * 2 - 1) * maxRadius;
  };
  return { x: quantize(x), y: quantize(y), z: quantize(z) };
}

export function decodePoint(q, { alpha = 0.28, omega = 0.18, centroid = {x:0,y:0,z:0} } = {}) {
  alpha = clamp(alpha, 0.001, 0.95);
  omega = clamp(omega, 0, 0.60);
  let { x, y, z } = q;
  const rf = Math.hypot(x, y, z) || 1e-12;
  const denominator = Math.max(1e-9, 1 - alpha * rf);
  const radius = rf / denominator;
  const scale = radius / rf;
  x *= scale; y *= scale; z *= scale;
  const inverseMemory = Math.max(1e-6, 1 - omega);
  x = (x - omega * centroid.x) / inverseMemory;
  y = (y - omega * centroid.y) / inverseMemory;
  z = (z - omega * centroid.z) / inverseMemory;
  return rotateInverse({ x, y, z });
}

export function encodeSwarm(points, config = {}) {
  const centroid = config.centroid ?? rotatedCentroid(points);
  return { centroid, latent: points.map((p) => encodePoint(p, { ...config, centroid })) };
}

export function decodeSwarm(latent, centroid, config = {}) {
  return latent.map((p) => decodePoint(p, { ...config, centroid }));
}

export function reconstructionMSE(source, reconstructed) {
  if (!source.length || source.length !== reconstructed.length) throw new RangeError("source/reconstruction mismatch");
  let sum = 0;
  for (let i = 0; i < source.length; i++) {
    const dx = reconstructed[i].x - source[i].x;
    const dy = reconstructed[i].y - source[i].y;
    const dz = reconstructed[i].z - source[i].z;
    sum += dx * dx + dy * dy + dz * dz;
  }
  return sum / (3 * source.length);
}

export function evaluateConfig(source, { alpha, bits, omega, evalCount = 700 }) {
  const subset = source.slice(0, Math.min(evalCount, source.length));
  const { centroid, latent } = encodeSwarm(subset, { alpha, bits, omega });
  const reconstructed = decodeSwarm(latent, centroid, { alpha, omega });
  const mse = reconstructionMSE(subset, reconstructed);
  const payloadPenalty = 0.00012 * ((3 * bits) / 96);
  const foldPenalty = 0.002 * alpha * alpha;
  const memoryPenalty = 0.0015 * omega * omega;
  return { mse, score: mse + payloadPenalty + foldPenalty + memoryPenalty };
}

export function optimize3D(source, config = { alpha: 0.28, bits: 8, omega: 0.18 }) {
  const aSteps = [-0.08, -0.04, 0, 0.04, 0.08];
  const bSteps = [-2, -1, 0, 1, 2];
  const oSteps = [-0.08, -0.04, 0, 0.04, 0.08];
  let best = null;
  let tested = 0;
  for (let i = 0; i < 5; i++) {
    for (let j = 0; j < 5; j++) {
      for (let k = 0; k < 5; k++) {
        const candidate = {
          alpha: clamp(config.alpha + aSteps[i], 0.02, 0.80),
          bits: Math.round(clamp(config.bits + bSteps[j], 3, 16)),
          omega: clamp(config.omega + oSteps[k], 0, 0.60),
        };
        const metrics = evaluateConfig(source, candidate);
        tested++;
        if (!best || metrics.score < best.metrics.score) {
          best = { config: candidate, metrics, direction: [i - 2, j - 2, k - 2] };
        }
      }
    }
  }
  return { ...best, tested };
}
