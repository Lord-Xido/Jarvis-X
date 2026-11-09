// Explicit CPU reference benchmark; never presented as GPU throughput.
import { performance } from 'node:perf_hooks';
import { encodeRGBA, decodeRGBA, equalBytes, medianMs, throughputMetrics } from './codec.mjs';

const size = Number(process.argv[2] ?? 1048576);
const repeats = Number(process.argv[3] ?? 20);
if (!Number.isSafeInteger(size) || size < 1 || size > 16 * 1048576 ||
    !Number.isSafeInteger(repeats) || repeats < 2 || repeats > 1000) {
  throw new RangeError('usage: node benchmark_cpu.mjs [bytes 1..16777216] [repeats 2..1000]');
}
const source = Uint8Array.from({length: size}, (_, i) => (i * 17 + Math.floor(i / 251)) & 255);
const times = [];
for (let i = 0; i < repeats + 3; i++) {
  const start = performance.now();
  const {data,width,height} = encodeRGBA(source);
  const decoded = decodeRGBA(data,width,height,size);
  if (!equalBytes(source,decoded)) throw new Error('CPU reference round-trip failed');
  const elapsed = performance.now() - start;
  if (i >= 3) times.push(elapsed);
}
const median = medianMs(times);
console.log(JSON.stringify({backend:'CPU reference (not WebGL/GPU)', samples:repeats,
  ...throughputMetrics(size,256,1,median)}, null, 2));
