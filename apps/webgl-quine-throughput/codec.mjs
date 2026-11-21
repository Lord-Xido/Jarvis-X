// Exact byte-layout reference for the WebGL round-trip laboratory.
// No compression, learned inference, or source execution is performed here.
export function encodeRGBA(bytes, width = 256) {
  if (!(bytes instanceof Uint8Array)) throw new TypeError('bytes must be Uint8Array');
  if (!Number.isSafeInteger(width) || width < 1 || width > 16384) {
    throw new RangeError('invalid texture width');
  }
  const rowBytes = width * 4;
  const height = Math.max(1, Math.ceil(bytes.length / rowBytes));
  const allocatedBytes = rowBytes * height;
  if (!Number.isSafeInteger(allocatedBytes)) throw new RangeError('texture too large');
  const data = new Uint8Array(allocatedBytes);
  data.set(bytes);
  return { data, width, height, byteLength: bytes.length };
}

export function decodeRGBA(data, width, height, byteLength, flipY = false) {
  if (!(data instanceof Uint8Array)) throw new TypeError('data must be Uint8Array');
  if (![width, height, byteLength].every(Number.isSafeInteger) || width < 1 || height < 1 || byteLength < 0) {
    throw new RangeError('invalid decode dimensions');
  }
  const rowBytes = width * 4;
  if (data.length !== rowBytes * height || byteLength > data.length) {
    throw new RangeError('RGBA data size mismatch');
  }
  const output = new Uint8Array(byteLength);
  for (let y = 0; y < height; y++) {
    const dst = y * rowBytes;
    if (dst >= byteLength) break;
    const src = (flipY ? height - y - 1 : y) * rowBytes;
    output.set(data.subarray(src, src + Math.min(rowBytes, byteLength - dst)), dst);
  }
  return output;
}

export function equalBytes(a, b) {
  if (!(a instanceof Uint8Array) || !(b instanceof Uint8Array) || a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

export function bitErrorCount(a, b) {
  if (!(a instanceof Uint8Array) || !(b instanceof Uint8Array) || a.length !== b.length) {
    throw new RangeError('comparison requires equal-sized byte arrays');
  }
  let bits = 0;
  for (let i = 0; i < a.length; i++) {
    let xor = a[i] ^ b[i];
    while (xor) { bits += xor & 1; xor >>>= 1; }
  }
  return bits;
}

export function medianMs(samples) {
  if (!Array.isArray(samples) || !samples.length || samples.some(x => !Number.isFinite(x) || x <= 0)) {
    throw new RangeError('expected positive timing samples');
  }
  const ordered = [...samples].sort((a, b) => a - b);
  const middle = Math.floor(ordered.length / 2);
  return ordered.length % 2 ? ordered[middle] : (ordered[middle - 1] + ordered[middle]) / 2;
}

// Rates refer to useful payload, not to GPU FLOPs or 3D neural operations.
export function throughputMetrics(byteLength, width, iterations, durationMs) {
  if (![byteLength, width, iterations].every(Number.isSafeInteger) || byteLength < 0 || width < 1 || iterations < 1 ||
      !Number.isFinite(durationMs) || durationMs <= 0) {
    throw new RangeError('invalid throughput arguments');
  }
  const height = Math.max(1, Math.ceil(byteLength / (width * 4)));
  const allocatedBytes = width * height * 4;
  const seconds = durationMs / 1000;
  const payloadBytes = byteLength * iterations;
  return {
    width, height, byteLength, allocatedBytes, iterations, durationMs,
    inputGBps: payloadBytes / seconds / 1e9,
    inputGbps: payloadBytes * 8 / seconds / 1e9,
    roundTripsPerSecond: iterations / seconds,
    fragmentSamplesPerSecond: width * height * iterations / seconds,
    uploadReadbackGBpsLowerBound: 2 * allocatedBytes * iterations / seconds / 1e9,
    visualPointAttributeBytesIfAllBytesRendered: byteLength * 24
  };
}
