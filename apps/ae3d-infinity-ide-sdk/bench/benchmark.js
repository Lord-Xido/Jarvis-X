const { RecursiveByteANN3D } = require('../src/engine.js');
const { performance } = require('perf_hooks');
const input = new TextEncoder().encode('AUTOENCODER_3D_BYTES');
const depths = [1,4,16,64,256];
for (const depth of depths) {
  const e = new RecursiveByteANN3D({ recursionDepth: depth, foldGain: 0.04 });
  for(let i=0;i<2;i++) e.run(input);
  const t0 = performance.now();
  const r = e.run(input);
  const t1 = performance.now();
  console.log(JSON.stringify({
    depth,
    ms: +(t1-t0).toFixed(3),
    accuracy: r.metrics.byteAccuracy,
    ber: r.metrics.ber,
    qMSE: r.metrics.quantMSE
  }));
}
