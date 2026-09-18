const { RecursiveByteANN3D } = require('../src/engine.js');
const engine = new RecursiveByteANN3D({ recursionDepth: 4, beta: 0.85, foldGain: 0.04 });
const input = new TextEncoder().encode('AUTOENCODER_3D_BYTES');
const result = engine.run(input);
console.log({
  decoded: new TextDecoder().decode(result.output),
  metrics: result.metrics,
  config: engine.getConfig()
});
