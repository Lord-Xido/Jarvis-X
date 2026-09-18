const assert = require('assert');
const { RecursiveByteANN3D } = require('../src/engine.js');

function roundTrip(text, depth=4) {
  const e = new RecursiveByteANN3D({ recursionDepth: depth, foldGain: 0.04, seed: 1337 });
  const r = e.run(text);
  return {e,r,text:new TextDecoder().decode(r.output)};
}

{
  const {r,text} = roundTrip('AUTOENCODER_3D_BYTES');
  assert.strictEqual(text, 'AUTOENCODER_3D_BYTES');
  assert.strictEqual(r.metrics.ber, 0);
  assert.strictEqual(r.metrics.byteAccuracy, 1);
}
{
  const payload = Uint8Array.from({length: 128}, (_,i)=>i);
  const e = new RecursiveByteANN3D({ recursionDepth: 1, foldGain: 0 });
  const r = e.run(payload);
  assert.strictEqual(r.output.length, payload.length);
  assert.ok(r.metrics.byteAccuracy > 0.95, 'Expected high initial codebook accuracy');
}
{
  const e = new RecursiveByteANN3D({ recursionDepth: 2 });
  const model = e.exportModel();
  const e2 = new RecursiveByteANN3D({ recursionDepth: 2 });
  e2.importModel(model);
  const r = e2.run('SDK_MODEL_ROUNDTRIP');
  assert.strictEqual(new TextDecoder().decode(r.output), 'SDK_MODEL_ROUNDTRIP');
}
console.log('all AE3D engine tests passed');
