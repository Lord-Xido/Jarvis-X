#!/usr/bin/env node
'use strict';
// Offline static artifact audit: NEVER evaluates the uploaded HTML module.
const fs = require('node:fs');
const assert = require('node:assert/strict');
const path = require('node:path');
const { sha256, classicalQuineSource, verifyBundledClassicQuine } = require('./quine_bitwise3d');

const filename = process.argv[2];
if (!filename) {
  console.error('Usage: node audit_react_artifact10.js "/path/to/React Artifact(10).html"');
  process.exitCode = 2;
} else {
  const bytes = fs.readFileSync(filename);
  const html = bytes.toString('utf8');
  const match = /<script type="module">([\s\S]*?)<\/script>/.exec(html);
  assert.ok(match, 'Expected an embedded module source');
  const js = match[1];
  assert.ok(js.includes('latent[i%512] += s.charCodeAt(i) * 0.0007;'), 'Expected displayed encoder not found');
  assert.ok(js.includes('const decode = (latent) => src;'), 'Expected displayed decoder not found');
  const fold = s => {
    const z = new Float32Array(512);
    for (let i = 0; i < s.length; i++) z[i % 512] += s.charCodeAt(i) * 0.0007;
    return Buffer.from(z.buffer);
  };
  const a = 'A' + '\0'.repeat(511) + 'B';
  const b = 'B' + '\0'.repeat(511) + 'A';
  assert.notEqual(a, b);
  assert.deepEqual(fold(a), fold(b), 'Expected a documented latent collision');
  const quine = verifyBundledClassicQuine(1);
  assert.equal(quine.source, classicalQuineSource());
  console.log(JSON.stringify({
    artifact: path.basename(filename),
    bytes: bytes.length,
    bits: bytes.length * 8,
    sha256: sha256(bytes),
    displayed512FloatEncoder: true,
    displayedClosureDecoderIgnoresLatent: true,
    distinctInputsCollideAtLatentBytes: true,
    classicQuineBytes: quine.bytes,
    classicalSourceSelfReproduces: true,
    exactHtmlSelfReproductionProven: false,
    warning: 'Source presentation and visual auto-evolution do not establish a true lossless ANN or full-HTML quine.'
  }, null, 2));
}
