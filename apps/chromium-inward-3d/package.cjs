'use strict';
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const {createHash} = require('node:crypto');

const html = fs.readFileSync(path.join(__dirname, 'index.html'), 'utf8');
const scripts = [...html.matchAll(/<script>\s*([\s\S]*?)<\/script>/g)];
assert.equal(scripts.length, 1, 'one self-contained emulator script');
new vm.Script(scripts[0][1], {filename: 'index.html'});
assert.ok(!/<(?:script|link|iframe)\b[^>]*(?:src|href)=/i.test(html), 'no external runtime assets');
const destination = path.join(__dirname, 'dist', 'chromium-inward-3d');
fs.mkdirSync(destination, {recursive: true});
const manifest = {schemaVersion: 1, app: 'chromium-inward-3d', version: '0.1.0',
  classification: 'bounded browser demonstration', runtimeDependencies: [],
  pipelineStages: 10, framebuffer: {width: 320, height: 226, format: 'RGBA8'},
  logicalCells: 1024 ** 3, denseLogicalAllocation: false,
  optimization: {maximumCandidateProbes: 8, softComputeBudgetMs: 2500,
    maximumProbePixelBytes: 2000000, acceptance: 'zero pixel differences and strictly lower counted work'},
  canonicalRuntimeAuthority: false};
fs.writeFileSync(path.join(destination, 'manifest.json'), JSON.stringify(manifest, null, 2) + '\n');
for (const name of ['index.html', 'serve.cjs', 'README.md']) fs.copyFileSync(path.join(__dirname, name), path.join(destination, name));
const names = ['README.md', 'index.html', 'manifest.json', 'serve.cjs'];
const sums = names.map(name => createHash('sha256').update(fs.readFileSync(path.join(destination, name))).digest('hex') + '  ' + name);
fs.writeFileSync(path.join(destination, 'SHA256SUMS'), sums.join('\n') + '\n');
console.log('Packaged offline app and SHA-256 manifest: ' + destination);
