import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { once } from 'node:events';
import { existsSync, readFileSync } from 'node:fs';
import http from 'node:http';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { test } from 'node:test';
import vm from 'node:vm';
import { createBridgeServer } from './server.mjs';

const dir = dirname(fileURLToPath(import.meta.url));
const bin = resolve(process.env.CLIFFORD_BIN ?? resolve(dir, '../../build/clifford-torus-live/jarvisx-clifford-torus4d'));
assert.ok(existsSync(bin), 'compile the C++ target and set CLIFFORD_BIN before testing');

function run(...args) {
  return spawnSync(bin, args, { encoding: 'utf8', timeout: 10000, maxBuffer: 1024 * 1024 });
}

test('existing C++ geometric invariants and queue accounting', () => {
  const result = run('--self-test');
  assert.equal(result.status, 0, result.stderr);
  assert.match(result.stdout, /PASS: 4D geometry/);
});

test('C++ produces bounded, numerically valid 3D vertices over NDJSON', () => {
  const result = run('--stream-json', '--frames', '2', '--grid', '8',
    '--dt', '0.015', '--time-scale', '1000000', '--pace-ms', '0');
  assert.equal(result.status, 0, result.stderr);
  const frames = result.stdout.trim().split('\n').map(JSON.parse);
  assert.equal(frames.length, 2);
  for (let i = 0; i < frames.length; i++) {
    const f = frames[i];
    assert.equal(f.protocol, 'jx-clifford-frame-v1');
    assert.equal(f.frame, i);
    assert.equal(f.grid, 8);
    assert.equal(f.vertices.length, 8 * 8 * 3);
    assert.ok(f.vertices.every(Number.isFinite));
    assert.ok(f.max_norm2_error < 1e-10);
    assert.ok(f.orthogonality_error < 1e-12);
    assert.ok(f.min_projection_denominator > 0);
    assert.ok(f.compute_seconds > 0);
    assert.ok(f.json_encode_seconds >= 0);
    assert.ok(f.measured_vertices_per_s > 0);
    assert.equal(f.simulated_s, (i + 1) * 15000);
  }
  // Independently reconstruct first vertex: (2,0,1,0), XW then YW.
  const ax = 0.012 * 15000;
  const ay = 0.008 * 15000;
  const x = 2 * Math.cos(ax);
  const w1 = 2 * Math.sin(ax);
  const y = -w1 * Math.sin(ay);
  const w2 = w1 * Math.cos(ay);
  const scale = 4 / (4 - w2);
  const expected = [x * scale, y * scale, scale];
  for (let i = 0; i < 3; i++)
    assert.ok(Math.abs(frames[0].vertices[i] - expected[i]) < 1e-8);
});

test('independent 4D validator accepts C++ geometry and rejects corrupted coordinates', () => {
  const verifierSource = readFileSync(resolve(dir, 'frame_verifier.js'), 'utf8');
  const isolated = {};
  vm.runInNewContext(verifierSource, isolated, { filename:'frame_verifier.js' });
  const verify = isolated.JarvisXFrameVerifier.verifyFrame;
  const result = run('--stream-json', '--frames', '1', '--grid', '36',
    '--time-scale', '1000000', '--pace-ms', '0');
  assert.equal(result.status, 0, result.stderr);
  const frame = JSON.parse(result.stdout.trim());
  const valid = verify(frame, 64);
  assert.equal(valid.ok, true, valid.error);
  assert.equal(valid.checked, 64);
  assert.ok(valid.maxError < 5e-8);
  const tampered = { ...frame, vertices:[...frame.vertices] };
  // Change a sampled vertex deliberately, forcing the verifier to fail closed.
  tampered.vertices[0] += 0.15;
  const invalid = verify(tampered, 64);
  assert.equal(invalid.ok, false);
  assert.match(invalid.error, /Projection mismatch/);
  assert.equal(verify({ ...frame, json_encode_seconds: NaN }).ok, false);
  assert.equal(verify({ ...frame, max_norm2_error: 1 }).ok, false);
});

test('stream rejects unbounded grid and pacing', () => {
  assert.equal(run('--stream-json', '--frames', '2', '--grid', '65').status, 1);
  assert.equal(run('--stream-json', '--frames', '2', '--pace-ms', '1001').status, 1);
});

test('browser script parses and retains a GPU buffer between updates', () => {
  const html = readFileSync(resolve(dir, 'index.html'), 'utf8');
  const scripts = [...html.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/g)];
  const inline = scripts.filter((match) => match[1].trim().length > 0);
  assert.equal(inline.length, 1);
  new vm.Script(inline[0][1], { filename:'index-inline.js' });
  assert.match(html, /new EventSource/);
  assert.match(html, /THREE.DynamicDrawUsage/);
  assert.match(html, /needsUpdate = true/);
  assert.doesNotMatch(html, /updateTorusMesh\(/);
});

test('HTTP bridge serves page, health and an actual C++ SSE frame', async () => {
  const server = createBridgeServer({ bin });
  server.listen(0, '127.0.0.1');
  await once(server, 'listening');
  const port = server.address().port;
  const base = 'http://127.0.0.1:' + port;
  try {
    const health = await fetch(base + '/health');
    assert.equal(health.status, 200);
    assert.equal((await health.json()).protocol, 'jx-clifford-frame-v1');
    const home = await fetch(base);
    assert.equal(home.status, 200);
    assert.match(await home.text(), /JARVIS X :: CLIFFORD 4D/);
    const verifier = await fetch(base + '/frame_verifier.js');
    assert.equal(verifier.status, 200);
    assert.match(verifier.headers.get('content-type'), /application\/javascript/);
    assert.match(await verifier.text(), /verifyFrame/);
    const invalid = await fetch(base + '/events?time_scale=NaN');
    assert.equal(invalid.status, 400);
    const invalid2 = await fetch(base + '/events?time_scale=0');
    assert.equal(invalid2.status, 400);

    const frame = await new Promise((resolveFrame, reject) => {
      const req = http.get(base + '/events?time_scale=1000', (res) => {
        assert.equal(res.statusCode, 200);
        assert.match(res.headers['content-type'], /text\/event-stream/);
        let buffer = '';
        const timeout = setTimeout(() => { res.destroy(); reject(new Error('SSE frame timeout')); }, 7000);
        res.setEncoding('utf8');
        res.on('data', (chunk) => {
          buffer += chunk;
          const match = buffer.match(/data: (\{[^\n]*\})\n\n/);
          if (match) {
            clearTimeout(timeout);
            res.destroy();
            resolveFrame(JSON.parse(match[1]));
          }
        });
        res.on('error', (err) => { clearTimeout(timeout); reject(err); });
      });
      req.on('error', reject);
    });
    assert.equal(frame.protocol, 'jx-clifford-frame-v1');
    assert.equal(frame.grid, 36);
    assert.equal(frame.vertices.length, 36 * 36 * 3);
    assert.ok(frame.json_encode_seconds >= 0);
  } finally {
    await new Promise((resolveClose) => server.close(resolveClose));
  }
});
