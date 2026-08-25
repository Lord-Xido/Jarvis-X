import test from 'node:test';
import assert from 'node:assert/strict';
import {
  DEFAULT_FIELD,
  createChatPayload,
  fingerprint,
  localControlPlaneResponse,
  normalizeMessages,
  stepField,
} from './core.mjs';
import { createServer } from './server.mjs';

test('message normalization is bounded and role-safe', () => {
  const result = normalizeMessages([{ role: 'tool', content: 'abc' }, { role: 'assistant', content: '' }]);
  assert.deepEqual(result, [{ role: 'user', content: 'abc' }]);
});

test('field evolution is deterministic and bounded', () => {
  const a = stepField(DEFAULT_FIELD, 'encode this state');
  const b = stepField(DEFAULT_FIELD, 'encode this state');
  assert.deepEqual(a, b);
  assert.ok(a.psi >= 0 && a.psi <= 1);
  assert.ok(a.phi >= -1 && a.phi <= 1);
  assert.ok(a.lambda > 0 && a.lambda <= 1);
  assert.ok(a.omega >= 0 && a.omega <= 1);
  assert.ok(a.theta >= -1 && a.theta <= 1);
  assert.equal(a.tick, 1);
});

test('fingerprint changes with content', () => {
  assert.equal(fingerprint('abc'), fingerprint('abc'));
  assert.notEqual(fingerprint('abc'), fingerprint('abd'));
});

test('chat payload requires a user message', () => {
  assert.throws(() => createChatPayload([{ role: 'assistant', content: 'x' }]), /user message/);
  const payload = createChatPayload([{ role: 'user', content: 'hello' }], { temperature: 9 });
  assert.equal(payload.temperature, 2);
  assert.equal(payload.stream, false);
});

test('local fallback clearly discloses no model was used', () => {
  const field = stepField(DEFAULT_FIELD, 'hello world');
  const text = localControlPlaneResponse('hello world', field);
  assert.match(text, /no remote model response was used/i);
  assert.match(text, /Ψ=/);
});

async function withServer(callback) {
  const server = createServer();
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  try {
    const address = server.address();
    return await callback(`http://127.0.0.1:${address.port}`);
  } finally {
    await new Promise((resolve) => server.close(resolve));
  }
}

test('health endpoint exposes readiness without secrets', async () => {
  await withServer(async (base) => {
    const response = await fetch(`${base}/healthz`);
    assert.equal(response.status, 200);
    const health = await response.json();
    assert.equal(health.ok, true);
    assert.equal(health.service, 'dmvann-chat');
    assert.equal(typeof health.remoteEnabled, 'boolean');
    assert.equal('apiKey' in health, false);
  });
});

test('chat endpoint fails closed when remote inference is disabled', async () => {
  await withServer(async (base) => {
    const response = await fetch(`${base}/api/chat`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ messages: [{ role: 'user', content: 'hello' }] }),
    });
    assert.equal(response.status, 503);
    const payload = await response.json();
    assert.ok(['UPSTREAM_NOT_CONFIGURED', 'REMOTE_MODEL_DISABLED'].includes(payload.error));
  });
});
