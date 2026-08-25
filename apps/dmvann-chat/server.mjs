import http from 'node:http';
import { readFile, stat } from 'node:fs/promises';
import { extname, join, normalize } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createChatPayload, RUNTIME_VERSION } from './core.mjs';

const ROOT = fileURLToPath(new URL('.', import.meta.url));
const PORT = Number.parseInt(process.env.PORT || '8787', 10);
const UPSTREAM_URL = (process.env.DMVANN_UPSTREAM_URL || '').replace(/\/$/, '');
const UPSTREAM_API_KEY = process.env.DMVANN_UPSTREAM_API_KEY || '';
const UPSTREAM_MODEL = process.env.DMVANN_MODEL || 'dmvann-default';
const REMOTE_ENABLED = Boolean(UPSTREAM_URL && process.env.DMVANN_ENABLE_REMOTE === '1');
const MAX_BODY = 1_048_576;

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.webmanifest': 'application/manifest+json',
};

function send(res, status, body, type = 'application/json; charset=utf-8') {
  res.writeHead(status, {
    'content-type': type,
    'cache-control': status >= 400 ? 'no-store' : 'no-cache',
    'x-content-type-options': 'nosniff',
    'x-frame-options': 'DENY',
    'referrer-policy': 'no-referrer',
    'permissions-policy': 'camera=(), microphone=(), geolocation=()',
    'content-security-policy': "default-src 'self'; connect-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; script-src 'self'; base-uri 'none'; frame-ancestors 'none'",
  });
  res.end(body);
}

async function readJson(req) {
  const chunks = [];
  let size = 0;
  for await (const chunk of req) {
    size += chunk.length;
    if (size > MAX_BODY) throw Object.assign(new Error('request body too large'), { statusCode: 413 });
    chunks.push(chunk);
  }
  const raw = Buffer.concat(chunks).toString('utf8');
  return raw ? JSON.parse(raw) : {};
}

async function proxyChat(req, res) {
  if (!REMOTE_ENABLED) {
    return send(res, 503, JSON.stringify({
      ok: false,
      error: UPSTREAM_URL ? 'REMOTE_MODEL_DISABLED' : 'UPSTREAM_NOT_CONFIGURED',
      message: UPSTREAM_URL
        ? 'Set DMVANN_ENABLE_REMOTE=1 on the server to opt in to remote inference.'
        : 'Set DMVANN_UPSTREAM_URL and DMVANN_ENABLE_REMOTE=1 on the server to enable generative chat.',
    }));
  }

  try {
    const body = await readJson(req);
    const payload = createChatPayload(body.messages, {
      model: UPSTREAM_MODEL,
      temperature: body.temperature,
    });
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 45_000);
    const headers = { 'content-type': 'application/json' };
    if (UPSTREAM_API_KEY) headers.authorization = `Bearer ${UPSTREAM_API_KEY}`;
    const response = await fetch(`${UPSTREAM_URL}/v1/chat/completions`, {
      method: 'POST',
      headers,
      body: JSON.stringify(payload),
      signal: controller.signal,
    }).finally(() => clearTimeout(timeout));
    const text = await response.text();
    if (!response.ok) {
      return send(res, 502, JSON.stringify({
        ok: false,
        error: 'UPSTREAM_FAILURE',
        upstreamStatus: response.status,
      }));
    }
    let parsed;
    try { parsed = JSON.parse(text); } catch { parsed = null; }
    const content = parsed?.choices?.[0]?.message?.content;
    if (typeof content !== 'string') {
      return send(res, 502, JSON.stringify({ ok: false, error: 'INVALID_UPSTREAM_RESPONSE' }));
    }
    return send(res, 200, JSON.stringify({
      ok: true,
      model: parsed?.model || payload.model,
      message: { role: 'assistant', content },
    }));
  } catch (error) {
    const status = error?.statusCode || (error instanceof SyntaxError ? 400 : 500);
    return send(res, status, JSON.stringify({
      ok: false,
      error: status === 400 ? 'INVALID_JSON' : 'CHAT_REQUEST_FAILED',
      message: String(error?.message || error),
    }));
  }
}

async function serveStatic(pathname, res) {
  const requested = pathname === '/' ? 'index.html' : pathname.replace(/^\/+/, '');
  const safe = normalize(requested).replace(/^([.][.][/\\])+/, '');
  const path = join(ROOT, safe);
  if (!path.startsWith(ROOT)) return send(res, 403, 'forbidden', 'text/plain; charset=utf-8');
  try {
    const info = await stat(path);
    if (!info.isFile()) throw new Error('not a file');
    const data = await readFile(path);
    send(res, 200, data, MIME[extname(path)] || 'application/octet-stream');
  } catch {
    send(res, 404, 'not found', 'text/plain; charset=utf-8');
  }
}

export function createServer() {
  return http.createServer(async (req, res) => {
    const url = new URL(req.url || '/', 'http://localhost');
    if (req.method === 'GET' && url.pathname === '/healthz') {
      return send(res, 200, JSON.stringify({
        ok: true,
        service: 'dmvann-chat',
        version: RUNTIME_VERSION,
        upstreamConfigured: Boolean(UPSTREAM_URL),
        remoteEnabled: REMOTE_ENABLED,
      }));
    }
    if (req.method === 'GET' && url.pathname === '/api/runtime') {
      return send(res, 200, JSON.stringify({
        ok: true,
        version: RUNTIME_VERSION,
        model: UPSTREAM_MODEL,
        upstreamConfigured: Boolean(UPSTREAM_URL),
        remoteEnabled: REMOTE_ENABLED,
      }));
    }
    if (req.method === 'POST' && url.pathname === '/api/chat') return proxyChat(req, res);
    if (req.method !== 'GET' && req.method !== 'HEAD') return send(res, 405, 'method not allowed', 'text/plain; charset=utf-8');
    return serveStatic(url.pathname, res);
  });
}

if (import.meta.url === `file://${process.argv[1]}`) {
  createServer().listen(PORT, '0.0.0.0', () => {
    console.log(`DMVANN Chat listening on http://0.0.0.0:${PORT}`);
  });
}
