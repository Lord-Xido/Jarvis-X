// Local-only development bridge: streams the real C++ Clifford frame protocol.
// Node 20+, no third-party dependencies, no shell execution.
import http from 'node:http';
import { spawn } from 'node:child_process';
import { existsSync, readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const directory = dirname(fileURLToPath(import.meta.url));
const defaultBinary = resolve(directory, '../../build/clifford-torus-live/jarvisx-clifford-torus4d');
const binaryPath = resolve(process.env.CLIFFORD_BIN || defaultBinary);
const MAX_CLIENTS = 2;
const MAX_LINE_BYTES = 1024 * 1024;

function parseSpeed(url) {
  const raw = url.searchParams.get('time_scale') ?? '1000000';
  const speed = Number(raw);
  if (!Number.isFinite(speed) || speed < 1 || speed > 2000000 || !/^\d+(?:\.\d+)?$/.test(raw)) {
    throw new RangeError('time_scale must be a number from 1 to 2000000');
  }
  return speed;
}

export function createBridgeServer({ bin = binaryPath, page = resolve(directory, 'index.html') } = {}) {
  let activeClients = 0;
  const processes = new Set();
  const server = http.createServer((req, res) => {
    const url = new URL(req.url || '/', 'http://localhost');
    if (req.method !== 'GET') {
      res.writeHead(405, { 'Allow': 'GET' });
      res.end('Method not allowed');
      return;
    }
    if (url.pathname === '/health') {
      const ready = existsSync(bin);
      res.writeHead(ready ? 200 : 503, { 'Content-Type': 'application/json; charset=utf-8', 'Cache-Control': 'no-store' });
      res.end(JSON.stringify({ ready, activeClients, protocol: 'jx-clifford-frame-v1' }));
      return;
    }
    if (url.pathname === '/') {
      const html = readFileSync(page);
      res.writeHead(200, {
        'Content-Type': 'text/html; charset=utf-8',
        'Cache-Control': 'no-store',
        'X-Content-Type-Options': 'nosniff',
      });
      res.end(html);
      return;
    }
    if (url.pathname === '/frame_verifier.js') {
      const js = readFileSync(resolve(directory, 'frame_verifier.js'));
      res.writeHead(200, {
        'Content-Type': 'application/javascript; charset=utf-8',
        'Cache-Control': 'no-store',
        'X-Content-Type-Options': 'nosniff',
      });
      res.end(js);
      return;
    }
    if (url.pathname !== '/events') {
      res.writeHead(404);
      res.end('Not found');
      return;
    }
    let speed;
    try { speed = parseSpeed(url); } catch (err) {
      res.writeHead(400);
      res.end(err.message);
      return;
    }
    if (!existsSync(bin)) {
      res.writeHead(503);
      res.end('Build the C++ Clifford executable first');
      return;
    }
    if (activeClients >= MAX_CLIENTS) {
      res.writeHead(429);
      res.end('Too many live streams');
      return;
    }
    const args = [
      '--stream-json', '--frames', '1000000', '--grid', '36',
      '--dt', '0.015', '--time-scale', String(speed), '--pace-ms', '40'
    ];
    const child = spawn(bin, args, { stdio: ['ignore', 'pipe', 'pipe'] });
    processes.add(child);
    activeClients++;
    let closed = false;
    let pending = '';
    let stderr = '';
    const cleanup = () => {
      if (closed) return;
      closed = true;
      processes.delete(child);
      activeClients--;
      if (!child.killed) child.kill('SIGTERM');
      if (!res.writableEnded) res.end();
    };
    res.writeHead(200, {
      'Content-Type': 'text/event-stream; charset=utf-8',
      'Cache-Control': 'no-cache, no-transform',
      'Connection': 'keep-alive',
      'X-Accel-Buffering': 'no',
      'X-Content-Type-Options': 'nosniff',
    });
    res.write('retry: 2500\n\n');
    child.stdout.setEncoding('utf8');
    child.stdout.on('data', (chunk) => {
      if (closed) return;
      pending += chunk;
      if (Buffer.byteLength(pending, 'utf8') > MAX_LINE_BYTES) {
        cleanup();
        return;
      }
      let pos;
      while ((pos = pending.indexOf('\n')) >= 0) {
        const line = pending.slice(0, pos);
        pending = pending.slice(pos + 1);
        if (!line) continue;
        // The spawned C++ process is the sole trusted producer of the frame payload.
        if (!res.write('data: ' + line + '\n\n')) {
          child.stdout.pause();
          res.once('drain', () => { if (!closed) child.stdout.resume(); });
        }
      }
    });
    child.stderr.setEncoding('utf8');
    child.stderr.on('data', (s) => { stderr = (stderr + s).slice(-2000); });
    child.on('error', (err) => {
      if (!closed) res.write('event: engine-error\ndata: ' + JSON.stringify({ message: err.message }) + '\n\n');
      cleanup();
    });
    child.on('close', (code) => {
      if (!closed) res.write('event: engine-end\ndata: ' + JSON.stringify({ code, message: stderr }) + '\n\n');
      cleanup();
    });
    // IncomingMessage 'close' may mean the GET request was read, not that the SSE client disconnected.
    res.on('close', cleanup);
  });
  server.on('close', () => { for (const child of processes) child.kill('SIGTERM'); });
  return server;
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const port = Number(process.env.PORT ?? '8787');
  if (!Number.isInteger(port) || port < 1 || port > 65535) {
    throw new Error('PORT must be an integer from 1 to 65535');
  }
  const server = createBridgeServer();
  // Deliberately loopback-only: do not expose a shell-spawning service to the internet.
  server.listen(port, '127.0.0.1', () => {
    process.stdout.write('Clifford viewer http://127.0.0.1:' + port + '\n');
  });
}
