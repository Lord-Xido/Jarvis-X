'use strict';
const http = require('node:http');
const fs = require('node:fs');
const path = require('node:path');
const {createHash} = require('node:crypto');

function createServer() {
  const html = fs.readFileSync(path.join(__dirname, 'index.html'));
  const health = JSON.stringify({status: 'ok', app: 'chromium-inward-3d', schemaVersion: 1,
    htmlSha256: createHash('sha256').update(html).digest('hex')});
  return http.createServer((request, response) => {
    response.setHeader('Cache-Control', 'no-store');
    response.setHeader('X-Content-Type-Options', 'nosniff');
    response.setHeader('Content-Security-Policy', "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; base-uri 'none'; frame-ancestors 'none'; form-action 'none'");
    if (!['GET', 'HEAD'].includes(request.method)) {
      response.writeHead(405, {'Allow': 'GET, HEAD'}); response.end(); return;
    }
    let pathname;
    try { pathname = new URL(request.url, 'http://localhost').pathname; }
    catch { response.writeHead(400); response.end(); return; }
    const isPage = pathname === '/' || pathname === '/index.html';
    const isHealth = pathname === '/health.json';
    if (!isPage && !isHealth) { response.writeHead(404); response.end(); return; }
    response.writeHead(200, {'Content-Type': isPage ? 'text/html; charset=utf-8' : 'application/json; charset=utf-8'});
    response.end(request.method === 'HEAD' ? undefined : isPage ? html : health);
  });
}

if (require.main === module) {
  const port = Number(process.env.PORT || 8000);
  if (!Number.isInteger(port) || port < 0 || port > 65535) {
    console.error('PORT must be an integer from 0 to 65535.'); process.exitCode = 1;
  } else {
    const server = createServer();
    server.on('error', error => { console.error(error.message); process.exitCode = 1; });
    server.listen(port, '127.0.0.1', () => console.log('Chromium 3D emulator: http://127.0.0.1:' + server.address().port));
    for (const signal of ['SIGINT', 'SIGTERM']) process.on(signal, () => server.close());
  }
}
module.exports = {createServer};
