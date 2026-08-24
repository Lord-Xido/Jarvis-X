const CACHE_NAME = 'dm-cognitive-v2';
const LOCAL_SHELL = [
  './',
  './index.html',
  './core.mjs',
  './app.mjs',
  './manifest.webmanifest',
  './icon.svg',
];
const VENDOR_ORIGINS = new Set([
  'https://cdnjs.cloudflare.com',
  'https://cdn.jsdelivr.net',
]);

self.addEventListener('install', event => {
  event.waitUntil(caches.open(CACHE_NAME).then(cache => cache.addAll(LOCAL_SHELL)));
  self.skipWaiting();
});

self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(keys => Promise.all(keys.filter(key => key !== CACHE_NAME).map(key => caches.delete(key))))
      .then(() => self.clients.claim()),
  );
});

self.addEventListener('fetch', event => {
  const request = event.request;
  if (request.method !== 'GET') return;
  const url = new URL(request.url);
  const sameOrigin = url.origin === self.location.origin;
  const cacheableVendor = VENDOR_ORIGINS.has(url.origin);
  if (!sameOrigin && !cacheableVendor) return;

  event.respondWith((async () => {
    const cache = await caches.open(CACHE_NAME);
    const cached = await cache.match(request);
    if (cached) {
      event.waitUntil(fetch(request).then(response => {
        if (response.ok || response.type === 'opaque') return cache.put(request, response.clone());
        return undefined;
      }).catch(() => undefined));
      return cached;
    }
    try {
      const response = await fetch(request);
      if (response.ok || response.type === 'opaque') await cache.put(request, response.clone());
      return response;
    } catch (error) {
      if (sameOrigin && request.mode === 'navigate') {
        const fallback = await cache.match('./index.html');
        if (fallback) return fallback;
      }
      throw error;
    }
  })());
});
