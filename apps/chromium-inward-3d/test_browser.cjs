'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {pathToFileURL} = require('node:url');
const playwright = require('playwright');
const {createServer} = require('./serve.cjs');

const engine = process.env.BROWSER_NAME || 'chromium';
assert.ok(['chromium', 'webkit'].includes(engine), 'supported browser engine');
const output = path.join(__dirname, 'test-results', engine);
fs.mkdirSync(output, {recursive: true});
const server = createServer();
const policyKey = 'chromium-3d-inward-policy-v1';
const root = 'document.getElementById("chromium-rendering-3d").crEmulator';
const read = page => page.evaluate(() => {
  const api = document.getElementById('chromium-rendering-3d').crEmulator;
  return {snapshot: api.snapshot(), feedback: api.feedback(), trace: api.trace()};
});
const waitReady = page => page.waitForFunction(`!!${root}`, undefined, {timeout: 15000});
const settle = page => page.waitForFunction(() => {
  const api = document.getElementById('chromium-rendering-3d')?.crEmulator;
  if (!api) return false;
  const s = api.snapshot();
  return !s.playing && !s.optimising && !s.feedbackQueued && api.feedback().pass > 0;
}, undefined, {timeout: 20000});
const completeFrame = page => page.locator('#cr-stage-select').selectOption('9');
const autoOff = page => page.locator('[data-action="auto"]').uncheck();

async function newPage(browser, profile = {}, seed = {}) {
  const context = await browser.newContext({viewport: {width: 1280, height: 1000},
    reducedMotion: 'reduce', ...profile});
  const page = await context.newPage();
  const errors = [], externalRequests = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('request', request => {
    if (/^https?:/.test(request.url()) && !request.url().startsWith(baseURL)) externalRequests.push(request.url());
  });
  await context.route('**/*', route => {
    const url = route.request().url();
    return url.startsWith(baseURL) || url.startsWith('file:') ? route.continue() : route.abort();
  });
  await page.addInitScript(({policyKey, stored, blocked, fault}) => {
    // Keep the first few offscreen canvases for failure evidence, without
    // changing renderer state or injecting a candidate into the live model.
    const createElement = document.createElement.bind(document);
    window.__crTestCanvases = [];
    document.createElement = (...args) => {
      const element = createElement(...args);
      if (String(args[0]).toLowerCase() === 'canvas' && window.__crTestCanvases.length < 256) window.__crTestCanvases.push(element);
      return element;
    };
    if (blocked) Object.defineProperty(window, 'localStorage', {get() { throw new DOMException('Storage denied', 'SecurityError'); }});
    else if (stored !== undefined) try { localStorage.setItem(policyKey, stored); } catch {}
    if (fault) {
      const drawImage = CanvasRenderingContext2D.prototype.drawImage;
      CanvasRenderingContext2D.prototype.drawImage = function (image, ...args) {
        // Only corrupt 128px raster tiles. The reference uses the complete
        // display list and therefore detects this missing rendering work.
        if (image instanceof HTMLCanvasElement && image.width === 128 && image.height <= 128) return;
        return drawImage.call(this, image, ...args);
      };
      const putImageData = CanvasRenderingContext2D.prototype.putImageData;
      CanvasRenderingContext2D.prototype.putImageData = function (pixels, ...args) {
        if (pixels.width === 128 && pixels.height <= 128) return;
        return putImageData.call(this, pixels, ...args);
      };
    }
  }, {policyKey, ...seed});
  return {page, context, errors, externalRequests};
}

let baseURL;
const reports = [];
async function failureEvidence(page, name, error) {
  await page.screenshot({path: path.join(output, name + '-failure.png'), fullPage: true}).catch(() => {});
  const diagnostics = await page.evaluate(() => {
    const api = document.getElementById('chromium-rendering-3d')?.crEmulator;
    const images = (window.__crTestCanvases || []).filter(c => c.width === 320 && c.height === 226).slice(0, 3);
    const actual = images[1]?.getContext('2d').getImageData(0, 0, 320, 226).data;
    const expected = images[2]?.getContext('2d').getImageData(0, 0, 320, 226).data;
    const samples = [];
    if (actual && expected) for (let i = 0; i < actual.length && samples.length < 20; i += 4) {
      if ([0, 1, 2, 3].some(j => actual[i + j] !== expected[i + j])) samples.push({x: i / 4 % 320, y: Math.floor(i / 4 / 320), actual: Array.from(actual.slice(i, i + 4)), expected: Array.from(expected.slice(i, i + 4))});
    }
    return {snapshot: api?.snapshot(), feedback: api?.feedback(), samples,
      images: images.map(c => c.toDataURL('image/png'))};
  }).catch(error => ({diagnosticError: String(error)}));
  for (const [i, data] of (diagnostics.images || []).entries()) fs.writeFileSync(path.join(output, `${name}-buffer-${i}.png`), Buffer.from(data.split(',')[1], 'base64'));
  delete diagnostics.images;
  return {profile: name, status: 'FAIL', error: String(error.stack || error), diagnostics};
}
async function run() {
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  baseURL = 'http://127.0.0.1:' + server.address().port;
  let browser;
  try {
    const health = await fetch(baseURL + '/health.json');
    assert.equal(health.status, 200);
    assert.match((await health.json()).htmlSha256, /^[0-9a-f]{64}$/);
    assert.equal((await fetch(baseURL + '/README.md')).status, 404, 'server serves only explicit public routes');
    assert.equal((await fetch(baseURL, {method: 'POST'})).status, 405);
    browser = await playwright[engine].launch({headless: true});
    for (const profile of [
      {name: 'desktop-light', options: {colorScheme: 'light', reducedMotion: 'no-preference'}},
      {name: 'mobile-dark', options: {viewport: {width: 390, height: 844}, colorScheme: 'dark', isMobile: true, hasTouch: true}},
    ]) {
      const m = await newPage(browser, profile.options);
      const {page, context} = m;
      try {
        await page.goto(baseURL, {waitUntil: 'load'});
        await waitReady(page);
        assert.equal((await read(page)).snapshot.pixelDiff, 0, 'the first frame verifies before automatic tuning');
        await settle(page);
        const tuned = await read(page), f = tuned.feedback;
        assert.equal(tuned.snapshot.pixelDiff, 0);
        assert.ok(tuned.snapshot.frame >= 1);
        assert.equal(f.best.checkedPixels, 433920);
        assert.ok(f.best.valid && f.best.peakPixelBytes <= 2000000);
        assert.ok(f.best.work.drawCalls < f.baseline.work.drawCalls, 'native renderer adopts a measured reduction');
        assert.ok(f.iterations <= 8 && f.accepted > 0);
        assert.ok(f.history.every(r => !r.accepted || r.mismatchedPixels === 0));
        assert.ok(f.radius < f.initialRadius / 50);
        assert.equal(tuned.snapshot.logicalCells, 1024 ** 3);
        assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false, 'no horizontal overflow');
        await page.screenshot({path: path.join(output, profile.name + '.png'), fullPage: true});
        await autoOff(page);
        // Once loaded, operations require no network connection or assets.
        await context.setOffline(true);
        const audit = await page.evaluate(() => {
          const api = document.getElementById('chromium-rendering-3d').crEmulator;
          return api.probePolicy(api.snapshot().policy);
        });
        assert.equal(audit.valid, true);
        assert.equal((await read(page)).snapshot.frameHash, tuned.snapshot.frameHash, 'isolated probes preserve the live image');
        const click = action => profile.options.hasTouch ? page.locator(`[data-action="${action}"]`).tap() : page.locator(`[data-action="${action}"]`).click();
        await click('run'); await completeFrame(page);
        const replay = (await read(page)).snapshot;
        assert.deepEqual(replay.plan, replay.policy.reuseDocument ? [0, 8, 9] : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        assert.equal(replay.work.stageExecutions, replay.policy.reuseDocument ? 3 : 10);
        assert.equal(replay.rasterized, 0);
        assert.equal(replay.frameHash, tuned.snapshot.frameHash);
        await click('mutate'); await click('run');
        const paused = (await read(page)).snapshot;
        assert.equal(paused.playing, false);
        await page.waitForTimeout(100);
        assert.equal((await read(page)).snapshot.frame, paused.frame);
        await completeFrame(page);
        const changed = (await read(page)).snapshot;
        assert.equal(changed.pixelDiff, 0);
        assert.notEqual(changed.frameHash, tuned.snapshot.frameHash);
        assert.ok(changed.rasterized > 0 && changed.reused > 0);
        await click('scroll'); await completeFrame(page);
        const scrolled = (await read(page)).snapshot;
        assert.equal(scrolled.scroll, 24);
        assert.equal(scrolled.pixelDiff, 0);
        assert.equal(scrolled.rasterized, 0);
        assert.deepEqual(scrolled.plan, [8, 9]);

        const reference = await newPage(browser, profile.options);
        try {
          await reference.page.goto(baseURL); await waitReady(reference.page); await autoOff(reference.page);
          await reference.page.locator('[data-action="mutate"]').click(); await completeFrame(reference.page);
          await reference.page.locator('[data-action="scroll"]').click(); await completeFrame(reference.page);
          assert.equal(scrolled.frameHash, (await read(reference.page)).snapshot.frameHash, 'optimized mutation and scroll match untuned output');
        } finally { await reference.context.close(); }

        await page.locator('#cr-stage-select').selectOption('2');
        assert.equal((await read(page)).snapshot.stage, 2);
        await completeFrame(page);
        await page.locator('.cr-scene').scrollIntoViewIfNeeded();
        const box = await page.locator('.cr-scene').boundingBox();
        const yaw = (await read(page)).snapshot.yaw;
        const x = box.x + box.width * 0.35, y = box.y + box.height * 0.35;
        if (profile.options.hasTouch && engine === 'chromium') {
          const cdp = await context.newCDPSession(page);
          await cdp.send('Input.dispatchTouchEvent', {type: 'touchStart', touchPoints: [{x, y}]});
          await cdp.send('Input.dispatchTouchEvent', {type: 'touchMove', touchPoints: [{x: x + 48, y: y + 3}]});
          await cdp.send('Input.dispatchTouchEvent', {type: 'touchEnd', touchPoints: []});
          await cdp.detach();
        } else {
          await page.mouse.move(x, y); await page.mouse.down();
          await page.mouse.move(x + 48, y + 3, {steps: 3}); await page.mouse.up();
        }
        assert.notEqual((await read(page)).snapshot.yaw, yaw, 'pointer delivery or native Chromium touch changes the 3D camera');
        const beforeTheme = (await read(page)).snapshot.frame;
        await page.emulateMedia({colorScheme: profile.options.colorScheme === 'dark' ? 'light' : 'dark'});
        await page.waitForFunction(frame => document.getElementById('chromium-rendering-3d').crEmulator.snapshot().frame > frame, beforeTheme);
        assert.equal((await read(page)).snapshot.pixelDiff, 0);
        if (profile.options.hasTouch) {
          await page.setViewportSize({width: 320, height: 780});
          assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false, '320px layout fits');
        }
        await page.screenshot({path: path.join(output, profile.name + '-theme-switched.png'), fullPage: true});
        // Reload verifies the persisted policy on a fresh renderer before use.
        await context.setOffline(false);
        const stored = await page.evaluate(key => JSON.parse(localStorage.getItem(key)), policyKey);
        await page.reload(); await waitReady(page); await autoOff(page);
        const reload = (await read(page)).snapshot;
        assert.equal(reload.pixelDiff, 0);
        assert.deepEqual(reload.policy, {tile: stored.tile, tightBounds: stored.tightBounds, visibleOnly: stored.visibleOnly, reuseDocument: stored.reuseDocument});
        // A stopped controller retains the verified image and policy.
        await click('optimise');
        assert.equal((await read(page)).snapshot.optimising, true);
        await click('optimise');
        assert.equal((await read(page)).snapshot.optimising, false);
        assert.equal((await read(page)).snapshot.pixelDiff, 0);
        assert.deepEqual(m.errors, []);
        assert.deepEqual(m.externalRequests, []);
        reports.push({profile: profile.name, status: 'PASS', tuned, replay, changed, scrolled,
          reload, nativeTouchOrbit: !!profile.options.hasTouch && engine === 'chromium', externalRequests: 0});
      } catch (error) {
        reports.push(await failureEvidence(page, profile.name, error));
        throw error;
      } finally { await context.close(); }
    }

    for (const test of [
      {name: 'fault-rejection', seed: {fault: true}},
      {name: 'stored-policy-rollback', seed: {fault: true, stored: JSON.stringify({tile: 128, tightBounds: true, visibleOnly: true, reuseDocument: true})}},
      {name: 'corrupt-storage', seed: {stored: '{broken'}},
      {name: 'blocked-storage', seed: {blocked: true}},
      {name: 'direct-file-offline', file: true, seed: {}},
    ]) {
      const m = await newPage(browser, {}, test.seed);
      try {
        if (test.file) await m.context.setOffline(true);
        await m.page.goto(test.file ? pathToFileURL(path.join(__dirname, 'index.html')).href : baseURL);
        await settle(m.page);
        const result = await read(m.page);
        assert.equal(result.snapshot.pixelDiff, 0);
        assert.ok(result.snapshot.frame >= 1);
        if (test.seed.fault) {
          assert.notEqual(result.snapshot.policy.tile, 128);
          // A slow engine may exhaust the bounded search budget before visiting
          // 128px. Exercise that exact fault independently without changing the
          // live image or extending the automatic controller's time budget.
          const faultProbe = await m.page.evaluate(() => document.getElementById('chromium-rendering-3d').crEmulator.probePolicy({tile: 128, tightBounds: true, visibleOnly: true, reuseDocument: true}));
          assert.equal(faultProbe.valid, false, 'dropped raster tiles cannot pass the independent verification gate');
          assert.ok(faultProbe.mismatchedPixels > 0);
          assert.equal((await read(m.page)).snapshot.frameHash, result.snapshot.frameHash);
          result.faultProbe = faultProbe;
        }
        if (test.name === 'stored-policy-rollback') assert.ok(result.feedback.rollbacks > 0);
        assert.deepEqual(m.errors, []);
        assert.deepEqual(m.externalRequests, []);
        reports.push({profile: test.name, status: 'PASS', ...result});
      } catch (error) {
        reports.push(await failureEvidence(m.page, test.name, error));
        throw error;
      } finally { await m.context.close(); }
    }
    fs.writeFileSync(path.join(output, 'validation.json'), JSON.stringify({schemaVersion: 1,
      status: 'PASS', engine, browserVersion: browser.version(), node: process.version,
      scope: 'native browser rendering, offline controls, responsive layout, policy persistence, fault rejection and rollback', reports}, null, 2) + '\n');
    console.log('PASS ' + engine + ': ' + reports.length + ' profiles; pixel verification, rollback and offline interaction');
  } finally {
    if (browser) await browser.close();
    await new Promise(resolve => server.close(resolve));
  }
}
run().catch(error => {
  fs.writeFileSync(path.join(output, 'validation.json'), JSON.stringify({schemaVersion: 1,
    status: 'FAIL', engine, node: process.version, error: String(error.stack || error), reports}, null, 2) + '\n');
  console.error(error); process.exitCode = 1;
});
