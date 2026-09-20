'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const http = require('node:http');
const path = require('node:path');
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const cdn = 'https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js';
const threePath = process.env.THREE_JS_PATH || require.resolve('three/build/three.min.js');
const html = fs.readFileSync(path.join(__dirname, 'index.html'));
const output = process.env.SCREENSHOT_DIR || path.join(__dirname, 'test-results');
const server = http.createServer((request, response) => {
  if (request.url !== '/' && request.url !== '/index.html') { response.writeHead(204); response.end(); return; }
  response.writeHead(200, {'Content-Type': 'text/html; charset=utf-8'});
  response.end(html);
});
const snapshot = page => page.evaluate(() => window.getHyperscaleTelemetry());

async function run() {
  fs.mkdirSync(output, {recursive: true});
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const url = 'http://127.0.0.1:' + server.address().port;
  let browser;
  try {
    browser = await chromium.launch({headless: true, args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader']});
    for (const profile of [
      {name: 'desktop-offline', viewport: {width: 1440, height: 960}, offline: true},
      {name: 'mobile-offline', viewport: {width: 390, height: 844}, offline: true, isMobile: true, hasTouch: true},
      {name: 'short-mobile-offline', viewport: {width: 375, height: 667}, offline: true, isMobile: true, hasTouch: true},
      {name: 'desktop-webgl', viewport: {width: 1440, height: 960}, offline: false},
    ]) {
      const context = await browser.newContext({viewport: profile.viewport, isMobile: !!profile.isMobile, hasTouch: !!profile.hasTouch, reducedMotion: 'reduce'});
      const page = await context.newPage();
      const pageErrors = [];
      page.on('pageerror', error => pageErrors.push(error.message));
      await page.route(cdn, route => profile.offline ? route.abort() : route.fulfill({path: threePath, contentType: 'application/javascript'}));
      await page.goto(url, {waitUntil: 'load'});
      await page.waitForFunction(expected => window.getHyperscaleTelemetry?.().backend === expected, profile.offline ? 'canvas' : 'webgl');
      const initial = await snapshot(page);
      assert.equal(initial.paused, true, 'reduced motion starts paused');
      assert.equal(initial.simulatedParticles, 64000);
      assert.equal(initial.renderedParticles, profile.offline ? 4000 : 64000);
      assert.equal(initial.radialSummary.length, 64);
      assert.equal(initial.updatedParticles, 0);
      assert.ok(initial.targetRMSE > 5);
      const overflow = await page.evaluate(() => document.documentElement.scrollWidth > innerWidth);
      assert.equal(overflow, false, 'no horizontal overflow');
      const bounds = await page.evaluate(() => {
        const header = document.querySelector('header').getBoundingClientRect();
        const panel = document.querySelector('main').getBoundingClientRect();
        return {headerBottom: header.bottom, panelTop: panel.top};
      });
      assert.ok(bounds.panelTop >= bounds.headerBottom, 'controls must not overlap telemetry');
      await page.screenshot({path: path.join(output, profile.name + '.png'), fullPage: true});

      // Pause in the middle of a burst, then resume to its analytic endpoint.
      const click = async id => profile.hasTouch ? page.locator(id).tap() : page.locator(id).click();
      await click('#toggleHyperBtn');
      await click('#pauseBtn');
      const paused = await snapshot(page);
      assert.equal(paused.paused, true);
      await page.waitForTimeout(150);
      const still = await snapshot(page);
      assert.equal(still.elapsedSeconds, paused.elapsedSeconds);
      assert.equal(still.updatedParticles, paused.updatedParticles);
      await click('#pauseBtn');
      await page.waitForFunction(() => getHyperscaleTelemetry().state === 'contracted', undefined, {timeout: 15000});
      const finished = await snapshot(page);
      assert.ok(finished.targetRMSE < 1e-6);
      assert.ok(finished.updatedParticles >= 64000);
      assert.equal(await page.locator('#toggleHyperBtn').isDisabled(), true);
      await click('#pauseBtn');
      await click('#resetHyperBtn');
      const reset = await snapshot(page);
      assert.equal(reset.paused, true);
      assert.equal(reset.state, 'shell');
      assert.equal(reset.updatedParticles, 0);
      assert.deepEqual(reset.radialSummary, initial.radialSummary);

      // Keyboard and pointer controls keep working while motion is paused.
      const canvas = page.locator('canvas');
      await canvas.focus();
      await page.keyboard.press('ArrowRight');
      assert.ok((await snapshot(page)).orbit.yaw > reset.orbit.yaw);
      await page.keyboard.press('+');
      assert.ok((await snapshot(page)).orbit.radius < reset.orbit.radius);
      await page.evaluate(() => scrollTo(0, 0));
      if (profile.hasTouch) {
        const session = await context.newCDPSession(page);
        const touch = async (type, touchPoints) => session.send('Input.dispatchTouchEvent', {type, touchPoints});
        const before = (await snapshot(page)).orbit.radius;
        await touch('touchStart', [{x: 110, y: 290}, {x: 220, y: 290}]);
        await touch('touchMove', [{x: 105, y: 290}, {x: 225, y: 290}]);
        await touch('touchMove', [{x: 85, y: 290}, {x: 245, y: 290}]);
        await touch('touchEnd', []);
        assert.ok((await snapshot(page)).orbit.radius < before, 'pinch zoom changes camera radius');
      } else {
        const before = (await snapshot(page)).orbit.yaw;
        await page.mouse.move(900, 300);
        await page.mouse.down();
        await page.mouse.move(960, 330, {steps: 3});
        await page.mouse.up();
        assert.notEqual((await snapshot(page)).orbit.yaw, before);
      }

      if (!profile.offline) {
        await page.evaluate(() => {
          const canvas = document.querySelector('canvas');
          const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
          const extension = gl?.getExtension('WEBGL_lose_context');
          if (!extension) throw new Error('Context-loss extension unavailable');
          extension.loseContext();
        });
        await page.waitForFunction(() => getHyperscaleTelemetry().backend === 'canvas');
        assert.equal((await snapshot(page)).renderedParticles, 4000);
        assert.equal((await snapshot(page)).state, 'shell');
      }
      assert.deepEqual(pageErrors, []);
      console.log('PASS ' + profile.name + ': layout, controls, measurements and reset');
      await context.close();
    }
  } finally {
    if (browser) await browser.close();
    await new Promise(resolve => server.close(resolve));
  }
}
run().catch(error => { console.error(error); process.exitCode = 1; });
