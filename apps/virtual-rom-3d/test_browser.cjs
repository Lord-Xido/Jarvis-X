'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const http = require('node:http');
const path = require('node:path');
const {pathToFileURL} = require('node:url');
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const threePath = process.env.THREE_JS_PATH || require.resolve('three/build/three.min.js');
const sourcePath = path.join(__dirname, 'index.html');
const html = fs.readFileSync(sourcePath);
const output = process.env.SCREENSHOT_DIR || path.join(__dirname, 'test-results');
const server = http.createServer((req, res) => {
  if (req.url === '/' || req.url === '/index.html') {res.writeHead(200, {'Content-Type':'text/html; charset=utf-8'});res.end(html);}
  else {res.writeHead(404);res.end();}
});
const snapshot = page => page.evaluate(() => window.getVirtualROMTelemetry());
async function run() {
  fs.mkdirSync(output, {recursive:true});
  await new Promise(resolve => server.listen(0,'127.0.0.1',resolve));
  const base = 'http://127.0.0.1:' + server.address().port;
  let browser;
  try {
    browser = await chromium.launch({headless:true,args:['--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader']});
    for (const profile of [
      {name:'desktop-offline',width:1440,height:1000,offline:true},
      {name:'mobile-offline',width:390,height:844,offline:true,mobile:true},
      {name:'short-mobile-offline',width:375,height:667,offline:true,mobile:true},
      {name:'desktop-webgl',width:1440,height:1000,offline:false},
      {name:'file-offline',width:1280,height:900,offline:true,file:true},
    ]) {
      const context = await browser.newContext({viewport:{width:profile.width,height:profile.height},isMobile:!!profile.mobile,hasTouch:!!profile.mobile,reducedMotion:'reduce'});
      const page = await context.newPage(), errors=[];
      page.on('pageerror',e=>errors.push(e.message));
      await page.route('https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js', route => profile.offline ? route.abort() : route.fulfill({path:threePath,contentType:'application/javascript'}));
      await page.goto(profile.file ? pathToFileURL(sourcePath).href : base, {waitUntil:'load'});
      await page.waitForFunction(backend => window.getVirtualROMTelemetry?.().backend === backend,profile.offline?'canvas':'webgl');
      const initial = await snapshot(page);
      assert.equal(initial.logicalBytes,2**30);assert.equal(initial.windowBytes,2**21);assert.equal(initial.residentPageBytes,0);assert.equal(initial.paused,true);
      assert.equal(initial.renderedSamples,profile.offline?4096:32768);
      assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false,'no page-wide horizontal overflow');
      if(profile.offline) {
        await page.waitForFunction(()=>{
          const canvas=document.getElementById('canvas'),bytes=canvas.getContext('2d').getImageData(0,0,canvas.width,canvas.height).data;
          let colored=0;for(let i=3;i<bytes.length;i+=4)if(bytes[i])colored++;return colored>500;
        });
      }
      const click = async selector => profile.mobile ? page.locator(selector).tap() : page.locator(selector).click();
      const settled = async () => page.waitForFunction(()=>!getVirtualROMTelemetry().busy);
      await click('#view-nebula');assert.equal((await snapshot(page)).view,'nebula');
      await page.screenshot({path:path.join(output,profile.name+'.png'),fullPage:true});
      await page.locator('#viewport').focus();await page.keyboard.press('ArrowRight');
      await page.waitForFunction(before=>getVirtualROMTelemetry().rotation.y>before,initial.rotation.y);
      await page.keyboard.press('+');assert.ok((await snapshot(page)).distance<initial.distance);
      const still=await snapshot(page);await page.waitForTimeout(80);assert.equal((await snapshot(page)).elapsedSeconds,still.elapsedSeconds);
      await click('[data-image="zero"]');await settled();assert.equal((await snapshot(page)).windowEntropyBits,0);
      await page.locator('#address').fill('00000000');await page.locator('#byte-value').fill('AA');await click('#write-byte');await settled();
      assert.equal((await snapshot(page)).selectedByte,170);
      await page.locator('#address').fill('00200000');await click('#address-go');assert.equal((await snapshot(page)).selectedByte,0);
      await page.locator('#byte-value').fill('55');await click('#write-byte');await settled();
      await page.locator('#address').fill('00000000');await click('#address-go');assert.equal((await snapshot(page)).selectedByte,170);
      await page.locator('#bank-number').fill('1');await click('#bank-go');assert.equal(await page.locator('#hex-content th').first().textContent(),'00000080');
      await page.locator('#address').fill('3FFFFFFF');await click('#address-go');
      assert.equal((await snapshot(page)).bank,8388607);assert.equal(await page.locator('#hex-content th').first().textContent(),'3FFFFF80');
      await page.locator('#byte-value').fill('FF');await click('#write-byte');await settled();
      const high=await snapshot(page);
      await page.locator('#address').fill('40000000');await click('#address-go');assert.equal((await snapshot(page)).address,high.address);
      assert.ok(await page.locator('#status').evaluate(el=>el.classList.contains('error')));
      await page.locator('#address').fill('3FFFFFFF');
      await click('#encode');await settled();
      const encoded=await snapshot(page);assert.equal(encoded.codec.verified,true);assert.equal(encoded.codec.rawBytes,2**21);assert.equal(encoded.selectedByte,255);
      assert.ok(encoded.codec.encodedBytes<encoded.codec.rawBytes);
      if(profile.name==='desktop-offline') {
        const downloadEvent=page.waitForEvent('download');await click('#download');const download=await downloadEvent;
        const saved=fs.readFileSync(await download.path());assert.equal(saved.length,encoded.codec.encodedBytes);
        await page.locator('#byte-value').fill('00');await click('#write-byte');await settled();assert.equal((await snapshot(page)).codec,null);
        await page.locator('#file-input').setInputFiles({name:'window.vrm',mimeType:'application/octet-stream',buffer:saved});await settled();
        assert.equal((await snapshot(page)).selectedByte,255);
        const revision=(await snapshot(page)).revision,bad=Buffer.from(saved);bad[12]^=1;
        await page.locator('#file-input').setInputFiles({name:'corrupt.vrm',mimeType:'application/octet-stream',buffer:bad});await settled();
        assert.equal((await snapshot(page)).revision,revision);assert.match(await page.locator('#status').textContent(),/CRC32/);
      }
      if(!profile.offline) {
        await page.selectOption('#budget','65536');assert.equal((await snapshot(page)).renderedSamples,65536);
        await page.evaluate(()=>{
          const canvas=document.querySelector('#viewport canvas:not([hidden])');
          const gl=canvas.getContext('webgl2')||canvas.getContext('webgl');
          const extension=gl.getExtension('WEBGL_lose_context');if(!extension)throw new Error('No context-loss extension');extension.loseContext();
        });
        await page.waitForFunction(()=>getVirtualROMTelemetry().backend==='canvas');
        assert.equal((await snapshot(page)).renderedSamples,4096);assert.equal((await snapshot(page)).selectedByte,255);
      }
      assert.deepEqual(errors,[]);
      console.log('PASS '+profile.name+': layout, input, addressing, codec and graphics');
      await context.close();
    }
  } finally {if(browser)await browser.close();await new Promise(resolve=>server.close(resolve));}
}
run().catch(error=>{console.error(error);process.exitCode=1;});
