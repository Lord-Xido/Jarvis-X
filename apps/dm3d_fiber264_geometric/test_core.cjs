// Exact finite algebra and deterministic latent contraction checks; no GPU required.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const html = fs.readFileSync(__dirname + '/index.html', 'utf8');
const code = html.match(/<script id="math-core">([\s\S]*?)<\/script>/)?.[1];
assert.ok(code, 'math core script located');
const box = { window: {}, Float64Array, Uint8Array, Math, Number, RangeError, Error };
vm.runInNewContext(code, box);
const C = box.window.DM3DCore;
let assertions = 0;
const check = (test, label) => { assert.ok(test, label); assertions++; };
check(C.CARD === 18_399_744, '264^3 cardinality');
check(C.order(C.maxShift) === 132, 'maximal translation period 132');
for (const i of [0, 1, 263, 264, 265, 2718, 752001, C.CARD - 1]) {
  const x = C.stateFromIndex(i);
  check(C.stateToIndex(x) === i, 'index roundtrip ' + i);
  check(C.equal(C.collapse(x), C.ZERO), 'constant collapse ' + i);
  check(C.equal(C.add(x, C.maxShift), C.add(C.maxShift, x)), 'abelian law ' + i);
  check(C.equal(C.decode512(C.embed512(x)), x), '512D observation reconstruction ' + i);
}
let x = C.stateFromIndex(752001);
const originalIndex = C.stateToIndex(x);
for (let i = 0; i < 132; i++) x = C.add(x, C.maxShift);
check(C.stateToIndex(x) === originalIndex, 'exact 132-step translation orbit');
const folded = C.contraction(C.embed512(x), 0.8);
C.measure(folded);
check(folded.error === 1, 'initial error exactly one');
for (let i = 0; i < 62; i++) C.fold(folded);
check(folded.error <= 1e-6, '62 anchored contractions reach epsilon');
check(folded.k === 62, 'exact iteration count');
check(folded.history.every((e, i) => i === 0 || e <= folded.history[i-1]), 'monotone residual');
check(C.required(0.8, 1e-6) === 62, 'analytical K');
const immediate = C.contraction(C.embed512(x), 0);
C.fold(immediate);
check(immediate.error === 0, 'lambda zero exact step');
let threw = false;
try { C.stateFromIndex(C.CARD); } catch { threw = true; }
check(threw, 'out-of-range state rejected');
const ctr = C.auditBasic();
check(ctr >= 30, 'built-in CTR checks');
console.log('PASS ' + assertions + ' numerical assertions and ' + ctr + ' CTR checks');
console.log('Inward contraction: iterations=' + folded.k + ' target_error=' + folded.error + ' mse=' + folded.mse);
