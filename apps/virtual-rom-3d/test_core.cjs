'use strict';
const assert = require('node:assert/strict');
const test = require('node:test');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const html = fs.readFileSync(path.join(__dirname, 'index.html'), 'utf8');
const context = {module: {exports: {}}, Uint8Array, Uint32Array, DataView};
vm.runInNewContext(html.match(/<script id="rom-core">([\s\S]*?)<\/script>/)[1], context);
const C = context.module.exports;

test('all inline scripts parse', () => {
  for (const match of html.matchAll(/<script[^>]*>([\s\S]*?)<\/script>/g)) new vm.Script(match[1]);
});

test('logical addresses are independent across windows and include the final byte', () => {
  const rom = new C.SparseROM({mode: 'zero'});
  const addresses = [0, 4095, 4096, C.WINDOW_BYTES, C.LOGICAL_BYTES - 1];
  addresses.forEach((address, i) => rom.writeByte(address, i + 1));
  addresses.forEach((address, i) => assert.equal(rom.readByte(address), i + 1));
  assert.equal(rom.readByte(C.WINDOW_BYTES * 2), 0);
  assert.equal(rom.residentBytes, 4 * C.PAGE_BYTES);
  assert.deepEqual(rom.readRange(C.LOGICAL_BYTES - 1, 1), Uint8Array.of(5));
});

test('read snapshots are isolated and returning a page to its image releases it', () => {
  const rom = new C.SparseROM({mode: 'zero'});
  rom.writeByte(10, 255);
  const read = rom.readRange(0, 128); read[10] = 9;
  assert.equal(rom.readByte(10), 255);
  rom.writeByte(10, 0);
  assert.equal(rom.pageCount, 0);
});

test('cross-page capacity rejection is atomic, including existing pages and revision', () => {
  const rom = new C.SparseROM({mode: 'zero', limitBytes: 4096});
  rom.writeByte(0, 7);
  const revision = rom.revision;
  assert.throws(() => rom.writeRange(4095, Uint8Array.of(8, 9)), /limit exceeded/);
  assert.equal(rom.revision, revision);
  assert.equal(rom.readByte(0), 7);
  assert.equal(rom.readByte(4095), 0);
  assert.equal(rom.readByte(4096), 0);
  assert.equal(rom.residentBytes, 4096);
});

test('invalid reads, writes, ranges and image resets leave the image unchanged', () => {
  const rom = new C.SparseROM({mode: 'zero'}), revision = rom.revision;
  for (const address of [-1, .5, NaN, Infinity, C.LOGICAL_BYTES]) {
    assert.throws(() => rom.readByte(address)); assert.throws(() => rom.writeByte(address, 3));
  }
  for (const byte of [-1, .1, 256, NaN]) assert.throws(() => rom.writeByte(0, byte));
  assert.throws(() => rom.readRange(C.LOGICAL_BYTES - 1, 2));
  assert.throws(() => rom.readRange(0, C.WINDOW_BYTES + 1));
  assert.throws(() => rom.writeRange(0, [1,2,3]));
  assert.throws(() => rom.reset('unknown', 42));
  assert.throws(() => rom.reset('random', -1));
  assert.throws(() => new C.SparseROM({limitBytes: 3}));
  assert.equal(rom.mode, 'zero'); assert.equal(rom.revision, revision);
});

test('procedural images replay deterministically without page allocation', () => {
  for (const mode of ['structured', 'random', 'zero', 'sine']) {
    const a = new C.SparseROM({mode, seed: 42}), b = new C.SparseROM({mode, seed: 42});
    for (const base of [0, 8192, C.WINDOW_BYTES, C.LOGICAL_BYTES - 128]) assert.deepEqual(a.readRange(base, 128), b.readRange(base, 128));
    assert.equal(a.pageCount, 0);
  }
  const a = new C.SparseROM({mode:'random', seed:1}), b = new C.SparseROM({mode:'random', seed:2});
  assert.notDeepEqual(a.readRange(0, 128), b.readRange(0, 128));
});

test('Shannon entropy has the correct constant and uniform alphabet limits', () => {
  assert.equal(C.entropy(new Uint8Array(1024)), 0);
  assert.equal(C.entropy(Uint8Array.from({length: 1024}, (_, i) => i % 256)), 8);
  assert.equal(C.entropy(Uint8Array.of(0, 255, 0, 255)), 1);
});

test('CRC and raw packet match an independent published CRC32 check vector', () => {
  const source = Uint8Array.from(Buffer.from('123456789'));
  assert.equal(C.crc32(source), 0xcbf43926);
  const expected = Uint8Array.from([86,82,77,49,0,0,0,0,9,0,0,0,0x26,0x39,0xf4,0xcb,...source]);
  assert.deepEqual(C.encode(source), expected);
  assert.deepEqual(C.decode(expected), source);
});

test('RLE and raw round trips cover packet boundaries without mutating sources', () => {
  for (const length of [0,1,2,3,127,128,129,130,131,259,4097]) {
    const fixtures = [new Uint8Array(length), Uint8Array.from({length}, (_, i) => i % 256), Uint8Array.from({length}, (_, i) => Math.floor(i / 7) % 256)];
    for (const source of fixtures) {
      const before = source.slice(), packet = C.encode(source);
      assert.equal(C.verify(source, packet), true); assert.deepEqual(source, before);
      assert.deepEqual(C.decode(packet), source);
      const padded = new Uint8Array(packet.length + 10); padded.set(packet, 5);
      assert.deepEqual(C.decode(padded.subarray(5, 5 + packet.length)), source);
    }
  }
});

test('full 2 MiB windows round trip and report actual header-inclusive sizes', () => {
  for (const mode of ['zero', 'random', 'structured']) {
    const source = new C.SparseROM({mode}).readRange(0, C.WINDOW_BYTES), packet = C.encode(source);
    assert.equal(C.verify(source, packet), true);
    assert.ok(packet.length <= C.WINDOW_BYTES + 16);
    if (mode === 'zero') { assert.equal(packet[4], 1); assert.equal(packet.length, 32280); }
    if (mode === 'random') { assert.equal(packet[4], 0); assert.equal(packet.length, C.WINDOW_BYTES + 16); }
  }
});

test('malformed packets fail before restoration', () => {
  const packet = C.encode(new Uint8Array(1000));
  for (const offset of [0,3,4,5,6,7,12,packet.length - 1]) {
    const corrupt = packet.slice(); corrupt[offset] ^= 0x80;
    assert.throws(() => C.decode(corrupt));
  }
  assert.throws(() => C.decode(packet.subarray(0, 15)));
  assert.throws(() => C.decode(packet.subarray(0, packet.length - 1)));
  const bomb = packet.slice(); new DataView(bomb.buffer).setUint32(8, 0xffffffff, true);
  assert.throws(() => C.decode(bomb), /decoded length/);
  const trailing = new Uint8Array(packet.length + 1); trailing.set(packet);
  assert.throws(() => C.decode(trailing));
  assert.throws(() => C.verify(Uint8Array.of(1), C.encode(Uint8Array.of(2))), /byte mismatch/);
});

test('restore into a distant window retains independent source addresses', () => {
  const rom = new C.SparseROM({mode:'zero'}), source = Uint8Array.from({length:8193},(_,i)=>i%256);
  const restored = C.decode(C.encode(source)); rom.writeRange(C.WINDOW_BYTES + 4095, restored);
  assert.deepEqual(rom.readRange(C.WINDOW_BYTES + 4095, source.length), source);
  assert.deepEqual(rom.readRange(4095, source.length), new Uint8Array(source.length));
});

test('sampling is unique, bounded and spans all three axes', () => {
  for (const count of [4096,32768,65536]) {
    const offsets = C.sampleOffsets(count);
    assert.equal(offsets.length, count); assert.equal(new Set(offsets).size, count);
    assert.ok(offsets.every(i => i >= 0 && i < C.WINDOW_BYTES));
    assert.ok(new Set(Array.from(offsets, i => i % 256)).size >= 16);
    assert.ok(new Set(Array.from(offsets, i => Math.floor(i / 65536))).size >= 16);
  }
  assert.throws(() => C.sampleOffsets(2e6));
});

test('grid is centred and nebula radius tracks all nine bit populations', () => {
  const first = C.gridPosition(0), last = C.gridPosition(C.WINDOW_BYTES - 1);
  first.forEach((value,i) => assert.ok(Math.abs(value + last[i]) < 1e-12));
  for(let bits=0;bits<=8;bits++) {
    const value=(1<<bits)-1, p=C.nebulaPosition(0,4096,value);
    assert.ok(Math.abs(Math.hypot(...p)-(22+5*bits/8))<1e-10);
    assert.equal(C.popcount(value),bits);
  }
});

test('rotation response is elapsed-time based and bobbing derivatives are consistent', () => {
  const expected=1-Math.pow(.95,60);
  for(const hz of [30,60,120]) {let x=0;for(let i=0;i<hz;i++)x=C.smoothAngle(x,1,1/hz);assert.ok(Math.abs(x-expected)<1e-12);}
  const t=.7,h=1e-5,before=C.verticalMotion(t-h),after=C.verticalMotion(t+h),now=C.verticalMotion(t);
  assert.ok(Math.abs((after.position-before.position)/(2*h)-now.velocity)<1e-9);
  assert.ok(Math.abs((after.velocity-before.velocity)/(2*h)-now.acceleration)<1e-9);
});
