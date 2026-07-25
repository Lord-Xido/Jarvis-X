import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import vm from "node:vm";
import { webcrypto } from "node:crypto";

import { encodeROM, JarvisXRuntime, TraceBus } from "./runtime-core.mjs";
import {
  DR_MOAGI_INSTRUCTION_MANIFEST,
  DR_MOAGI_RELEASE_SEAL,
  DR_MOAGI_TRUST_ANCHOR,
} from "./dr-moagi-seal.mjs";
import { verifyDrMoagiSeal } from "./provenance.mjs";

if (!globalThis.crypto) globalThis.crypto = webcrypto;

async function loadAppCities() {
  const source = await readFile(new URL("./app.js", import.meta.url), "utf8");
  const match = source.match(/const SOURCE_CITIES = (\[[\s\S]*?\n\]);/);
  if (!match) throw new Error("Could not extract SOURCE_CITIES from app.js");
  return vm.runInNewContext(`(${match[1]})`, Object.create(null), { timeout: 1000 });
}

function instructionManifest(runtime) {
  const snapshot = runtime.snapshot().instructions;
  return {
    schema: DR_MOAGI_INSTRUCTION_MANIFEST.schema,
    version: snapshot.version,
    order: snapshot.order,
  };
}

async function verifiedFixture() {
  const cities = await loadAppCities();
  const runtime = new JarvisXRuntime(cities, { trace: new TraceBus(16) });
  const romBuffer = encodeROM(cities);
  return { cities, runtime, romBuffer, manifest: instructionManifest(runtime) };
}

test("detached development seal authenticates the shipped ROM and ISA", async () => {
  const fixture = await verifiedFixture();
  const result = await verifyDrMoagiSeal({
    seal: DR_MOAGI_RELEASE_SEAL,
    trustAnchor: DR_MOAGI_TRUST_ANCHOR,
    romBuffer: fixture.romBuffer,
    instructionManifest: fixture.manifest,
  });
  assert.equal(result.authentic, true);
  assert.equal(result.status, "AUTHENTIC");
  assert.ok(Object.values(result.checks).every(Boolean));
});

test("ROM mutation forces recovery mode", async () => {
  const fixture = await verifiedFixture();
  const tampered = fixture.romBuffer.slice(0);
  const bytes = new Uint8Array(tampered);
  bytes[bytes.length - 1] ^= 0x01;
  const result = await verifyDrMoagiSeal({
    seal: DR_MOAGI_RELEASE_SEAL,
    trustAnchor: DR_MOAGI_TRUST_ANCHOR,
    romBuffer: tampered,
    instructionManifest: fixture.manifest,
  });
  assert.equal(result.authentic, false);
  assert.equal(result.status, "RECOVERY");
  assert.equal(result.checks.romDigest, false);
});

test("instruction-order mutation forces recovery mode", async () => {
  const fixture = await verifiedFixture();
  const changedManifest = {
    ...fixture.manifest,
    order: fixture.manifest.order.slice().reverse(),
  };
  const result = await verifyDrMoagiSeal({
    seal: DR_MOAGI_RELEASE_SEAL,
    trustAnchor: DR_MOAGI_TRUST_ANCHOR,
    romBuffer: fixture.romBuffer,
    instructionManifest: changedManifest,
  });
  assert.equal(result.authentic, false);
  assert.equal(result.checks.instructionDigest, false);
});

test("signature mutation forces recovery mode", async () => {
  const fixture = await verifiedFixture();
  const alteredSignature = `${DR_MOAGI_RELEASE_SEAL.signatureBase64.slice(0, -2)}AA`;
  const result = await verifyDrMoagiSeal({
    seal: { ...DR_MOAGI_RELEASE_SEAL, signatureBase64: alteredSignature },
    trustAnchor: DR_MOAGI_TRUST_ANCHOR,
    romBuffer: fixture.romBuffer,
    instructionManifest: fixture.manifest,
  });
  assert.equal(result.authentic, false);
  assert.equal(result.checks.signature, false);
});

test("trust-anchor fingerprint mutation forces recovery mode", async () => {
  const fixture = await verifiedFixture();
  const result = await verifyDrMoagiSeal({
    seal: DR_MOAGI_RELEASE_SEAL,
    trustAnchor: { ...DR_MOAGI_TRUST_ANCHOR, spkiSha384: "0".repeat(96) },
    romBuffer: fixture.romBuffer,
    instructionManifest: fixture.manifest,
  });
  assert.equal(result.authentic, false);
  assert.equal(result.checks.trustAnchorFingerprint, false);
});
