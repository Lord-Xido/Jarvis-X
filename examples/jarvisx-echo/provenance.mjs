const encoder = new TextEncoder();

export function canonicalize(value) {
  if (Array.isArray(value)) return `[${value.map(canonicalize).join(",")}]`;
  if (value && typeof value === "object") {
    return `{${Object.keys(value)
      .sort()
      .map((key) => `${JSON.stringify(key)}:${canonicalize(value[key])}`)
      .join(",")}}`;
  }
  if (typeof value === "number" && !Number.isFinite(value)) {
    throw new TypeError("Canonical payload cannot contain non-finite numbers");
  }
  return JSON.stringify(value);
}

function bytesToHex(bytes) {
  return [...bytes].map((byte) => byte.toString(16).padStart(2, "0")).join("");
}

function base64ToBytes(value) {
  return Uint8Array.from(atob(value), (character) => character.charCodeAt(0));
}

async function sha384Bytes(bytes) {
  if (!globalThis.crypto?.subtle) throw new Error("Web Crypto API is unavailable");
  return new Uint8Array(await globalThis.crypto.subtle.digest("SHA-384", bytes));
}

export async function sha384HexBytes(bytes) {
  return bytesToHex(await sha384Bytes(bytes));
}

export async function sha384HexCanonical(value) {
  return sha384HexBytes(encoder.encode(canonicalize(value)));
}

export async function verifyDrMoagiSeal({ seal, trustAnchor, romBuffer, instructionManifest }) {
  const checks = {};
  try {
    if (!seal?.payload || !trustAnchor?.publicKeyJwk) {
      throw new TypeError("Seal and trust anchor are required");
    }
    if (!(romBuffer instanceof ArrayBuffer)) throw new TypeError("ROM must be an ArrayBuffer");

    checks.magic = seal.payload.magic === "DRMOAGI-ROM-SEAL";
    checks.keyId = seal.payload.keyId === trustAnchor.keyId;
    checks.algorithm = seal.payload.algorithm === trustAnchor.algorithm;
    checks.manifestDigest = await sha384HexCanonical(seal.payload.manifest) === seal.payload.digests.manifest;
    checks.romDigest = await sha384HexBytes(new Uint8Array(romBuffer)) === seal.payload.digests.rom;
    checks.instructionDigest = await sha384HexCanonical(instructionManifest) === seal.payload.digests.instructions;

    const payloadBytes = encoder.encode(canonicalize(seal.payload));
    checks.payloadDigest = await sha384HexBytes(payloadBytes) === seal.payloadDigest;

    const publicKey = await globalThis.crypto.subtle.importKey(
      "jwk",
      trustAnchor.publicKeyJwk,
      { name: "ECDSA", namedCurve: "P-384" },
      true,
      ["verify"],
    );
    const spki = new Uint8Array(await globalThis.crypto.subtle.exportKey("spki", publicKey));
    checks.trustAnchorFingerprint = await sha384HexBytes(spki) === trustAnchor.spkiSha384;
    checks.signature = await globalThis.crypto.subtle.verify(
      { name: "ECDSA", hash: "SHA-384" },
      publicKey,
      base64ToBytes(seal.signatureBase64),
      payloadBytes,
    );

    const authentic = Object.values(checks).every(Boolean);
    return Object.freeze({
      authentic,
      status: authentic ? "AUTHENTIC" : "RECOVERY",
      keyId: seal.payload.keyId,
      payloadDigest: seal.payloadDigest,
      checks: Object.freeze({ ...checks }),
      reason: authentic
        ? ""
        : Object.entries(checks).filter(([, passed]) => !passed).map(([name]) => name).join(", "),
    });
  } catch (error) {
    return Object.freeze({
      authentic: false,
      status: "RECOVERY",
      keyId: seal?.payload?.keyId ?? "unknown",
      payloadDigest: seal?.payloadDigest ?? "",
      checks: Object.freeze({ ...checks }),
      reason: error instanceof Error ? error.message : String(error),
    });
  }
}
