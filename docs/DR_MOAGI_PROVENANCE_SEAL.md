# Dr Moagi Provenance Seal

## Status

JARVIS X now has a boot-time provenance gate for the browser runtime. The gate
uses a canonical release payload, SHA-384 component digests, a detached
ECDSA-P384 signature, and a separately declared public-key trust anchor.

The committed key is explicitly a **development trust anchor**. It demonstrates
the complete verification and recovery mechanism, but it must be replaced by a
key generated and controlled by Dr Matladi Maxwell Moagi before production or
legal provenance claims are made.

No private signing key is stored in this repository.

## Authenticity invariant

A release is accepted only when all checks pass:

\[
\operatorname{Authentic}(JX)
\iff
\operatorname{Verify}_{pk}(P,\sigma)
\land
\operatorname{SHA384}(P)=d_P
\land
\operatorname{SHA384}(ROM)=d_{ROM}
\land
\operatorname{SHA384}(ISA)=d_{ISA}
\land
\operatorname{SHA384}(M)=d_M.
\]

The current seal covers:

- the canonical Dr Moagi manifest;
- the initial binary city ROM used by the neural echo demo;
- the initial instruction-registry version and execution order;
- the release identifier and signing-key identifier.

## Boot sequence

```text
READ_TRUST_ANCHOR
-> CANONICALIZE_RELEASE_PAYLOAD
-> VERIFY_PAYLOAD_SHA384
-> VERIFY_MANIFEST_SHA384
-> VERIFY_ROM_SHA384
-> VERIFY_INSTRUCTION_SHA384
-> VERIFY_PUBLIC_KEY_FINGERPRINT
-> VERIFY_ECDSA_P384_SIGNATURE
-> AUTHENTIC or RECOVERY
```

`examples/jarvisx-echo/app-sealed.js` performs this sequence before dynamically
importing `app.js`. A verification failure disables query execution and runtime
refinement and exposes the failed checks in recovery mode.

## Files

| File | Responsibility |
|---|---|
| `dr-moagi-seal.mjs` | Detached signature, signed payload, and development public key |
| `provenance.mjs` | Canonical encoding, SHA-384 hashing, signature verification |
| `app-sealed.js` | Boot gate and recovery-mode transition |
| `provenance.test.mjs` | Positive and tamper-detection tests |

## Mutable versus immutable domains

The release seal protects the initial identity boundary. It does not prohibit
normal runtime adaptation.

Immutable release domain:

- root identity and operator manifest;
- initial ROM digest;
- initial instruction-registry identity;
- signature algorithm and key identifier;
- recovery policy.

Mutable runtime domain:

- neural weights;
- query heat;
- cache contents;
- aliases;
- evidence-gated instruction ordering;
- usage-optimized ROM ordering;
- rendering LOD and execution telemetry.

Runtime mutations are valid only under the verified root policy. They do not
become newly signed official releases unless a new payload is built and signed
with the owner-controlled private key.

## Production key ceremony

1. Generate an ECDSA P-384 key pair on an offline or hardware-backed device.
2. Store the private key outside the repository and browser bundle.
3. Export only the public JWK and its independently recorded fingerprint.
4. Build the canonical release payload from reviewed release artifacts.
5. Compute SHA-384 digests for the manifest, ROM, and instruction identity.
6. Sign the canonical payload with ECDSA P-384 and SHA-384.
7. Commit the detached signature and public trust anchor through a reviewed PR.
8. Record the public-key fingerprint in an external location, signed release,
   protected tag, legal deposit, or other independently controlled registry.
9. Re-run all provenance and runtime tests before publishing.

## Boundary

A repository-contained public key provides an operational software trust
boundary, especially when combined with protected branches and signed release
artifacts. Strong authorship provenance requires the public-key fingerprint to
also be anchored outside the mutable repository.
