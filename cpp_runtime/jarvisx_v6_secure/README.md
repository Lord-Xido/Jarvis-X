# JARVIS X v6 — optional authenticated archive envelope

This optional C++17 module integrates the JARVIS X v6 `JX6AEAD1` authenticated encryption layer without replacing the repository's existing 3D codecs or the independent DM3D secure-codec draft PR #383.

It encrypts or decrypts **any binary file**, including a `JX3DSTR3` ANN residual stream from the separate v5 multimedia engine. The 3D byte codec, six-track generator, and 3D swarm **are not bundled into this optional target**; this PR does not claim they run in the repository.

## Build

```sh
cmake -S cpp_runtime -B build/cpp-runtime -DJARVISX_BUILD_V6_SECURE=ON -DJARVISX_BUILD_GL_VISUALIZER=OFF
cmake --build build/cpp-runtime --target jarvisx-v6-secure -j2
ctest --test-dir build/cpp-runtime -R jarvisx-v6-secure-regressions --output-on-failure
```

Standalone build:

```sh
cmake -S cpp_runtime/jarvisx_v6_secure -B /tmp/jx6-build
cmake --build /tmp/jx6-build -j2
ctest --test-dir /tmp/jx6-build --output-on-failure
```

## Secure usage

```sh
./build/cpp-runtime/jarvisx_v6_secure/jarvisx-v6-secure init-key /secure/master.key
./build/cpp-runtime/jarvisx_v6_secure/jarvisx-v6-secure encrypt input.jx3 output.jxe --key-file /secure/master.key
./build/cpp-runtime/jarvisx_v6_secure/jarvisx-v6-secure decrypt output.jxe recovered.jx3 --key-file /secure/master.key
```

Never commit `master.key`. Keep it owner-only and backed up separately.

## Format and security properties

- HKDF-SHA256 derives a per-file AES-256-GCM key from a 32-byte master secret and 16-byte random salt.
- The 44-byte header carries format/version, plaintext length, random salt and nonce prefix; the header is authenticated as AAD.
- Independent 64-KiB chunks have 16-byte authentication tags and index-derived nonces.
- Wrong keys, ciphertext/header/tag modification, truncated streams, trailing data and output collisions are rejected without publishing recovered plaintext.
- The output increases in size by `44 + 16 * max(1, ceil(bytes / 65536))` bytes.
- `inspect` shows **unauthenticated** header information; it cannot establish authenticity.

This reference is **not independently audited**. It does not provide signatures, key rotation, HSM management, replay protection, cross-device hardening or assured temporary-file erasure. The earlier DM3D secure codec PR #383 has a different archive format; interchangeability is not established. Logical 10^18-engine scale and learned multimedia generation are outside this module.
