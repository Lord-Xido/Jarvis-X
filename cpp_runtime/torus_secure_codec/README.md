# Jarvis-X secure 3D toroidal file codec (C++20)

**Status: verified Linux/POSIX prototype; not independently cryptographically audited.**

This module turns the toroidal 3D bitwise idea into a functional reversible
file transport: `plaintext -> 8×8×8 XOR-lifting -> AES-256-GCM encryption ->
file`, and `file -> GCM authentication/decryption -> inverse XOR-lifting ->
byte-identical plaintext`.

The toroidal aspect is *periodic-boundary XOR coupling on an 8×8×8 lattice*; the transform
is a reversible XOR lifting rather than a learned neural network, a torus
embedding, an error-reducing ANN, or a compressor. The former lossy toroidal
ANN remains a separate experimental model; this codec does not claim to preserve
its latent features or throughput claims.

## Build and run (Linux/macOS with POSIX APIs and OpenSSL)

```sh
cmake -S cpp_runtime/torus_secure_codec -B build/torus-secure -DCMAKE_BUILD_TYPE=Release
cmake --build build/torus-secure -j 2
ctest --test-dir build/torus-secure --output-on-failure
./build/torus-secure/torus_secure_codec keygen ./local-secret-key.bin
./build/torus-secure/torus_secure_codec seal ./input.bin ./input.jxt3 ./local-secret-key.bin
./build/torus-secure/torus_secure_codec open ./input.jxt3 ./recovered.bin ./local-secret-key.bin
cmp ./input.bin ./recovered.bin
```

Save the key offline and **never commit or upload keys**. Key loss prevents
recovery. Key files contain exactly 32 raw bytes and are created exclusively
with POSIX mode 0600. Existing destination files are never intentionally overwritten.

Run integration tests with `python3 cpp_runtime/torus_secure_codec/test_secure_codec.py build/torus-secure/torus_secure_codec`.

## Wire format v1

| Field | Length | Authentication |
| --- | ---: | --- |
| ASCII `JXTOR3D1` | 8 bytes | AAD |
| Original byte count | 8 bytes, little endian | AAD |
| Random nonce | 12 bytes | AAD |
| XOR-lifted 512-byte tiles, GCM-encrypted | ceil(n / 512) * 512 | ciphertext |
| GCM authentication tag | 16 bytes | tag |

A fresh, CSPRNG-generated 96-bit GCM nonce is generated per `seal` operation.
The 28-byte header is authenticated as GCM additional authenticated data. The
last source tile is zero padded before reversible lifting and encryption; the
plaintext size in the authenticated header removes padding after decryption.
The format is **not compressed** and grows by between 44 and 555 bytes.

The decoder authenticates the full ciphertext **before writing any plaintext**,
then decrypts in a second pass, rechecks the tag, and exclusively publishes a
mode-0600 output via an atomic POSIX hard link. Failed authentication removes
the temporary output and never publishes a destination. Both passes must read
the same stable source file: a same-UID adversary might change an input between
passes and inspect the second-pass temporary plaintext before rejection. This is
a local prototype, not hardened against hostile same-UID processes, untrusted
symlinked paths or concurrent mutation. Production use needs descriptor-bound
file access and independent security review.

## Security boundary

- Confidentiality/integrity comes **only** from standard OpenSSL AES-256-GCM,
  not the reversible 3D transform. Do not reuse nonce/key pairs, or treat XOR
  lifting as cryptography.
- There is **no password-based key derivation**, key rotation, recipient
  exchange, metadata privacy, compression, authentication of the file *name*,
  or forward secrecy. File size rounded to 512-byte boundaries is visible.
- No secrets, ciphertext fixtures or locally generated keys are committed.
- Large files stream in bounded 512-byte buffers; no dense 3D tensor memory.
- Integration tests cover empty, boundary and random inputs, wrong keys,
  corrupted headers/nonces/ciphertext/tags, truncation, and refusing overwrite.