# Universal Bitcode Permeation Runtime

## Implemented result

Jarvis-X now has one deterministic container for arbitrary digital artifacts:

```text
raw bits + representation contract
  -> bounded chunking
  -> reversible per-chunk codec
  -> canonical manifest
  -> SHA-256 + Merkle integrity
  -> .jxbi
  -> verified reconstruction
```

The implementation is dependency-free and lives in
`src/jarvisx/universal_bitcode.py`. The command-line entry point is
`jarvisx-universal-bitcode`.

The runtime accepts text, documents, images, audio, video, 3D scenes, source or
machine code, model files, archives, and unknown binary data. It does not need to
understand the semantics of every format in order to transport the bytes without
loss. Meaning remains bound to an explicit or inferred representation contract.

## Computational model

For a modality \(m\), let \(B_m\in\{0,1\}^*\) be its finite byte string and
\(C_m\) its representation contract. The implemented compiler is

\[
\mathcal E(B_m,C_m)=(H,M,P),
\]

where \(H\) is the fixed header, \(M\) is canonical JSON metadata, and \(P\) is
the concatenated encoded payload. The decoder is required to satisfy

\[
\mathcal D(\mathcal E(B_m,C_m))=(B_m,C_m).
\]

The CLI `cycle` command closes the recursive loop:

\[
Q_0=\mathcal E(B,C),\qquad
Q_1=\mathcal E(\mathcal D(Q_0)).
\]

Canonical serialization gives the executable fixed-point condition

\[
Q_1=Q_0,
\]

while the byte-level reality gap is

\[
\Delta_B(B,\hat B)
=
\sum_i [B_i\ne\hat B_i]
+
\left|\,|B|-|\hat B|\,\right|.
\]

A successful cycle requires \(\Delta_B=0\) and identical SHA-256 digests for
the original and reconstructed artifacts.

## Representation contract

Every container carries:

| Field | Meaning |
|---|---|
| `media_kind` | text, document, image, audio, video, scene-3d, code, model, archive, or binary |
| `media_type` | normalized MIME-style type/subtype token |
| `format_name` | bounded format identifier such as `png`, `wav`, `glb`, or `raw` |
| `source_name` | informational source name; never used as a decode output path |
| `schema` | optional application-level schema identifier |
| `metadata` | bounded canonical JSON object |

Detection uses a small deterministic table of file signatures, followed by
extension hints and bounded UTF-8/JSON recognition. Detection is a type hint, not
content authentication. Callers can provide the contract explicitly.

This distinction is central:

```text
opaque bytes != interpreted meaning
opaque bytes + representation contract -> typed information
```

## Container layout

The version-1 header is a 128-byte big-endian structure:

| Field | Width |
|---|---:|
| magic `JXUBIR1\0` | 8 bytes |
| version | 2 bytes |
| flags | 2 bytes |
| manifest length | 4 bytes |
| original byte length | 8 bytes |
| encoded payload length | 8 bytes |
| original SHA-256 | 32 bytes |
| manifest SHA-256 | 32 bytes |
| payload SHA-256 | 32 bytes |

The manifest records the representation contract and an ordered chunk table.
Each chunk binds its raw/encoded offsets, sizes, codec, raw digest, and encoded
digest. Offsets and indexes must be contiguous; missing, overlapping, reordered,
or trailing data is rejected.

## Hierarchical integrity

For raw chunks \(b_0,\ldots,b_{n-1}\), each leaf begins with

\[
d_i=\operatorname{SHA256}(b_i).
\]

The container then constructs a domain-separated binary Merkle tree. Leaves are

\[
\ell_i=\operatorname{SHA256}(0x00\,\|\,d_i),
\]

and parent nodes are

\[
p=\operatorname{SHA256}(0x01\,\|\,\ell_L\,\|\,\ell_R).
\]

An odd final node is duplicated. The empty-artifact root is SHA-256 of the empty
byte string. The root binds the hierarchy, while the whole-artifact, manifest,
payload, and per-chunk hashes localize corruption.

## Reversible coding

Each chunk independently chooses one of two version-1 codecs:

- `zlib` level 9 when it strictly reduces the chunk size;
- `identity` otherwise.

The encoded payload therefore never exceeds the original payload because of a
codec choice. Container metadata still adds overhead, so this is not a claim that
all files become smaller.

Decompression is bounded by the descriptor's declared raw size. The decoder
rejects truncated streams, concatenated streams, unused data, output beyond the
declared size, and any digest mismatch.

## Resource and failure contract

Default hard limits are:

| Resource | Default maximum |
|---|---:|
| original artifact | 256 MiB |
| canonical manifest | 8 MiB |
| metadata object | 64 KiB |
| source name | 4 KiB UTF-8 |
| chunks | 16,384 |
| raw or encoded chunk | 4 MiB |

Applications can inject a stricter `BitcodeBudget`. Parsing checks header sizes
before slicing or decoding. No parser path performs network access, executes
embedded code, imports the encoded artifact, or writes to a source-derived path.

The CLI writes through a same-directory temporary file, flushes and fsyncs it,
and then atomically replaces the destination. Existing outputs require
`--force`; input and output paths cannot be identical.

## Commands

Encode with inferred representation metadata:

```bash
jarvisx-universal-bitcode encode image.png image.jxbi
```

Run the complete recursive verification cycle:

```bash
jarvisx-universal-bitcode cycle animation.glb animation.jxbi
```

Inspect without decompressing:

```bash
jarvisx-universal-bitcode inspect animation.jxbi
```

Perform full reconstruction verification:

```bash
jarvisx-universal-bitcode verify animation.jxbi
```

Decode atomically:

```bash
jarvisx-universal-bitcode decode animation.jxbi animation.restored.glb
```

An explicit contract can be supplied when inference is insufficient:

```bash
jarvisx-universal-bitcode encode weights.bin weights.jxbi \
  --media-kind model \
  --media-type application/x-example-model \
  --format-name example-model \
  --schema example/v1 \
  --metadata '{"tensor_layout":"external-schema"}'
```

## ROM Forge source audit

The supplied ROM Forge package included useful architectural primitives—typed
multimedia ingress, reversible chunk transport, symbolic parsing, rendering,
state evolution, and API/deployment sketches—but much of the submitted code is
explicitly placeholder material. One notebook is not valid notebook JSON, several
C++ files contain only placeholder comments, the deployment stack identifies its
router and moral filter as stubs, and a presence prototype declares a trillion
iteration loop.

Those files are not imported or silently promoted into canonical execution.
`reference/rom_forge_legacy/manifest.json` records the SHA-256 and disposition of
all 16 supplied top-level artifacts. The safe, testable common behavior is
implemented anew in the universal bitcode runtime.

## Boundary of the claim

Implemented:

- typed arbitrary-byte ingestion;
- deterministic versioned binary IR;
- lossless bounded encode/decode;
- chunk-level adaptive reversible compression;
- canonical fixed-point cycling;
- corruption, malformed-input, and resource-limit rejection;
- a library API and atomic command-line interface.

Not implemented or implied:

- semantic parsing of every declared media format;
- learned latent compression;
- image-to-3D, text-to-audio, or other cross-modal generation;
- execution of embedded programs or model checkpoints;
- truth, safety, intelligence, or provenance from a digest;
- compression of every input into fewer total bytes;
- unbounded recursion, zero-time execution, or infinite physical storage.

Future semantic frontends and output backends can attach above this IR, but they
must retain the canonical VM authority boundary: decoded or generated data is a
candidate, not permission to execute or commit external effects.
