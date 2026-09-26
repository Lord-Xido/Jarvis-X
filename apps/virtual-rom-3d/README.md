# 3D Virtual ROM

Runnable browser demonstration derived from Dr Matladi Maxwell Moagi's supplied
`3D_1GB_Virtual_ROM_Engine.html`. It preserves the grid/nebula, byte inspection
and inward visual vocabulary while making the address, storage and codec
contracts executable. It is an isolated browser data plane, not the canonical
Jarvis-X VM or a trained autoencoder.

## Run

Open `index.html` directly, or serve the repository:

```bash
python -m http.server 8000
# Open http://localhost:8000/apps/virtual-rom-3d/
```

The single HTML file contains the memory model, codec, UI and Canvas renderer.
Canvas works offline. Three.js r128 loads opportunistically from its pinned CDN
for WebGL cubes; a blocked CDN or lost WebGL context leaves Canvas and all data
controls operational. No server, package installation or credentials are needed
to use the app. Save encoded windows explicitly; edits are otherwise volatile.

## Implemented contracts

| Layer | Contract |
|---|---|
| Address space | Exactly `0x00000000..0x3fffffff`, or 1 GiB of logical byte addresses |
| Image | Pure address/seed generator: structured, random, zero or sine |
| Mutable overlay | Independent 4 KiB pages, at most 16 MiB of resident page payload |
| Transaction | Stage touched pages, validate final capacity, then commit all or none |
| Active window | One 2 MiB copy; 512 possible windows |
| Hex inspector | 128 bytes per bank, 8,388,608 banks, exact byte addresses |
| Codec | Versioned VRM1 run-length or raw packet, CRC32 and byte-exact verification |
| Graphics | 4,096 / 32,768 / 65,536 sampled bytes; Canvas capped at 4,096 |
| Telemetry | Window Shannon entropy, actual packet size, measured render FPS and operation time |

Page payload is not total process memory: the inspection copy, staged pages,
codec buffers, graphics buffers and browser allocations are additional. The
default image is procedurally defined; the app neither loads nor allocates an
arbitrary dense 1 GiB file. Reads can cover the whole logical space, but a single
read/write operation is bounded to 2 MiB. A write may temporarily stage at most
513 pages before the 16 MiB committed-page limit is checked. Matching a page
back to its procedural image releases that page.

Presets replace the entire procedural image and clear edits. `Write byte` edits
one absolute address. `Encode + verify window` leaves memory unchanged. Saved
packets hold one window's bytes; restoring writes those bytes into the currently
selected window, after complete decode/CRC validation and a page-capacity check.
The packet has no embedded destination address. Corruption or quota rejection
leaves memory and its revision unchanged. CRC32 detects accidental corruption;
it is not cryptographic authentication.

## Geometry and motion

For an offset `i` in the active window:

```text
x = i % 256
y = floor(i / 256) % 256
z = floor(i / 65536)
i = x + 256*y + 65536*z
p = (0.18*(x-127.5), 0.18*(y-127.5), 0.45*(z-15.5))
```

The `z` centring corrects the source artifact's half-layer displacement. Samples
are stratified in all three lattice axes; a linear stride would alias columns.
Each sample is one exact addressed byte, not the complete contents of a cluster.
Cube scale is `0.25 + 0.85*popcount(byte)/8`. Nebula radius is
`22 + 5*popcount(byte)/8 + 0.0005*(sampleIndex % 70)`, producing nine bit-count
shells. The sphere uses the original spiral angular construction. The mapping
is a visualization and is not invertible as a byte codec.

Dragging changes target angles by `0.008` radians per pixel. Rotation easing is
elapsed-time based: `alpha = 1 - 0.95^(60*dt)`. Vertical motion is
`y(t)=0.25*sin(1.5*t)`, with velocity `0.375*cos(1.5*t)` and acceleration
`-0.5625*sin(1.5*t)`. Reduced-motion preference starts playback paused. Pausing
freezes the animation clock while direct inspection and rotation remain usable.

## VRM1 packet

All multi-byte header fields are unsigned little-endian integers.

| Offset | Bytes | Meaning |
|---|---:|---|
| 0 | 4 | ASCII `VRM1` (format version 1) |
| 4 | 1 | Mode: `0` raw, `1` RLE |
| 5 | 3 | Reserved, must be zero |
| 8 | 4 | Decoded byte length, at most 2 MiB |
| 12 | 4 | IEEE CRC32 of decoded bytes |
| 16 | variable | Payload |

RLE tokens `0..127` introduce `token+1` literal bytes. Tokens `128..255`
introduce one byte repeated `(token & 127)+3` times. The encoder chooses raw
storage unless the RLE payload is strictly shorter. The decoder rejects unknown
flags, excessive lengths, truncated tokens, overrun, trailing data and CRC
mismatch. Encoder output is never larger than input plus the 16-byte header.

`D(E(B)) = B` is checked byte by byte before the UI marks encoding verified.
Displayed encoded/raw percentage includes the header; incompressible inputs can
therefore exceed 100%. Window entropy is `-sum(p*log2(p))` in bits/byte, not a
compression prediction or a measure of the entire logical image.

## Integration and evidence

`window.getVirtualROMTelemetry()` returns a fresh snapshot with schema
`jarvisx.virtual-rom.v1`, logical/window/resident bytes, selected address/value,
image seed/revision, window entropy, rendered sample count, backend, motion
state, measured timing and an optional verified codec receipt. Navigating to a
different window or changing bytes invalidates the codec receipt.

The original artifact's 2 MiB modulo aliasing, destructive pseudo-LZ button,
heuristic entropy score and unmeasured 60 FPS claim are not carried forward.
This app does not execute opcodes, train model weights, inspect host RAM or
promote browser state into canonical Jarvis-X authority.

## Validation

Dependency-free core checks exercise the exact inline implementation:

```bash
node --test apps/virtual-rom-3d/test_core.cjs
```

Browser checks use Playwright 1.62.1 and Three.js 0.128.0. They route the pinned
Three.js source locally for WebGL and block it for offline profiles:

```bash
PLAYWRIGHT_MODULE=/path/to/node_modules/playwright \
THREE_JS_PATH=/path/to/node_modules/three/build/three.min.js \
SCREENSHOT_DIR=/tmp/virtual-rom-screenshots \
node apps/virtual-rom-3d/test_browser.cjs
```

Coverage includes first/final addresses, cross-window independence, atomic quota
rejection, malformed packets, full-window transport, deterministic replay,
geometry, frame-rate-independent rotation, responsive layouts, keyboard input,
save/restore, direct-file offline use and WebGL context-loss recovery. Headless
browser success is functional evidence, not an iPhone or production GPU benchmark.

The named Dr Moagi attribution follows
[`docs/attribution/DR_MOAGI_EPHEMERAL_NOTION_FRAMEWORK.md`](../../docs/attribution/DR_MOAGI_EPHEMERAL_NOTION_FRAMEWORK.md).
