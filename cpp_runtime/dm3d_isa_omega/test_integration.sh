#!/usr/bin/env bash
set -euo pipefail
exe="${1:?Pass path to dm3d_pipeline}"
root="$(mktemp -d)"
trap 'rm -rf "$root"' EXIT
"$exe" --e2e "$root/e2e"
cmp "$root/e2e/source.raw" "$root/e2e/restored.raw"
test -s "$root/e2e/raycast.ppm"
test -s "$root/e2e/pipeline.bytecode.bin"
test -s "$root/e2e/trained-model.q16"
# Empty inputs, arbitrary-sized inputs, cross-thread deterministic output.
: > "$root/empty.raw"
"$exe" --stream-encode "$root/empty.raw" "$root/empty.dm3ds2" 2
"$exe" --stream-decode "$root/empty.dm3ds2" "$root/empty.restored" 1
cmp "$root/empty.raw" "$root/empty.restored"
python3 - "$root/mixture.raw" <<'PY'
import random, sys
r = random.Random(4321)
with open(sys.argv[1], 'wb') as f:
    f.write(bytes(r.randrange(256) for _ in range(197703)))
    f.write(b'\0' * 32768)
    f.write(bytes(range(256)) * 1884)
    f.write(b'\x31\x04\x11')
PY
"$exe" --stream-encode "$root/mixture.raw" "$root/one.dm3ds2" 1
"$exe" --stream-encode "$root/mixture.raw" "$root/four.dm3ds2" 4
cmp "$root/one.dm3ds2" "$root/four.dm3ds2"
"$exe" --stream-decode "$root/four.dm3ds2" "$root/restored.raw" 3
cmp "$root/mixture.raw" "$root/restored.raw"
# Tampering must fail and must not leave a decoded output file behind.
python3 - "$root/four.dm3ds2" "$root/corrupt.dm3ds2" <<'PY'
import sys
payload = bytearray(open(sys.argv[1],'rb').read())
# A frame-data byte, after the fixed-length header and frame prefix.
payload[280] ^= 0x81
open(sys.argv[2],'wb').write(payload)
PY
if "$exe" --stream-decode "$root/corrupt.dm3ds2" "$root/tainted.raw" 2; then
    echo 'ERROR: corruption was accepted' >&2; exit 1
fi
test ! -e "$root/tainted.raw"
test ! -e "$root/tainted.raw.partial"
echo 'PASS: DM3D v2 self-hosted VM, trained codec, streaming, deterministic threading, disk persistence, raster, damage rejection'
