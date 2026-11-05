#!/usr/bin/env bash
set -euo pipefail
BIN="${1:-./cubepng_neural}"
SRC="${2:-fixture.png}"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
"$BIN" self-test
for mode in file pixels; do
  for strategy in fixed adaptive learned linear-hybrid neural hybrid; do
    archive="$TMP/${mode}_${strategy}.jx6"
    "$BIN" encode "$SRC" "$archive" --mode "$mode" --strategy "$strategy" --level 3 >/dev/null
    "$BIN" verify "$SRC" "$archive" >/dev/null
    "$BIN" decode "$archive" "$TMP/${mode}_${strategy}.png" >/dev/null
    "$BIN" verify "$TMP/${mode}_${strategy}.png" "$archive" >/dev/null
    if [[ "$mode" == file ]]; then cmp -s "$SRC" "$TMP/${mode}_${strategy}.png"; fi
  done
done
"$BIN" encode "$SRC" "$TMP/good.jx6" --mode pixels --strategy neural --level 3 >/dev/null
python3 - "$TMP/good.jx6" "$TMP" <<'PY'
from pathlib import Path
import sys
src=bytearray(Path(sys.argv[1]).read_bytes()); out=Path(sys.argv[2])
assert src[:7]==b'JX3STR3' and src[72]==9
for name,offset in [('weights',97),('payload',-17),('kernel',72)]:
    b=src.copy();b[offset]^=0x40
    (out/f'{name}.jx6').write_bytes(b)
(out/'truncated.jx6').write_bytes(src[:-13])
(out/'trailing.jx6').write_bytes(src+b'EXTRA')
(out/'old_magic.jx6').write_bytes(b'JX3STR2\0'+src[8:])
PY
for kind in weights payload kernel truncated trailing old_magic; do
  if "$BIN" decode "$TMP/${kind}.jx6" "$TMP/invalid.png" >/dev/null 2>&1; then
    echo "FAIL: $kind corruption accepted" >&2;exit 1
  fi
  if [[ -e "$TMP/invalid.png" ]]; then echo "FAIL: bad archive committed output" >&2;exit 1; fi
done
"$BIN" encode "$SRC" "$TMP/deterministicA.jx6" --mode pixels --strategy neural --level 3 >/dev/null
"$BIN" encode "$SRC" "$TMP/deterministicB.jx6" --mode pixels --strategy neural --level 3 >/dev/null
cmp -s "$TMP/deterministicA.jx6" "$TMP/deterministicB.jx6"
if "$BIN" encode "$SRC" "$SRC" >/dev/null 2>&1; then echo 'FAIL: overwrote source' >&2;exit 1;fi
echo 'PASS: 12 round-trips, determinism, metadata/payload/tail corruption rejection, safe output commit'
