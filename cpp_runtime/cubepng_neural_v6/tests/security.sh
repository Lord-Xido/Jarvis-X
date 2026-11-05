#!/usr/bin/env bash
set -euo pipefail
BIN="$1"; SRC="$2"; T="$(mktemp -d)";trap 'rm -rf "$T"' EXIT
printf 'DO-NOT-TOUCH\n' > "$T/existing.jx6"
if "$BIN" encode "$SRC" "$T/existing.jx6" --mode file --strategy fixed >/dev/null 2>&1; then echo 'existing archive overwritten' >&2;exit 1;fi
grep -qx 'DO-NOT-TOUCH' "$T/existing.jx6"
"$BIN" encode "$SRC" "$T/data.jx6" --mode file --strategy fixed --max-mib 1 >/dev/null
printf 'KEEP\n' > "$T/existing.png"
if "$BIN" decode "$T/data.jx6" "$T/existing.png" >/dev/null 2>&1; then echo 'existing image overwritten' >&2;exit 1;fi
grep -qx KEEP "$T/existing.png"
if "$BIN" decode "$T/data.jx6" "$T/too-small.png" --max-mib 0 >/dev/null 2>&1; then echo 'invalid bound accepted' >&2;exit 1;fi
printf 'foreign-data' > "$T/data.png.partial"
"$BIN" decode "$T/data.jx6" "$T/data.png" >/dev/null
cmp -s "$SRC" "$T/data.png"
grep -qx 'foreign-data' "$T/data.png.partial"
if find "$T" -type f -name '*.partial.*' | grep -q .; then echo 'unremoved staging files' >&2;exit 1;fi
echo 'PASS: no-overwrite, owner-only unique staging, size limits, exact decode'
