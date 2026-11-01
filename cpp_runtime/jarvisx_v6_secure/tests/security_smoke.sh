#!/usr/bin/env bash
set -euo pipefail
secure=${1:?expected path to jarvisx-v6-secure executable}
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
fail(){ echo "FAIL: $*" >&2; exit 1; }
should_fail(){ if "$@" >"$tmp/reject.log" 2>&1;then fail "accepted forbidden operation"; fi; }
"$secure" init-key "$tmp/key" >/dev/null
"$secure" init-key "$tmp/other" >/dev/null
[[ $(stat -c %a "$tmp/key") == 600 ]] || fail key_permission
head -c 131329 /dev/urandom >"$tmp/plain"
"$secure" encrypt "$tmp/plain" "$tmp/one.jxe" --key-file "$tmp/key" >/dev/null
"$secure" encrypt "$tmp/plain" "$tmp/two.jxe" --key-file "$tmp/key" >/dev/null
! cmp -s "$tmp/one.jxe" "$tmp/two.jxe" || fail nonce_reuse
"$secure" decrypt "$tmp/one.jxe" "$tmp/recovered" --key-file "$tmp/key" >/dev/null
cmp -s "$tmp/plain" "$tmp/recovered" || fail round_trip
should_fail "$secure" decrypt "$tmp/one.jxe" "$tmp/wrong" --key-file "$tmp/other"
[[ ! -e "$tmp/wrong" ]] || fail wrong_key_publication
python3 - "$tmp" <<'PY'
import sys
from pathlib import Path
p=Path(sys.argv[1]); b=bytearray((p/'one.jxe').read_bytes())
for name, index in [('header',39),('nonce',41),('cipher',50),('tag',44+65536)]:
    x=b.copy(); x[index]^=1; (p/f'{name}.jxe').write_bytes(x)
(p/'trunc.jxe').write_bytes(b[:-1]); (p/'extra.jxe').write_bytes(b+b'X')
(p/'empty').write_bytes(b'')
PY
for c in header nonce cipher tag trunc extra; do
  should_fail "$secure" decrypt "$tmp/$c.jxe" "$tmp/illegal-$c" --key-file "$tmp/key"
  [[ ! -e "$tmp/illegal-$c" ]] || fail "published modified $c"
done
"$secure" encrypt "$tmp/empty" "$tmp/empty.jxe" --key-file "$tmp/key" >/dev/null
"$secure" decrypt "$tmp/empty.jxe" "$tmp/empty.out" --key-file "$tmp/key" >/dev/null
[[ ! -s "$tmp/empty.out" ]] || fail empty_roundtrip
should_fail "$secure" decrypt "$tmp/one.jxe" "$tmp/recovered" --key-file "$tmp/key"
should_fail "$secure" encrypt "$tmp/plain" "$tmp/plain" --key-file "$tmp/key"
echo 'PASS: key permissions, randomized ciphertext, exact recovery, tamper/wrong-key/truncation/overwrite rejection and empty files'
