#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
cxx="${CXX:-g++}"
"$cxx" -std=c++20 -O2 -Wall -Wextra -Wpedantic -Werror \
  "$root/jarvisx_trillion3_inward_quine.cpp" -o "$work/jx_inward"
"$work/jx_inward" --selftest
"$work/jx_inward" --quine > "$work/regenerated.cpp"
cmp "$root/jarvisx_trillion3_inward_quine.cpp" "$work/regenerated.cpp"
"$cxx" -std=c++20 -O2 -Wall -Wextra -Wpedantic -Werror \
  "$work/regenerated.cpp" -o "$work/jx_regenerated"
"$work/jx_regenerated" --quine | cmp - "$work/regenerated.cpp"
"$work/jx_inward" --run 12 > "$work/first.log"
"$work/jx_inward" --run 12 > "$work/repeat.log"
cmp "$work/first.log" "$work/repeat.log"
for bad in '-1' '100001' 'NaN'; do
  if "$work/jx_inward" --run "$bad" > /dev/null 2>&1; then
    echo "FAIL: invalid step count accepted: $bad" >&2
    exit 1
  fi
done
echo "PASS: source identity, 10000 probes, contraction, invalid inputs and deterministic replay ($cxx)"
