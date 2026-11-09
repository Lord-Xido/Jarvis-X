#!/usr/bin/env bash
set -euo pipefail
BIN="${1:-./build/moagi_trace1000x}"
OUT="${2:-demo_output}"
mkdir -p "$OUT"
"$BIN" selftest
"$BIN" generate 1000 1000 1000 4 4 4 64 1000000 checker "$OUT/seed.tile"
"$BIN" run "$OUT/seed.tile" "$OUT/trace.csv" "$OUT/final.tile" "$OUT/plans.csv"
"$BIN" replay "$OUT/seed.tile" "$OUT/trace.csv" "$OUT/final.tile"
echo "Optional physical output (1,000,000,000 bytes):"
echo "  $BIN materialize $OUT/final.tile $OUT/final.raw"
