#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:-$ROOT/runtime_output}"
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
cmake -S "$ROOT" -B "$OUT/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$OUT/build" -j2
python3 "$ROOT/bootstrap_pixelrom.py" --vm "$OUT/build/jx_vm" --outdir "$OUT/build"
ctest --test-dir "$OUT/build" --output-on-failure
"$OUT/build/jx_pixelrom" --decode "$OUT/build/JARVIS_X_PIXEL_ROM_100.png" "$OUT/recovered.jxsf"
"$OUT/build/jx_vm" --run-control --control "$OUT/recovered.jxsf" --archive "$OUT/tiles.jxar" \
  --ply "$OUT/active_tiles.ply" --active 256 --workers 8 --iterations 3 --seed 42
"$OUT/build/jx_vm" --verify --control "$OUT/recovered.jxsf" --archive "$OUT/tiles.jxar"
python3 "$ROOT/verify_archive.py" "$OUT/recovered.jxsf" "$OUT/tiles.jxar"
python3 "$ROOT/decode_pixelrom.py" "$OUT/build/JARVIS_X_PIXEL_ROM_100.png" --out "$OUT/recovered_python.jxsf"
cmp "$OUT/recovered_python.jxsf" "$OUT/recovered.jxsf"
printf '\nPIXEL ROM END-TO-END PASS\nOutput: %s\n' "$OUT"
