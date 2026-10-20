#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 2
ctest --test-dir build --output-on-failure
./build/dm3d loop examples/generated/sample.wav demo.wav.dm3d demo_restored.wav --threads 2 --folds 12 --passes 5
cmp examples/generated/sample.wav demo_restored.wav
echo "PASS: Exact WAV reconstruction"
