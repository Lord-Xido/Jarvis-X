#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
g++ -O2 -std=c++20 -Wall -Wextra -Wpedantic tests/test_core.cpp src/jx_core.cpp -o /tmp/jarvisx_native_tests
/tmp/jarvisx_native_tests
python3 -m unittest -v tests.test_http
