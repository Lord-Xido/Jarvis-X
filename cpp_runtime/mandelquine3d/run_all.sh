#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
cxx="${CXX:-g++}"
"$cxx" -O2 -std=c++20 -Wall -Wextra -pedantic dm_mandelquine.cpp -o dm_mandelquine
"$cxx" -O2 -std=c++20 -Wall -Wextra -pedantic dm_inward_verifier.cpp -o dm_inward_verifier
./dm_mandelquine demo demo_outputs
./dm_mandelquine quine > replica.cpp
cmp dm_mandelquine.cpp replica.cpp
./dm_inward_verifier audit_outputs dm_mandelquine.cpp
printf 'PASS: Mandelquine3D demo, source Quine, and inward mathematical verifier\n'
