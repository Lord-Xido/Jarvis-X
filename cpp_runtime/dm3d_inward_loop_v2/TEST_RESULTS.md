# DM3D v2 local validation

Environment: Debian Linux x86-64, GCC 14.2, CMake, C++20. This report records tests performed in the execution container; it is not an independent benchmark or proof of AI capabilities.

- `cmake --build build -j 4`: PASS
- `ctest --test-dir build --output-on-failure`: PASS (1/1 CTest target; internal tests cover multiple codecs and file types)
- `dm3d selftest`: PASS, including all five 3D candidate modes, one-pass budget, 120-bit spatial index, sparse edge write/read, empties, zeros, random input, text, WAV/PNG/MP4 fixtures, tamper detection.
- `dm3d loop sample.wav ... --threads 2 --folds 12 --passes 5`: PASS; 4 tiles, 20 independently verified 3D proposals, exact 16,078-byte recovery.
- `dm3d loop sample.mp4 ... --threads 4 --folds 12 --passes 5`: PASS; one tile, five independently verified candidates, exact 3,163-byte recovery.
- Independently generated PNG, WAV, text and MP4 inputs: byte-for-byte comparison (`cmp`) PASS after encode/decode.
- Sparse voxel at `(999999999999,999999999999,999999999999)` set to byte `213`, retrieved as `213`.

No throughput, training, cloud deployment or full virtual-cube materialization claims are made.
