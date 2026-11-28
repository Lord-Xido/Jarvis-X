JARVIS-X Computational PDF companion / reference implementation v1.0
Extract attachments with a PDF reader that supports embedded files, or use the ZIP companion.
Python 3.9+ (selftest uses random.randbytes); g++ with C++17 for the native test.
Run: python3 selftest.py
Run: python3 jarvisx.py encode input.bin output.jx --trace trace.jsonl --workers 4
Run: python3 jarvisx.py decode output.jx restored.bin
Run: python3 jarvisx.py bytecode program.bin
Run: g++ -std=c++17 -O2 -Wall -Wextra -pedantic jx_native.cpp -o jx_native
Run: ./jx_native program.bin tile.bin residual.bin  # tile must contain 1-512 bytes
Run: python3 jarvisx.py generate input.bin preview.pgm --kind pgm
Run: python3 jarvisx.py generate input.bin preview.wav --kind wav
Use different input/output paths. The runtime never executes ingested source code.
The JX format stores all file bytes, not a semantic representation. No encryption, authentication, GPU execution or cloud sync. Do not use for confidential data without external controls. Local output files are overwritten on successful encode/decode. Container and trace commits are separate atomic replacements, not one filesystem transaction. The native C++ VM executes a single tile and verifies the raw codec only. Compression, streaming, hashing and fault injection are in the Python host. Worker threads bound outstanding tiles but do not guarantee parallel speedup (CPython GIL). Cross-version zlib byte identity is not guaranteed. Generation is toy byte prediction, not language/image/video generation in the modern semantic sense.
