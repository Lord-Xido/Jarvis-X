# JARVIS-X Ω³ v8 — 3D multimedia reference VM

Bounded CPU implementation. Native C++17 codec plus the existing PyTorch coordinate field and token ANN. No network or external model required. v7 source assets were used unchanged. Python 3.10+, NumPy, Pillow, PyTorch CPU; FFmpeg/FFprobe with libx264rgb; C++ compiler and zlib development headers.

## Build and run

```sh
g++ -std=c++17 -O3 -Wall -Wextra core.cpp -lz -o core
python hybrid.py encode image generated/generated.png sample.s3h
python hybrid.py decode sample.s3h sample.png
python hybrid.py scan sample.s3h 0 0 0
python generate_and_pack.py new_media --frames 240
python run8.py legacy_packets
python refine.py legacy_packets
python prune.py legacy_packets
```

The last three scripts reproduce the original million-iteration experiment, complete legacy-media comparison, 1000 gradient updates and generation, and additional pruning comparisons. The `verification.json` includes the recorded independent holdout test and pruning summary; rerunning individual experiments may overwrite portions of the consolidated report. `run8.py` and `refine.py` write to results/ and generated/. Original v7 packets are supplied in legacy_packets/. Output files included here are examples, not required to execute the CLI. Timings are host-specific. `engine.py`, `independent.py`, `neural.py`, `scan_volume.py` are preserved v7 code (legacy scanner is S3M7 only; use hybrid.py scan for S3H8).

## Native core

Block arrays are in (z,y,x,c) order. Six periodic neighbours within each block are cached; no cross-block halo exchange. For reference block p, motion m=roll(p,+1,x) and a=rounded average of six neighbours. Quantized predictor = (w0*p+w1*m+w2*a+8)>>4, with nonnegative weights summing to 16. Mutate by transferring one weight unit; compare actual serialized record cost, not just residual entropy. The baseline considers raw, zero-run RLE spatial/temporal/motion and optionally zlib alternatives. The ANN candidate and incumbent are selected only if smaller, including the three-byte weight overhead. A weight mutation is accepted only if it strictly improves both current incumbent and baseline for the current block. This is local online search, not a global optimum. Every iteration decodes the selected and candidate packet and compares both byte-for-byte to target, plus selected CRC32. The native core uses 12 fixed 64-bit phase instructions (OP 4 / descriptor 58 / routing 2), decoded in a switch. Functional shards are logical and sequential. It does not run arbitrary external bytecode or execute the neural generator directly in this ISA.

S3B8: 24-byte little-endian header (magic + z,y,x,c,record count); records `<BII` (mode, payload length, CRC32), three weights only in modes 4/8, payload. Modes 0 raw, 1 RLE raw, 2 RLE temporal XOR, 3 RLE x-motion XOR, 4 RLE ANN XOR, 5–8 zlib versions of 1–4. RLE token high bit is zero run, low seven bits length minus one; otherwise literal run. The independent Python decoder in verify8.py uses separate arithmetic. For benchmark, the native driver exports only first 256 records of each modality, but compares selected and candidate in all million iterations. Counters are actual native loop counters. Blocks from real v7 media are sampled deterministically and replayed cyclically; image samples are flattened spatial bytes; other modalities use first channel only and 4^3 blocks (including padded / repeated blocks). Full regression includes all channels, 4x32x32 or singleton-depth image blocks, padding removed after decoding. Tile ordering places temporal blocks together for each spatial tile. The full source video is v7's 320x180 decoded RGB preview, not native 4K. Source audio/text use byte-plane geometry, not physical wave propagation.

## Hybrid codec and scanner

S3H8 is an envelope (magic, JSON length, JSON, complete payload) containing SHA256 of the payload and original tensor. `hybrid.py` creates and verifies both v7 and v8 candidates; chooses the smaller complete envelope including metadata; verifies decoded tensor before atomic replacement. No guarantee of compression below raw file size, and a wrapper adds bytes. Original containers/metadata/alpha/variable timing should be preserved using `file` mode. Video mode decodes RGB, exports CFR lossless RGB video without embedded audio; WAV preserves PCM samples; combined AV preview uses lossy AAC. SHA256 and CRC are integrity checks, not authentication or sandbox security. The decoder loads all data in memory; do not treat this as hardened untrusted-input tooling. The scanner reads any valid tensor coordinate and displays byte addresses, binary channel values, and illustrative LOAD8 words (not executed by this scanner).

## Safe pruning

Native CLI optimization flag 1 evaluates all candidates; flag 2 skips ANN when baseline serialized size is <= 12 + minimal possible ANN payload (ceil(n/128) for RLE only, 1 conservatively when zlib is enabled). The ANN record header is 12 bytes. This cannot reject a strictly better ANN packet. The mutation RNG is advanced identically. The six extra million-iteration-equivalent runs (three repeats per policy, one million blocks per combined modality sweep) compare summary fields and first-256 packet hashes for each modality; full native verification of selected packets still runs. The performance numbers are sums of per-modality medians and not full multimedia throughput. Skipped candidates are not counted as independently reconstructed.

## Neural generation

The original coordinate field is fine-tuned for 1000 additional SGD/Adam updates, with 10 validation checkpoints. A checkpoint is accepted when aggregate validation MSE decreases and each modality remains within 102% of its original validation MSE. A separate seed (8802, 24576 samples) checks the refined and original checkpoint after selection. Targets are procedural light fields, tones and sphere occupancy, not photoreal or natural language training corpora. The token ANN remains unchanged. New image, 240x256x144 RGB video, PCM audio, text and 40^3 geometry are generated, encoded, independently decoded, exported and re-ingested. The video/audio preview uses AAC for the muxed audio; PCM is verified separately. The text model can repeat training sentences; no general text-to-video, cross-modal semantic alignment, GPU acceleration, 4K real-time or exabyte-scale execution is established.
