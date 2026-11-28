# JARVIS X Ω³ — trillion-token PDF ANN VM

Experimental reference implementation, not a production language model.

## Run

From this directory with Python 3.10+:

```sh
python -m pip install numpy pymupdf
python jarvis_trillion_ann.py --side 8 --rounds 1000 --out ann_trace.json
python build_carrier.py --out /tmp/jx-ann-carrier
python jarvis_pdf_bitvm.py extract /tmp/jx-ann-carrier/jarvis_trillion_token_ann.pdf /tmp/reconstructed.pdf
cmp /tmp/reconstructed.pdf /tmp/jx-ann-carrier/reconstructed_spec.pdf
```

The builder emits a new compact PDF carrier with source, both bytecode streams, an encoded specification PDF, and verification traces. It does not reproduce the original six-page report byte-for-byte. The original report SHA-256 is `5085c4ed6d9cccc4f4b4ac220932e67e5a50ee66148275c468080b8783159db0` (152385 bytes); the binary is not included in this commit.

## Architecture

Logical torus 1000 x 1000 x 1000, one billion streams at a target 1000 tokens/s each. The target is 10^12 tokens/s, not a benchmark. The reference VM executes a small Morton-indexed torus using integer weights, 16->8->8->16 and 32-symbol output logits. Synchronous state updates use six periodic neighbors. CTR verifies all cells in the initial tick and one rotating cell on subsequent ticks. No learned weights, natural-language generation, training or hardware concurrency at the target scale is demonstrated.

ANN bytecode: `01 10 20 50 52 53 60 61 70 FF` (load, map, encode, ANN, decode, token, verify, commit, recurse, halt).

Bit bytecode: `01 10 20 21 30 31 60 61 70 FF` (load, map, XOR, pack, unpack, inverse, verify, commit, recurse, halt). Bit coordinates are implicit Morton indices; the linear XOR predictor is reversible, not a neural lossless compressor. PDF extraction requires an external interpreter; do not execute untrusted embedded source.

At 1B dense parameters, the approximate weight-matmul demand is 2e21 FLOP/s. Illustrative KV assumptions: 16 layers, 4 KV heads, width 64, 2 bytes/element, 2048 cached tokens. This gives 33.55 PB aggregate KV capacity and 33.55 EB/s uncached KV reads. These are idealized estimates with no communication or attention overhead.

## Regression baseline

Original local run: 512000 symbols in 1000 iterations, 9001 bytecode dispatches, 1511 scalar checks, zero observed reference errors. Token-stream SHA-256: `2e81a7f2c992653af180e43a80cdfb02aafc71fc6a8a7df99313a8b2ef51228c`. Final state SHA-256: `43bb604a3379233e6feaf226125833c3bc394b5c56122aeec3da34b055f7d14a`. Measured original toy-symbol throughput about 692937/s; timing varies by host. Original PDF carrier completed 1000 bit round trips with exact equality and corruption rejection. The new builder independently checks these invariants for the newly generated carrier.
