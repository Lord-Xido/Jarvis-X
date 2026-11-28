# JARVIS-X Omega3 bytecode VM (v1)

A dependency-free Node.js reference implementation of the recursive 3D processor. It uses the JX3D v1 container format and a bounded, fixed-program 64-bit bytecode dispatcher.

## Run (Node.js 18+)

    node engines/omega3_bytecode/vm.js selftest
    node engines/omega3_bytecode/vm.js encode input.bin output.jx3d 48
    node engines/omega3_bytecode/vm.js decode output.jx3d restored.bin

Output paths must not already exist. Inputs are bounded to 1 MiB; padded bytes times iterations must not exceed 16 MiB, with 1–1000 iterations. The CLI defaults to one iteration; use 48 for exhaustive configuration search on a sufficiently small input.

## Geometry and invariants

The logical domain is (Z / 10^12 Z)^3, with 120-bit packed addresses (40 bits per axis). Voxel index maps to 8^3 tiles in tile-major order, then to a nine-bit local Morton index. The full volume is not allocated. The active host representation is a padded byte array, not a distributed toroidal mesh.

Forward lifting XORs pairs on each of three axes at three levels; optional byte rotation adds a reversible second shear. The inverse reverses axis/level order. A zero-run codec or raw payload is selected, with a 16-byte little-endian JX3D header containing magic, version, transform, codec, reserved byte, original length, and FNV-1a checksum. FNV is not authentication.

Each iteration decodes the serialized candidate and compares it byte-for-byte against the immutable original before commit or feedback. Only a strictly shorter valid container is accepted; failures abort. Final emission independently decodes the best container again. This is representation optimization, not semantic multimodal inference.

## Instruction format

Big-endian 8-byte instruction: OP:8 DST:8 SRC:8 AUX:8 IMM:32. Register operands and control flow are validated against the canonical bounded program; this is not a general-purpose arbitrary-program VM. The canonical program for 1000 iterations (zero-based PC):

| PC | Word | Meaning |
|---:|---|---|
|0|01000000000003E8| LOAD, initialize baseline |
|1|0201000000000000| MAP3D, pad |
|2|0302010100000000| ENCODE, policy-selected transform |
|3|0403020000000000| PACK, serialize |
|4|0504030000000000| UNPACK |
|5|0605040000000000| DECODE |
|6|0706050000000000| VERIFY immutable original |
|7|0807030600000000| OPTIMIZE, gated commit |
|8|0901050000000001| RECURSE to PC 1 until bound |
|9|0A08070500000000| EMIT best container and decoded bytes |

The instruction count is 8N+2 for N iterations. The feedback address is PC 1 (MAP3D), ensuring re-mapping on every cycle.

## Adaptive controller

A minimal one-input, 48-output linear neural policy learns a running mean of normalized container savings using squared-error updates with learning rate 1/count. It explores all 48 configurations once, then exploits the best estimated score. Ties select the earliest configuration. This is a deterministic per-input selector, not the proposed deep 16→64→64→48 network, and it does not establish cross-input generalization. Disable adaptive selection via run(input, program(N), false).

## Verification

The JavaScript core self-test ran before commit: 358 assertions; 1000 iterations; 8002 dispatched instructions; 8192-byte structured fixture; 7760-byte baseline; 110-byte best container; one accepted improvement; FNV-1a ccc92dc5; exact round-trip. Includes all 49 transforms on seven lengths, boundary addresses, malformed opcode, checksum corruption and empty input. The ratio is fixture-specific. Repository CI and Node CLI integration are separate checks.

No hardware throughput, cryptographic security, trained multimedia ANN, or physical 10^36-node claim is made.
