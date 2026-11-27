# S3D-BCx bounded 8K³ reference VM

Experimental implementation of the proposed recursive self-encoding pipeline, with a logical 8192³ address space and a deliberately reduced active RGB volume. Requires Node.js 18+; no third-party dependencies.

```sh
node experimental/s3dbcx_bounded_8k3/reference.cjs
```

## Execution contract

- 64-bit instruction words encoded as eight bytes, in order: opcode, phase, shard, Lambda, Theta, descriptor high, descriptor low, immediate. This is an experimental subset, not a complete implementation of the original ISA. The interpreter records the actual word at each dispatch; the descriptor/phase/mask fields are metadata, not enforced hardware policy.
- Logical coordinate address: z*2^26 + y*2^13 + x, with each coordinate in [0,8191]. The self-test covers selected boundary and interior addresses. No dense 8192³ allocation occurs; the active reference cube is restricted to side lengths 4–32 divisible by four.
- RGB8 input is copied, high nibble encoded as unsigned nonnegative Q16.16, low nibble retained as XOR residual. The immutable initial latent and residual reconstruct every source byte. This is a toy predictive/quantization codec, not a learned or necessarily compressed representation.
- AFIRE computes a seven-point fixed-point stencil of the initial latent; PERM contracts the current state toward that known target by half the integer error (rounding the correction away from zero); RENORM bounds the result. Four logical z shards use a shared immutable input and disjoint output ranges. No physical parallelism or trained ANN is claimed.
- Before committing each candidate, verify source reconstruction, numeric bounds and monotone L1 distance to the fixed target. Compare max absolute state delta to zero. Branch to output on fixed point; trap on exhausted iteration budget or invalid candidate.
- AES-256-GCM is provided by Node's crypto module. Each adapter instance generates a fresh ephemeral key and an eight-byte nonce prefix; a 32-bit monotonic counter prevents nonce reuse under the key. AAD authenticates the iteration and format. Checkpoints are sealed and immediately authenticated in memory before commit. This is not a durable encrypted checkpoint store; key persistence and secure lifecycle are not implemented. An explicitly unsealed reference path is available for testing arithmetic without Node crypto.
- SHA3-256 chains canonical JSON checkpoint content and commit instruction; plaintext audit hashes can be compared across runs despite randomized encryption. This is an audit digest, not independent proof of execution.
- After convergence, a further stencil sweep is projected to the first z-plane as RGB8 output. The output is a byte array, not a native framebuffer or MP4. No native Metal, OLED/camera, optical compute, multimedia semantic model, learned weight update, or self-modifying bytecode is implemented.

## Regression coverage

The arithmetic-only self-test was executed in a JavaScript isolate: 8³ voxels, 1,536 channels, 24 iterations, 175 dispatched words, final delta 0, 192 output bytes. It checked deterministic computational replay, bounds rejection, corrupted source residual rejection, budget exhaustion, selected 39-bit coordinate round trips, and HALT reachability.

The Node entry point additionally exercises actual AES-GCM round trips, ciphertext tamper rejection and SHA3 replay. Those Node-specific checks have been authored but were not executed in the JavaScript isolate. Run the command above to verify them in Node or CI. No full-volume throughput claim is made.

## Next steps

Add full descriptor validation, independently specified kernel oracles, persistent checkpoint/key management, native C++/Metal backends, trained modality-specific encoders and decoders, and measured device benchmarks before integration with production runtimes.
