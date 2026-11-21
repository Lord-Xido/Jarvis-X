# DM-vOmegaXi+ QUINE v2.0 - bounded research reference

Status: isolated executable research simulator; not a replacement for canonical JARVIS-X / DM-vOmegaXi state, an OS self-modifier, an autonomous deployment system, or a measured 1000x accelerator.

Provenance: operational interpretation of the user-supplied "DM-vOmegaXi+ QUINE v2.0 - SELF-REPLICATING / AUTO-EVOLVING" pseudocode (14 November 2026). The supplied assembly-like listing is an ISA proposal, not executable machine code or an established exact quine.

## Operational cycle

1. LOAD_SELF / LEN_SELF / HASH_SELF: reconstruct the complete UTF-8 source from an embedded template plus a 3-parameter integer genome. No source file is read by the --quine path. SHA-256 tracks parent, candidate, and committed source. Plain hashes do not authenticate authorship.
2. REPLICATE: --quine prints exactly one full current program. --streams 1000000 means a million LOGICAL stream addresses, without actual broadcasting, million-fold copying, network I/O, or process spawning.
3. INTROSPECT: a fixed synthetic 8x8x8 3D field undergoes 2x2x2 mean pooling to a 4x4x4 latent field. The decoder predicts pixels using per-axis finite differences.
4. MUTATE: a seeded PRNG selects an axis and adjusts its quantized decoder coefficient by +/-1, clamped to [0,16]. Only genome DATA is changed, not arbitrary executable code or instruction bytes.
5. SELECT / COMMIT / ROLLBACK: accept the proposed genome only if reconstruction MSE decreases by more than 1e-12. Otherwise preserve the parent. Record SHA-256 identities and state transitions in deterministic JSONL.
6. EVOLVE: iterate for 0..1000 cycles. A generated descendant can itself reproduce its full source exactly.
7. TURBO: 1000x remains a requested target. No speed gain is fabricated. Optional --bench reports elapsed time but not a speedup estimate.

Mathematics:

- z = E(X): average each nonoverlapping 2x2x2 block.
- Xhat(p) = z(block(p)) + sum_axis[(g_axis/16) * finite_gradient_axis(z) * local_offset_axis(p)].
- L(g) = mean_p (X(p) - Xhat(p))^2.
- g_(t+1) = candidate if L(candidate) < L(g_t) - 1e-12, otherwise g_t.
- At each accepted mutation, L(g_(t+1)) <= L(g_t). This is a local quality invariant, not global optimality, true self-modifying execution, neural learning, or 1000x acceleration.

BACKPROP, UPDATE_W, AUTO_TUNE, PRUNE, FUSE, UNROLL, VMATMUL, and PULSE_XI are reserved proposal-level opcodes and ARE NOT implemented by this simulator. The original infinite JMP REPLICATE is replaced with bounded iteration.

## Run (Python 3.10+; standard library)

    cd experimental/dm_voxi_quine_v2
    python3 dm_quine.py --selftest
    python3 dm_quine.py --quine > regenerated.py
    cmp dm_quine.py regenerated.py
    python3 dm_quine.py --steps 80 --seed 707 --streams 1000000 \
      --emit-child evolved.py --trace-out trace.jsonl
    python3 evolved.py --quine > evolved-reproduced.py
    cmp evolved.py evolved-reproduced.py
    python3 -m unittest discover -s . -p 'test_*.py' -v

Output paths use exclusive file creation; existing files are not overwritten. The executable makes no network requests, starts no child processes, uses no eval or unsafe deserialization, and never modifies itself in place. Running it does not deploy code or merge a GitHub PR.

## Integration and evidence boundaries

- cpp_runtime/trillion3_inward_quine already implements an independent C++20 exact-source quine; this experiment adds bounded parameter evolution to a different standalone quine.
- src/jarvisx/quine_rom_bridge.py already implements a lossless QROM data codec; this experiment is a lossy 3D predictive demonstration, without an exact residual bitstream or ANN.
- The submitted SELF_DATA lists only five literal instructions and cannot restore the complete supplied listing. This version embeds the entire generating template.
- The proposed XOR instruction mutates unvalidated bytes and conflates mutation value, locus and bit, so this reference mutates a typed 3D decoder genome instead.
- A hash is a source identity, not the CALC_FITNESS value, and cannot substitute for an objective loss measurement.
- Requested one-million streams and 1000x turbo are not measured compute resources or performance outcomes in this version.
- The fixed integer genome and synthetic sample do not prove general learning or generalization.
- This is no source-level access control; SHA-256 detects source mismatch but a modified author can change both code and hash check.

## Validation

A path-scoped GitHub Actions workflow checks byte-exact parent/child quine identity, deterministic replay, spatial codec geometry, monotonic accepted reconstruction MSE, trace integrity, truncation of disallowed iteration/stream counts, and exclusive-write protection. Keep the PR in draft until CI results and code review are complete.

Existing canonical code, PRs, branches, and main remain untouched.
