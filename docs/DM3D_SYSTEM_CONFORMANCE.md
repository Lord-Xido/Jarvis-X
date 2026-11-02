# DM3D ROM system-wide transactional conformance

**Status:** candidate integration, 3 November 2026. This describes one bounded Python
reference runtime, not universal conformance across all Jarvis-X backends.

## Architecture and scope

The existing canonical dependency remains:

`representation -> validation -> execution -> observation -> journaling`

This implementation is a bounded vertical slice of the Dr Moagi operational
auto-encoding/decoding contract in ADR-016 and ADR-017:

1. A ROM instruction materializes an active `64^3` byte tile, never a dense
   `(10^9)^3` field.
2. `ENCODE` builds a quantized `8^3`-byte block-average latent.
3. `DECODE` expands the latent to a block-constant reconstruction.
4. `RESIDUAL` retains signed byte differences; `CORRECT` restores exact bytes.
5. `VERIFY` compares the restored tile to the source before a successful
   ROM execution can promote any candidate state.

The ROM's entire execution operates on isolated copies of the current
resident tile, latent, reconstruction, residual, statistics and program-counter
state. An exception rejects **all** those tentative changes.

### Execution budgets

The standalone `DM3DVM` defaults to:

| Limit | Default | Enforcement |
| --- | ---: | --- |
| Executed instructions | 4,096 | Count *every* opcode, including repeated jumps |
| Simultaneously active source tiles | 8 | Check before allocating any new tile |
| Conservative tile-work sweeps | 128 | Charge heavy transforms and declared refinement passes |
| ROM instruction count | 4,096 | Reject oversized instruction sections before dispatch |

The limits are explicit constructor parameters. CLI `run` exposes
`--max-steps`, `--max-active-tiles` and `--max-tile-sweeps`.
Header version, geometry, code offset and exact instruction-section length
must match the declared ROM v1 format.

`max_tile_sweeps` accounts a conservative full-tile traversal for each
heavy opcode; it is not a hardware-instruction, FLOP, wall-time or energy
budget. For deployments requiring strict CPU and memory isolation, use an
OS-managed subprocess/container with dedicated external quotas.

## Executable end-to-end evidence

The existing `jarvisx.empirical_validation` receipt now includes
`dm3d_rom_transactional_closure`. It:

- creates one deterministic 262,144-byte source tile;
- produces a 512-byte latent, decodes it, preserves 262,144 signed residual
  entries and checks exact source-byte restoration;
- attempts a new tile write followed by `JUMP 1` in a bounded candidate;
- requires exhaustion of the step budget and preservation of the existing
  committed state, statistics, active tile set and program counter;
- reports real resident tiles separately from the `10^27` logical byte
  positions and records the source SHA-256.

The same evidence protocol is invoked by the
`jarvisx-operationalize smoke --json` gate, so the CI operational smoke and
the canonical empirical evidence use one shared oracle.

Run:

```bash
pytest tests/test_dm3d_rom.py tests/test_empirical_validation.py tests/test_operationalize.py -q
python -m jarvisx.empirical_validation --repetitions 4 --octree-max-depth 3 --output artifacts/empirical-validation.json
jarvisx-operationalize smoke --json
```

## Exact capability boundary

```text
virtual space != resident tiles != physically processed voxels
latent bytes != total lossless archive bytes
candidate verification != production security certification
bounded source ROM != universal fixed-point / self-hosted intelligence
```

This patch does **not** assert a 512:1 lossless compression ratio: full signed
residual values and archive metadata must be accounted for. It does not
upgrade the ROM to a trained ANN, introduce physical `10^27`-byte memory,
prove global convergence, or benchmark native C++ / GPU throughput.

## Next interoperability gates

- Assign one typed envelope and versioned serialisation for Python, C++ and
  JavaScript codec/VM adapters.
- Build cross-language golden vectors and mutation-rejection oracles.
- Export canonical state, residual and audit digests with exact schemas.
- Benchmark real workloads with warm-ups, repeat trials, peak RSS and full
  verification overhead.
- Permit authority promotion only after independently reviewed evidence.
