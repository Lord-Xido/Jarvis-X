# Perfected DM-vOmegaXi+ Transactional Firmware

This package converts the original pseudo-assembly into a bounded, auditable reference design.

## Included

- `src/fragments/dm_voxi_firmware.asm/*.part` — ordered, line-preserving fragments of the perfected transactional ROM source.
- `ISA.md` — machine model, operand convention, invariants, and instruction semantics.
- `reference_vm.py` — deterministic executable reference model.
- `test_reference_vm.py` — commit, rollback, budget, authorization, and retry tests.
- `demo_output.json` — canonical committed transaction receipt.
- `MATRIX_1000KB3.md` — formal 1000³ logical-matrix extension, runtime alignment table, measurable targets, and capability boundaries.
- `SHA256SUMS.txt` — integrity manifest for source fragments, generated ROM, documentation, VM, tests, and demo output.

## Operational browser runtime

The executable sparse-matrix implementation is under `../../apps/dmvx-matrix/`. It operationalizes the matrix specification with:

- a deterministic logical `1000³` domain backed by at most 4,096 resident cells;
- a 64-value encoder, Q16.16 latent state, decoder, and measured reconstruction RMSE;
- finite, bounds, and resource-budget validation with atomic commit or rollback;
- leaky residual correction memory and inward active-set reduction;
- versioned CRC-32 ROM persistence, local autoboot, and binary export;
- SHA-256 receipt chaining and measured browser telemetry;
- six Node.js invariant tests and GitHub Pages deployment workflows.

Run its invariant suite with:

```bash
node --test ../../apps/dmvx-matrix/test_core.mjs
```

The browser runtime is an executable reference implementation, not a claim of a physically resident billion-cell tensor, electromagnetic bytecode, zero latency, or measured `1000^n` acceleration.

## Source transport

The authoritative ROM source is stored as ordered text fragments so connector transport can be verified independently. Materialize the original source with:

```bash
cat src/fragments/dm_voxi_firmware.asm/*.part > dm_voxi_firmware.asm
```

The generated file must have SHA-256:

```text
8df61acd5d40b9d4281b7c5f43a22074575b6983021f64fcdc0bea22b0760b7a
```

## Governing transition

```text
candidate = encode -> measure -> quantize -> stage -> decode -> validate
committed_next = candidate when valid, otherwise committed_previous
```

## Important capability boundary

The `1000^1000 GB` manifold is treated only as a symbolic virtual address domain. Physical allocation and execution are explicitly bounded. The reference codec uses measured reconstruction tolerance rather than claiming universal zero loss.

## Run

```bash
cat src/fragments/dm_voxi_firmware.asm/*.part > dm_voxi_firmware.asm
python3 reference_vm.py 0.1 -0.2 0.3 -0.4
python3 -m unittest -v test_reference_vm.py
sha256sum -c SHA256SUMS.txt
```

A committed transaction returns a provenance receipt containing input, candidate, and committed-state SHA-256 digests.
