# Perfected DM-vOmegaXi+ Transactional Firmware

This package converts the original pseudo-assembly into a bounded, auditable reference design.

## Included

- `dm_voxi_firmware.asm` — perfected transactional ROM source.
- `ISA.md` — machine model, operand convention, invariants, and instruction semantics.
- `reference_vm.py` — deterministic executable reference model.
- `test_reference_vm.py` — commit, rollback, budget, authorization, and retry tests.

## Governing transition

```text
candidate = encode -> measure -> quantize -> stage -> decode -> validate
committed_next = candidate when valid, otherwise committed_previous
```

## Important capability boundary

The `1000^1000 GB` manifold is treated only as a symbolic virtual address domain. Physical allocation and execution are explicitly bounded. The reference codec uses measured reconstruction tolerance rather than claiming universal zero loss.

## Run

```bash
python3 reference_vm.py 0.1 -0.2 0.3 -0.4
python3 -m unittest -v test_reference_vm.py
```

A committed transaction returns a provenance receipt containing input, candidate, and committed-state SHA-256 digests.
