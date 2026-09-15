# EM-Sig 1000³ — bounded end-to-end emulation

**Status:** Experimental research layer  
**Repository:** `Lord-Xido/Jarvis-X`  
**Execution:** `python -m jarvisx.em_sig_emulation`  
**Evidence class:** deterministic software emulation; not hardware measurement

## 1. Purpose

This module turns the proposed EM-Sig path into an executable, falsifiable software trace:

```text
bits
  -> QPSK symbols
  -> array-factor abstraction
  -> reduced scalar interference field
  -> 6-bit latent state
  -> residual correction
  -> reconstructed bits
```

The logical address space may be described as `1000³`, but the reference run intentionally materializes a much smaller proxy grid. This follows the repository rule that experimental spatial engines remain isolated from the canonical VM until evidence justifies promotion.

The existing electromagnetic research specification is authoritative about the physical boundary: visual or scalar field simulations do not by themselves solve Maxwell's equations or establish measured electromagnetic computation.

## 2. L1 — bit ingest / SERDES arithmetic

`create_world()` generates a seeded binary source. Pairs are mapped to QPSK symbols.

For a configured sample rate of

```text
f_s = 120e9 samples/s
```

and one QPSK symbol per sample with two payload bits per symbol, the modeled serial payload rate is

```text
R = 2 f_s = 240e9 bit/s.
```

Therefore, without an additional parallel-lane multiplier,

```text
1e9 bits  / 240e9 bit/s = 4.1667 ms
1e12 bits / 240e9 bit/s = 4.1667 s
```

The previously proposed `1 Tbit -> 8.33 ns` value is therefore **not** produced by a single 120 GSa/s QPSK lane. Achieving that aggregate time would require an explicitly specified parallel fabric with roughly `1e12 / (8.33e-9 * 240e9) ≈ 5.0e8` such idealized lanes before overheads.

The emulation reports this arithmetic directly instead of labeling it measured throughput.

## 3. L2 — QPSK / carrier parameter

The executable mapping is

```text
00 -> (+1 + j)/sqrt(2)
01 -> (-1 + j)/sqrt(2)
11 -> (-1 - j)/sqrt(2)
10 -> (+1 - j)/sqrt(2)
```

so every ideal constellation point has unit symbol energy.

`carrier_hz` is retained as a model parameter. The reference does **not** emulate:

- a current-steering DAC;
- reconstruction-filter parasitics;
- mixer conversion loss;
- oscillator phase noise;
- a measured phase-lock transient;
- a 5 fs (`0.005 ps`) physical switching/locking event.

Those require device/circuit models or laboratory measurements.

## 4. L3 — 1024-element spatial abstraction

A `32 x 32` square array produces a normalized far-field angular cut:

```text
AF(theta) = sum_m,n exp(j phi_mn(theta)).
```

This is an array-factor calculation, not a full antenna solver.

The inward field is separately represented by a bounded scalar interference proxy

```text
I_proxy(r) = | sum_s exp(j k |r-r_s|) / |r-r_s| |².
```

The proxy is evaluated on the materialized grid and normalized before core sampling.

It is intentionally **not** called Poynting flux. Physical Poynting flow requires electromagnetic fields

```text
S = E x H
```

that satisfy Maxwell's equations together with constitutive relations and boundary conditions.

Likewise, `VoxelAutomataEngine::stepXORDiffusion()` is a useful computational analogy for local interaction, but XOR cellular evolution is not mathematically equivalent to Maxwell superposition. `FabrikSolver` is an iterative geometric target solver; it can motivate a target-alignment control pattern, but it is not an RF beamforming phase solver.

### Propagation arithmetic

The configured design parameter

```text
0.14 ps/hop * 1000 hops = 140 ps
```

is arithmetically preserved. Its physical validity depends on the actual hop length, medium, group velocity, dispersion and device/interconnect delays.

## 5. L4 — six-bit latent bottleneck

The normalized central `3 x 3 x 3` proxy energy is quantized to

```text
z in {0, ..., 63}.
```

This means **six bits and 64 representable states**.

The latent fixed-point loop moves toward the target six-bit word and records every residual Hamming distance. `max_latent_iterations=64` is a software iteration bound; it must not be reinterpreted as proof that a physical wave literally bounces 64 times.

The latent criterion is

```text
z_(k+1) = z_k
```

only when the numerical state has actually stabilized at the target word.

## 6. L5 — residual correction, BER and active collapse

A seeded flip channel creates an initial reconstruction error. The decoder surrogate computes

```text
r_i = b_i XOR b_hat_i
```

and updates only the residual active set.

For each correction cycle the report records:

```text
errors
observed_ber = errors / tested_bits
active_count
active_fraction
```

The trace must be measured from the actual materialized bitstream; the values are not hard-coded to a desired curve.

A zero observed error count means only

```text
observed BER = 0 / N
```

for that finite software run. It does **not** verify a physical BER of `1e-12`. The report therefore also exposes the single-error resolution `1/N` and explicitly marks sub-resolution BER claims as unverified.

The full correction fixed point requires both:

```text
residual_count == 0
```

and

```text
reconstructed_(t+1) == reconstructed_t.
```

Only then is the software trace marked `fixed_point: true`.

## 7. Compression accounting

For the proposed raw logical volume

```text
1000³ voxels * 3 channels * 1 byte = 3,000,000,000 bytes.
```

A six-bit latent plus `10,000` residual values can approach a `~300,000:1` ratio only under an optimistic representation in which residual coordinates are implicit and each residual consumes about one byte.

For an explicitly indexed sparse representation, each `1000³` coordinate needs at least

```text
ceil(log2(1000)) = 10 bits/axis
3 axes = 30 coordinate bits
+ 6 value bits
= 36 bits/node minimum.
```

Thus `10,000` nodes require at least about `45,000` bytes before container/index/checksum overheads, giving an idealized ratio nearer `~66,667:1` rather than `300,000:1`.

Both estimates are emitted by the executable report so representation assumptions remain visible.

## 8. What the emulation verifies

A passing run establishes only the following bounded software properties:

1. deterministic seeded bit generation;
2. four-state unit-energy QPSK mapping;
3. a 1024-element array-factor abstraction;
4. deterministic reduced scalar inward-field computation;
5. six-bit latent quantization and numerical convergence;
6. monotonically shrinking residual error under the configured correction rule;
7. fixed-point detection from actual state equality;
8. transparent rate and compression arithmetic.

It does **not** establish 120 GSa/s fabricated hardware, a 5 fs phase lock, a physical dense `1000³` Maxwell solution, `114 Peta-nodes/s`, `18.5 GPix/s`, physical `BER=1e-12`, or a measured 300,000:1 hardware codec.

## 9. Run

```bash
python -m pip install -e ".[test]"
pytest tests/test_em_sig_emulation.py -q --no-cov
python -m jarvisx.em_sig_emulation \
  --bit-count 65536 \
  --materialized-extent 12 \
  --seed 42 \
  --output artifacts/em-sig-emulation.json
python -m json.tool artifacts/em-sig-emulation.json > /dev/null
```

The generated JSON uses

```text
schema_version = jarvisx.em-sig-emulation.v1
provenance = simulated
```

so downstream tooling cannot silently confuse this reference emulation with laboratory telemetry.
