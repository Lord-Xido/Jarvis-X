# ADR-011: Adopt Dr Moagi permeation transport v1 as a bounded software channel

**Status:** Proposed  
**Date:** 2026-08-18  
**Extends:** ADR-010, ADR-003

## Context

The Dr Moagi architecture uses **permeation** to describe externalising an internal state so that another endpoint can reconstruct the same representation. Earlier mathematical descriptions used a 333.33 MHz carrier, scalar Green-function propagation, a mixed monopole/quadrupole angular pattern, and the commands `FOCUS`, `MODULATE`, and `ABSORB`.

Those descriptions must not be interpreted as proof that Jarvis-X radiates a physical electromagnetic field. The repository currently has no RF device driver, SDR backend, antenna model, transmitter power stage, spectrum-control layer, or receiver hardware integration.

The useful executable contract is therefore the information-transport path:

```text
internal state
-> canonical serialization
-> modulation
-> bounded channel model
-> equalization / demodulation
-> digest verification
-> reconstructed state
```

This can be implemented and tested now while keeping physical-RF claims explicitly false.

## Decision

Jarvis-X adopts `jarvisx.dr-moagi-permeation.v1` as a deterministic **software channel simulator and transport contract**.

The reference path is:

```text
Xi*
-> canonical JSON
-> SHA-256 frame identity
-> BPSK symbols
-> scalar 1/r Green-function channel
-> carrier phase e^(ikr)
-> optional deterministic Gaussian noise
-> equalization
-> BPSK demodulation
-> SHA-256 verification
-> JSON reconstruction
-> Pi_Lambda / cloud commit
```

The scalar channel coefficient is

```text
h(r, theta) = Q * A(theta) / (4 pi r) * exp(i k r)
```

with

```text
k = 2 pi / lambda
lambda = c / f
```

and the v1 angular pattern

```text
A(theta) = w0 + w2 * P2(cos(theta))
P2(x) = (3x^2 - 1) / 2.
```

The default numerical parameters are:

```text
f = 333.33 MHz
c = 299,792,458 m/s
Q = 0.941
coherence = 0.967
w0 = 0.6
w2 = 0.4
range = 1 m
```

which produce approximately:

```text
lambda = 0.899386 m
k = 6.986080 rad/m
delay(1 m) = 3.335641 ns
|h|(1 m, aligned axis) = 0.0748824
```

`coherence` is retained as declared transport metadata in v1; it is not silently converted into transmitter power, SNR, or a physical coherence model.

## Command semantics

### `FOCUS Phi [x,y,z]`

Rotates the simulated quadrupole axis. Because the v1 `l=2` pattern uses `P2`, the pattern is symmetric about `+axis` and `-axis`. It is **not** one-sided beamforming. One-sided steering would require a separately accepted phased-array or vector-field model.

### `MODULATE Phi [state]`

Canonicalizes the JSON state and maps each bit to deterministic BPSK:

```text
0 -> -1
1 -> +1
```

The canonical payload digest is bound to the frame before channel propagation.

### `ABSORB Phi`

Equalizes the known software-channel coefficient, demodulates the BPSK symbols, reconstructs bytes, verifies the SHA-256 digest, parses canonical JSON, and rejects corrupted frames.

## Cloud integration

The reference cloud operation is:

```text
permeate-roundtrip.v1
```

with request shape:

```json
{
  "operation": "permeate-roundtrip.v1",
  "input": {
    "state": {"latent": [0, 1, 2]},
    "config": {
      "carrier_hz": 333330000.0,
      "range_m": 1.0,
      "axis": [0.0, 1.0, 0.0],
      "receiver_direction": [0.0, 1.0, 0.0]
    }
  }
}
```

The operation runs inside the ADR-010 job transaction. A transport result becomes authoritative only after the cloud verifier/policy gate commits the job.

Every result includes:

```text
physical_rf = false
```

so simulated carrier parameters cannot be mistaken for actual transmitter activity.

## Required invariants

1. Physical-RF status is explicit and false for this implementation.
2. Payloads are canonicalized before hashing and modulation.
3. The payload digest is verified after demodulation before reconstruction is accepted.
4. Symbol corruption must fail closed.
5. Payload size is bounded before bit/symbol expansion.
6. Channel configuration values are finite and bounded.
7. A channel null is rejected before execution.
8. `FOCUS` rotates only the software angular model.
9. The quadrupole pattern is not described as one-sided beamforming.
10. The cloud operation remains subject to ADR-010 job identity, resource limits, verification, evidence journaling, and commit semantics.

## Non-goals

This ADR does not implement or authorize:

- an SDR or RF transmitter;
- antenna control;
- spectrum selection or regulatory compliance;
- radiated-power accounting;
- hardware beamforming;
- wireless key exchange;
- physical receiver synchronization;
- a Maxwell-equation vector-field solver.

Any future physical-RF adapter requires its own architecture decision, hardware abstraction, safety/resource controls, and jurisdiction-appropriate spectrum compliance.

## Validation

Acceptance requires:

- exact 16D latent round-trip reconstruction in the zero-noise reference case;
- numerical regression checks for wavelength, wave number, 1 m delay, and default 1/r amplitude;
- quadrupole-axis rotation tests;
- digest rejection after symbol corruption;
- payload-budget rejection before symbol expansion;
- authenticated cloud create -> execute -> verify -> committed-event replay;
- CI quality/type/security gates on the stacked branch.

The normative operational specification is maintained in `docs/DR_MOAGI_PERMEATION_TRANSPORT_V1.md`.
