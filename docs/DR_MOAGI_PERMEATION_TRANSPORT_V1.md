# Dr Moagi Permeation Transport v1

## Purpose

This module turns the permeation concept into a bounded, replayable software transport path:

```text
state -> encode -> modulate -> channel -> demodulate -> verify -> reconstruct
```

It is intentionally a **simulation/transport layer**. It does not transmit RF energy and does not access radio hardware.

## Runtime equation

For one BPSK symbol `s_n in {-1, +1}` the reference channel is:

```text
y_n = h s_n + noise_n
```

where

```text
h = Q * A(theta) / (4 pi r) * exp(i k r)
k = 2 pi / lambda
lambda = c / f
A(theta) = w0 + w2 * (3 cos(theta)^2 - 1) / 2
```

The receiver uses the known simulated coefficient:

```text
s_hat_n = y_n / h
bit_n = 1 if Re(s_hat_n) >= 0 else 0
```

The reconstructed byte stream is accepted only when its SHA-256 digest equals the digest bound before modulation.

## Defaults

| Parameter | Default |
|---|---:|
| carrier frequency | 333,330,000 Hz |
| propagation speed | 299,792,458 m/s |
| source strength `Q` | 0.941 |
| coherence metadata | 0.967 |
| range | 1 m |
| monopole weight | 0.6 |
| quadrupole weight | 0.4 |
| axis | `[0, 1, 0]` |
| receiver direction | `[0, 1, 0]` |
| noise standard deviation | 0 |
| payload ceiling | 65,536 bytes |

At the defaults:

```text
wavelength ~= 0.8993863679 m
wave number ~= 6.9860802117 rad/m
1 m propagation delay ~= 3.335640952 ns
aligned scalar amplitude ~= 0.0748824007
```

## Operational commands

### FOCUS

Software meaning:

```python
focused = config.focused((x, y, z))
```

This rotates the quadrupole axis used by the scalar angular-gain model. The v1 quadrupole is symmetric about both directions of the axis.

### MODULATE

Software meaning:

```python
frame = modulate(state, config)
```

The state is serialized as canonical JSON, hashed, converted to bits, and mapped to BPSK symbols.

### PROPAGATE

Software meaning:

```python
received = propagate(frame, config)
```

Each symbol receives the configured `1/r` amplitude, carrier phase, angular gain, and optional seeded Gaussian complex noise.

### ABSORB

Software meaning:

```python
state_hat = absorb(received)
```

The receiver equalizes the simulated channel, demodulates bits, reconstructs bytes, verifies the digest, and parses the JSON state. Digest mismatch fails closed.

## Cloud operation

The default Dr Moagi Cloud coordinator registers:

```text
permeate-roundtrip.v1
```

Example request:

```json
{
  "operation": "permeate-roundtrip.v1",
  "request_id": "permeation-demo-001",
  "input": {
    "state": {
      "latent": [0.1, 0.2, 0.3],
      "intent": "externalize-and-reconstruct"
    },
    "config": {
      "carrier_hz": 333330000.0,
      "range_m": 1.0,
      "axis": [0.0, 1.0, 0.0],
      "receiver_direction": [0.0, 1.0, 0.0]
    }
  }
}
```

A successful result contains transport telemetry similar to:

```json
{
  "operation": "permeate-roundtrip.v1",
  "protocol": "jarvisx.dr-moagi-permeation.v1",
  "physical_rf": false,
  "model": "deterministic-bpsk-scalar-free-space-simulation",
  "carrier_hz": 333330000.0,
  "wavelength_m": 0.8993863678636786,
  "range_m": 1.0,
  "propagation_delay_ns": 3.3356409519815204,
  "verified": true,
  "reconstructed": {}
}
```

The result is still only a candidate until the Cloud Runtime verification and promotion gate records `COMMITTED`.

## Resource behavior

The transport rejects a payload before BPSK symbol expansion if its canonical JSON representation exceeds `max_payload_bytes`.

The cloud adapter additionally prevents `max_payload_bytes` from exceeding the enclosing cloud job input budget.

A payload of `N` encoded bytes produces exactly:

```text
8N BPSK symbols
```

so symbol-memory cost remains explicit and bounded.

## Interpretation boundary

`carrier_hz`, `wavelength_m`, `range_m`, `propagation_delay_ns`, and `amplitude_at_receiver` describe the **software channel model**. They do not establish that an RF carrier exists outside the process.

Every transport result therefore exposes:

```text
physical_rf = false
```

A future hardware adapter must be implemented and validated separately.
