# DM-vΩΞ⁺ Bit-Serial Mixed-Signal Control Reference

## Implemented result

Jarvis-X now has a dependency-free, bounded software reference for the complete
sensor-to-gate-intent path:

```text
sampled sensor values
  -> first-order delta-sigma bitstream
  -> XNOR/popcount matrix
  -> bounded signed scores
  -> 16-bit XOR/rotate Omega memory
  -> Theta bit mask + Hamming error
  -> independent hardware interlock
  -> pulse-density logic frames
```

The implementation is `src/jarvisx/dm_vomegaxi_mixed_signal.py`. It emits data
structures only. It has no GPIO, PWM peripheral, network, filesystem, MOSFET, or
H-bridge authority.

## Closed operator map

For sampled physical state (S_k\), the bounded discrete reference is

\[
S_k
\xrightarrow{\Delta\Sigma}
b_k
\xrightarrow{\Psi,\Phi}
x_k
\xrightarrow{\Lambda}
z_k
\xrightarrow{\Omega}
h_{k+1}
\xrightarrow{\Theta}
u_k
\xrightarrow{\mathcal I_{\mathrm{HW}}}
\tilde u_k
\xrightarrow{\mathrm{PDM}}
p_k.
\]

The independent interlock (\mathcal I_{\mathrm{HW}}\) is deliberately outside
Theta. Theta expresses application policy; the interlock fail-closes on measured
electrical, thermal, timing, watchdog, or emergency conditions.

No physical system is zero-latency. The implementation therefore claims a finite,
deterministic number of software operations per bounded frame, not instantaneous
execution.

## Ψ — sampled state acquisition

The reference accepts finite sensor samples within a declared interval

\[
s_c\in[s_{\min},s_{\max}].
\]

It normalizes each channel to (v_c\in[-1,1]\) and executes a stateful first-order
one-bit modulator for every oversampling tick:

\[
a_{c,n}'=a_{c,n}+v_c,
\]

\[
b_{c,n}=\mathbf 1[a_{c,n}'\ge 0],
\]

\[
a_{c,n+1}=a_{c,n}'-(2b_{c,n}-1).
\]

Bits are laid out **time-major, then channel-major**. An oversampling ratio (R\)
and (C\) channels therefore produce exactly (N=RC\) bits per engine step.

This is a numerical modulator reference. A real analog front end still requires
an ADC/comparator, clocking, anti-aliasing, calibration, isolation, and signal
conditioning appropriate to the plant.

## Φ and Λ — Boolean description under capacity bounds

For input bits (b\in\{0,1\}^N\) and one binary weight row
(w_j\in\{0,1\}^N\), the implementation computes

\[
m_j=\operatorname{popcount}(\operatorname{XNOR}(b,w_j)),
\]

\[
r_j=2m_j-N.
\]

This is exactly the bipolar dot-product identity for values encoded as
(0\mapsto-1\) and (1\mapsto+1\). Lambda applies a finite integer projection

\[
\bar r_j=\operatorname{clip}(r_j,-B_\Lambda,+B_\Lambda),
\]

followed by the one-bit latent decision

\[
z_j=\mathbf 1[\bar r_j>0].
\]

Input width, output rows, score magnitude, memory words, and pulse-frame length
all have explicit configuration limits. “XNOR/popcount” is an arithmetic model;
Python execution is not a claim of a single FPGA/ASIC pipeline stage.

## Ω — 16-bit recurrent register memory

Latent bits are packed least-significant-bit first into a fixed number of unsigned
16-bit words. For incoming word (q_i\), prior state (h_{i,k}\), rotation (r\),
and retention mask (M\), the exact update is

\[
v_i=\operatorname{ROTL}_{16}(h_{i,k}\oplus q_i,r),
\]

\[
h_{i,k+1}
=(M\land h_{i,k})
\lor
(\neg M\land v_i).
\]

The word count is finite. Inputs exceeding the declared register capacity are
rejected rather than truncated.

## Θ — application constraint and measurable error

The first output-count bits of the updated Omega state are the candidate vector
(y_k\). Theta applies an explicit mask

\[
\hat y_k=M_{\Theta}\land y_k.
\]

When a target vector is supplied, the reference reports the Hamming residual

\[
e_k=y_k^{\mathrm{target}}\oplus\hat y_k,
\]

\[
E_H=\operatorname{popcount}(e_k),
\qquad
\epsilon_H=\frac{E_H}{n_{\mathrm{out}}}.
\]

This binary metric is valid only for binary outputs. Continuous residuals require
a metric defined in their own signal space. The Theta mask is caller-supplied
policy data; it does not establish moral truth, authentication, or a security
sandbox.

## Independent hardware interlock

`HardwareInterlock` evaluates plant-specific limits supplied by the integrator.
There are intentionally no universal current, voltage, temperature, watchdog, or
dead-time defaults. The caller must provide limits for its actual hardware.

The reference fail-closes on any of:

- emergency stop;
- absolute overcurrent;
- absolute overvoltage;
- overtemperature;
- watchdog timeout;
- observed bridge overlap;
- insufficient observed dead time.

If one or more trips occur, every emitted bit and duty command is zero and the PDM
accumulators are reset. The cognitive/memory step remains inspectable, so the
report distinguishes `theta_output_bits` from `emitted_bits`.

Software checks are not the primary protection layer. A physical implementation
must retain independent hardware comparators or protection devices, gate-driver
interlock/dead time, current limiting, thermal shutdown, undervoltage lockout,
watchdog, galvanic isolation where required, and a de-energizing emergency path.

## Pulse-density and gate-intent frames

For a permitted channel with duty (d\in[0,d_{\max}]\), the accumulator is

\[
c_{n}'=c_n+d,
\]

\[
p_n=\mathbf 1[c_n'\ge1],
\qquad
c_{n+1}=c_n'-p_n.
\]

The returned `HBridgeGateFrame` contains positive and negative logic-intent
vectors. Its constructor rejects overlap and rejects a gate on the side opposite
the declared polarity. This is a software invariant, not generated switching
waveform timing. A certified gate driver remains authoritative.

## Internal fixed point

Let the finite controller state be

\[
X_k=(a_k,h_k,c_k,u_{k-1}),
\]

where (a\) is delta-sigma accumulator state, (h\) is Omega register memory,
(c\) is PDM accumulator state, and (u\) is the emitted bit vector. The report
marks an internal fixed point only when

\[
\Delta_{\mathrm{bits}}(X_{k+1},X_k)=0
\]

and

\[
\Delta_{\mathrm{numeric}}(X_{k+1},X_k)\le\epsilon.
\]

This is an equality test over the software controller state. It is not evidence
that the external plant, environment, or world is at equilibrium.

## Reference API

```python
from jarvisx.dm_vomegaxi_mixed_signal import (
    DMvOmegaXiMixedSignalEngine,
    HardwareInterlockLimits,
    HardwareTelemetry,
    MixedSignalConfig,
)

config = MixedSignalConfig(
    sensor_channels=1,
    oversample=4,
    score_bound=4,
    memory_words=1,
    omega_rotate_bits=0,
    pdm_period=8,
)
limits = HardwareInterlockLimits(
    max_abs_current_a=5.0,
    max_abs_voltage_v=24.0,
    max_temperature_c=80.0,
    watchdog_timeout_ticks=3,
    min_dead_time_ticks=2,
)
engine = DMvOmegaXiMixedSignalEngine(
    weights=[(1, 1, 1, 1)],
    interlock_limits=limits,
    config=config,
)

report = engine.step(
    [1.0],
    HardwareTelemetry(
        current_a=1.0,
        voltage_v=12.0,
        temperature_c=30.0,
        watchdog_age_ticks=1,
        observed_dead_time_ticks=2,
    ),
    target_bits=(1,),
)

assert report.actuation_permitted
assert report.hamming_error_bits == 0
```

The numbers above are an executable example, not a recommendation for any
specific circuit.

## Verified scope

The focused tests cover:

- delta-sigma extrema, zero input, channel ordering, range rejection, and reset;
- the XNOR/popcount signed-dot identity;
- 16-bit packing, unpacking, rotation, capacity, and persistence masking;
- each independent interlock trip and exact boundary acceptance;
- bounded stateful PDM and mutually exclusive gate intents;
- Theta masking and Hamming error;
- fail-closed override of otherwise permitted Theta output;
- deterministic replay and reset;
- exact discrete internal fixed-point detection;
- rejection before state mutation for malformed targets.

Not implemented or implied:

- analog acquisition hardware;
- a trained binary neural network;
- calibrated plant dynamics or closed-loop stability;
- FPGA/ASIC synthesis or timing closure;
- real-time scheduling guarantees;
- safe direct MOSFET or H-bridge control;
- autonomous ethical judgment;
- a physical fixed point or zero-latency computation.
