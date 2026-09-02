# DM-vOmegaXi+ Mechatronic Control Contract

## Master fixed-point loop

The implementable closed-loop interpretation of `I AM = I DESCRIBE` is

```text
physical state -> sensing -> pulse description -> Boolean compute
               -> recurrent memory -> admissibility -> power command
               -> physical state
```

For tick `t`,

```text
Psi_t       = normalized sensor vector in [-1, 1]^N
b_t         = DeltaSigma(Psi_t)
r_t         = popcount(XNOR(b_t, W_t))
d_t         = 2 r_t - N
yhat_t      = 1[d_t >= 0]
e_t         = y_t XOR yhat_t
Omega_t+1   = ROTL16(Omega_t, k) XOR pack16(b_t) XOR (e_t << 15)
u_raw,t     = round(10^6 d_t / N)
u_t         = Theta(u_raw,t, stop_t, limits)
Psi_t+1     = Plant(Psi_t, PowerStage(u_t), disturbances_t)
```

The software reference terminates at `u_t`. `PowerStage` and `Plant` remain
explicit interfaces until hardware-specific implementations and evidence are
provided.

## Operator contract

| Symbol | Concrete primitive | Persistent state | Verified property |
|---|---|---|---|
| Psi | normalized sensor vector | external plant state | finite bounded input |
| Phi | first-order delta-sigma bank | one integrator/channel | one-bit output; density fixture |
| Lambda | XNOR plus popcount | immutable binary weights | exact bipolar dot identity |
| Omega | 16-bit rotate/XOR register | `uint16` word | deterministic bounded recurrence |
| Theta | safety governor | prior duty/dead-time counter | clamp, slew, stop, reversal interlock |
| Actuation | hardware-neutral H-bridge command | none in reference | no same-leg high/low overlap |

## Timing domains

Three clocks must never be collapsed into one headline number:

1. **Computational timing** measures delta-sigma state updates, popcount,
   recurrence, and constraint evaluation on a named processor.
2. **Electrical timing** measures PWM resolution, dead time, propagation delay,
   rise/fall time, and protection response on a named driver/power stage.
3. **Plant timing** measures sensor sampling, actuator response, settling time,
   overshoot, and closed-loop stability for a named mechanical load.

Sub-nanosecond device or logic events do not establish a sub-nanosecond
sensor-to-mechanical-to-sensor control loop. Every latency or throughput claim
must identify its boundary, clock, hardware, workload, statistic, and sample
count.

## Gate-command semantics

Duty is a signed integer in parts per million. Positive duty commands the left
high-side PWM and right low-side switch; negative duty commands the mirrored
pair. Zero, emergency stop, and direction dead time produce a coast command.
The reference never commands a high-side and low-side device on the same leg.

These values are logical commands, not GPIO signals. A physical integration
must add an independent hardware dead-time generator and fail-safe protection;
software invariants are not a substitute for electrical protection.

## Verification boundary

The included tests establish arithmetic and state-machine invariants only.
They do not establish functional safety, ethical adequacy, electromagnetic
compatibility, thermal performance, motor-control stability, or suitability
for medical, vehicle, industrial, or other safety-critical use.

Acceptance for physical use requires, at minimum, plant identification,
control-loop stability analysis, fault-tree/FMEA work, watchdog testing,
hardware-in-the-loop testing, and compliance review for the target domain.
