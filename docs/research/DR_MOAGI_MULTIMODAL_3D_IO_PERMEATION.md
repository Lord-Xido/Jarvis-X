# Dr Moagi Multimodal 3D I/O Permeation Runtime

**Canonical research operator:** `G_{Omega Xi,IO}^3D`  
**Implementation:** `src/jarvisx/dr_moagi_multimodal_io.py`  
**Decision:** `docs/adr/0006-dr-moagi-multimodal-3d-io-runtime.md`

## 1. Purpose

This specification turns the Dr Moagi 3D auto-encoding/decoding equation into one bounded systems contract for all input and output media.

The physical/digital boundary is modeled as:

```text
WORLD / EXTERNAL SYSTEM
        |
        v
sensor/transducer
        |
        v
medium encoder E_m
        |
        +----> Z_m --+
                     |
medium encoder E_k ->+--> FUSE^3D --> Xi_t^3D
                     |                 |
                     |                 v
                     |              PREDICT
                     |                 |
                     |                 v
                     +<-- VERIFY <-- DECODE --> actuator/transducer
                                           |
                                           v
                                  WORLD / EXTERNAL SYSTEM
```

The unifying statement is:

```text
input medium  = transduction + encoding
output medium = decoding + actuation
closed pair   = adaptive control/verification loop
```

This is an architectural abstraction. It does not make heterogeneous physics identical.

## 2. Canonical operator

```math
\mathfrak G_{\Omega\Xi,\mathrm{IO}}^{3D}:
(\Xi_t^{3D}, \mathbf X_t, \Omega_t^{3D}, \Theta_t)
\mapsto
(\Xi_{t+1}^{3D}, \widehat{\mathbf X}_t,
 \Omega_{t+1}^{3D}, \Theta_{t+1})
```

with

```math
\Xi_{t+1}^{3D}
=
\Pi_{\Lambda_t}
\left[
\Xi_t^{3D}
+
P_{1:M}^{\circlearrowleft}
\left(
\Xi_t^{3D},
\mathcal F^{3D}\mathcal E_{1:M}(\mathbf X_t)
\right)
-
E_t^{3D}
+
\Omega_t^{3D}
+
U_t^{3D}
-
\eta_t\nabla_\Theta\mathcal L_t
\right].
```

The runtime reference realizes a conservative subset:

```text
Xi_candidate =
Pi_Lambda(
    Xi
    + dt * [
        k_input      * (Z_fused - Xi)
        + k_predict  * P(Xi, Z_fused)
        + k_error    * E
        + k_memory   * Omega_candidate
    ]
)
```

and

```text
Omega_candidate =
Pi_Lambda(
    rho * Omega
    + eta_omega * E
)
```

`Theta` adaptation remains external to the first reference runtime. That keeps the fast I/O transaction deterministic while permitting slower bounded policy optimization around it.

## 3. Per-cycle transaction

One `step()` invocation is one macro-cycle:

1. snapshot authoritative research-layer `Xi_t` and `Omega_t`;
2. reject unknown channel names;
3. call each active adapter's `encode_input`;
4. validate coordinates, vector widths and finite numbers;
5. preserve explicit zero observations at the I/O boundary;
6. perform deterministic weighted 3D fusion;
7. evaluate the configured predictor;
8. decode current state to output representations;
9. when targets exist, invoke explicit output loopback;
10. compute target-minus-observed residual fields;
11. fuse residuals into `E_t^3D`;
12. update the `Omega` candidate;
13. compute the candidate `Xi_(t+1)^3D`;
14. project every value through `Pi_Lambda`;
15. reject active-cell budget violations;
16. compute distortion/resource loss telemetry;
17. invoke an optional validator;
18. atomically commit both state and memory or roll back both;
19. decode committed outputs;
20. emit metrics.

No partial candidate is authoritative.

## 4. Channel interface

```python
class MediumAdapter(Protocol):
    def encode_input(self, observation):
        ...

    def decode_output(self, field):
        ...

    def observe_output(self, output):
        ...
```

Examples:

| Channel | `encode_input` | `decode_output` | `observe_output` |
|---|---|---|---|
| Vision/display | camera frame -> field | field -> framebuffer | calibrated camera/colorimeter |
| Audio | microphone samples -> field | field -> PCM/DAC command | microphone loopback |
| Text | tokens/events -> field | field -> text/message | parser/ACK/task result |
| Touch/haptic | position/pressure -> field | field -> haptic command | force/position sensor |
| Robotics | encoder/IMU -> field | field -> motor command | encoder/IMU/force loopback |
| RF | IQ samples -> field | field -> IQ/symbol command | receiver/EVM/BER |
| Network | packets/events -> field | field -> packets | ACK/response/telemetry |
| Storage | read data -> field | field -> write request | read-after-write/checksum |
| Thermal | temperature -> field | field -> heater/cooler command | temperature sensor |
| Electrical | ADC measurements -> field | field -> DAC/PWM command | voltage/current sensing |

Real adapters must define units, calibration, sampling rate, authorization, fault semantics and resource ceilings.

## 5. 3D representation

A logical cell is addressed by:

```text
(x, y, z),  0 <= x,y,z < side
```

and holds a bounded vector:

```text
v_xyz in R^K.
```

The common field is sparse:

```text
Xi_t^3D = {(x,y,z): v_xyz, ...}
```

so:

```text
logical capacity = side^3
resident cells   = number of explicit coordinates
```

These quantities must never be conflated.

## 6. Multimodal fusion

For explicit observations at coordinate `c`:

```math
Z(c)
=
\frac{\sum_m w_m Z_m(c)}
     {\sum_m w_m}.
```

A supplied zero vector is retained as an observation during fusion. Missing support and an explicit measurement of zero are therefore distinct at the medium boundary.

Fusion is ordered deterministically by channel name and coordinate linear address.

## 7. Feedback and physical truth

The runtime separates:

```text
command generated
```

from:

```text
effect observed.
```

For channel `m`:

```math
E_m = Y_m^\star - Y_m^{observed}.
```

If a target is supplied, the adapter must provide `observe_output`. If the hardware has no measurement path, the adapter should fail rather than substitute link metadata.

Examples:

- HDMI/DisplayPort link state is not proof of emitted luminance.
- a successful socket write is not proof of remote task completion.
- a storage write syscall is not proof of durable media persistence.
- a motor command is not proof of achieved joint position.

## 8. Omega memory

Residuals persist as bounded correction memory:

```math
\Omega_{t+1}
=
\rho\Omega_t
+
\eta_\Omega E_t.
```

The reference runtime commits `Omega` and `Xi` together. Validator rejection preserves both previous snapshots.

## 9. Pi-Lambda

`Pi_Lambda` is implemented by concrete checks:

```text
coordinate inside lattice
AND vector width == K
AND every scalar finite
AND value_min <= scalar <= value_max after projection
AND active_cells <= max_active_cells
AND channels <= max_channels
AND validator accepts candidate
```

Malformed state fails closed.

## 10. Objective and telemetry

The exposed objective is:

```math
L_t =
lambda_D D_t
+ lambda_tau tau_t
+ lambda_B B_t
+ lambda_P P_t
+ lambda_C C_t.
```

The reference runtime computes distortion and an approximate scalar-operation count. Latency, bandwidth and energy terms are accepted only as measured non-negative inputs.

Metrics include:

```text
cycle
channels_seen
active_cells_before/after
fused_input_cells
prediction_cells
error_cells
distortion_mse
memory_l2
approximate_scalar_ops
loss
committed/rejection_reason
```

## 11. Fast and slow loops

Production deployments should separate timescales:

```text
FAST:
sense -> encode -> fuse -> predict -> decode -> verify -> commit

SLOW:
aggregate telemetry
-> evaluate parameter/policy candidates
-> benchmark
-> Pi_Lambda validate
-> version
-> atomic policy swap or rollback
```

An ANN may propose predictions or policy parameters. It must not bypass the deterministic transaction boundary.

## 12. Operational channel families

The operator is intended to permeate these medium families:

```text
optical
acoustic
symbolic/textual
touch/haptic
mechanical/motion
electromagnetic/RF
network/protocol
storage/persistence
thermal
pressure/fluid
chemical
electrical
```

New families require only an adapter contract, not a new global control law.

## 13. Non-claims

This runtime does not claim:

- that one latent representation is information-lossless for arbitrary media;
- that an adapter can infer physical output without sensing it;
- that a logical 3D field corresponds to physical 3D hardware;
- that adaptive parameters are globally optimal;
- that simulation proves safety of a physical actuator;
- that the research runtime has authority over the canonical VM.

## 14. Validation baseline

The first conformance suite requires:

```text
[PASS] weighted multimodal fusion
[PASS] residual feedback
[PASS] Omega persistence
[PASS] validator rollback
[PASS] value projection
[PASS] unknown channel rejection
[PASS] sparse active-cell ceiling
[PASS] explicit physical-loopback semantics
```

These tests establish the software contract. Hardware-specific adapters require their own calibration, protocol and safety tests.
