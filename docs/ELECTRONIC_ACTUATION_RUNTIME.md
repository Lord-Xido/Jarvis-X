# Jarvis-X Electronic Actuation Runtime

## Scope

This document defines the literal electronic implementation boundary beneath the Jarvis-X inward multimodal runtime.

The mapping is:

```text
algorithmic state
-> bounded numerical representation
-> voltage / bit representation
-> switching and memory activity
-> software or I/O command representation
-> observed result
-> re-encoding and verification
```

Jarvis-X does **not** identify semantic state with an electromagnetic field. Semantic variables are algorithmic variables; electrical variables are the physical quantities used to encode and execute them.

## 1. Scalar-to-voltage encoding

For a bounded scalar `x in [x_min, x_max]`, define

```text
u = (x - x_min) / (x_max - x_min)
V = V_min + u (V_max - V_min)
```

and inverse

```text
x = x_min + (V - V_min)/(V_max - V_min) (x_max - x_min)
```

This is the explicit bridge from a runtime control scalar to an electrical node representation.

## 2. Bit-level logic

A digital bit is represented by two voltage regions:

```text
0 <-> V_low
1 <-> V_high
```

with decoding threshold

```text
b = 0, V < V_TH
b = 1, V >= V_TH
```

A finite switching event charges or discharges effective capacitance. A representative transition energy is

```text
E_switch ~= 1/2 C_eff (Delta V)^2
```

and conventional dynamic power is

```text
P_dyn ~= alpha_sw C_eff (Delta V)^2 f
```

where `alpha_sw` is switching activity and `f` is clock rate.

## 3. Runtime execution chain

The practical actuation chain is

```text
Z_t
-> candidate action
-> constraint projection Pi_K
-> executable instruction / I/O representation
-> transistor switching and memory transactions
-> external observation O_(t+1)
-> E(O_(t+1))
-> correction
```

Thus the intelligence layer does not directly move hardware. It emits bounded numerical actions which are lowered through software and hardware layers.

## 4. Relation to the inward RC analogue

The existing inward swarm runtime exposes

```text
C dV_i/dt =
    g_phi (V_phi_i - V_i)
  + g_c sum_j A_ij (V_j - V_i)
  + I_ext_i
```

This is a circuit-level dynamical correspondence for the algorithmic recurrence

```text
dZ/dt = lambda(Phi(Z)-Z) - gamma L Z + F_ext.
```

The correspondence is structural:

```text
latent scalar z_i    <-> node voltage V_i
inward gain lambda   <-> g_phi / C
coupling A_ij        <-> effective conductance weights
external forcing     <-> injected current term
```

It does not imply that semantic distance is literal physical distance or that the runtime is a Maxwell-field solver.

## 5. Practical electronic observability

The hardware boundary should expose measurable quantities wherever available:

- node voltage or digital logic level;
- switching count or activity estimate;
- memory transaction count;
- estimated dynamic energy and power;
- instruction or kernel provenance;
- I/O command provenance;
- observed result after execution.

These become part of the external audit trail rather than relying on hidden model reasoning.

## 6. Safety and authority boundary

The runtime separates intelligence from authority:

```text
A_intent = G(Z_t, Omega_t)
A_authorized = Pi_K(A_intent)
A_executable = Lower(A_authorized)
```

The research module in this branch performs **representation and estimation only**. It does not perform GPIO writes, device-driver operations, direct peripheral control, or unrestricted hardware actuation.

## 7. Current implementation

`src/jarvisx/electronic_actuation_runtime.py` adds:

- `VoltageCodec` for scalar <-> voltage mapping;
- `CMOSLogicModel` for logic thresholding and switching-energy/power estimates;
- `PWMCommand` / `PWMMapper` as a normalized command representation;
- `ElectronicActuationTrace` for auditable end-to-end mapping of one control scalar;
- `trace_control_scalar()` and `trace_vector()` helpers.

The module deliberately stops before any hardware API or device driver.

## 8. Engineering interpretation

The complete physical stack is therefore

```text
semantic/task state
-> numeric tensors
-> bits / encoded voltages
-> logic and memory transitions
-> electrical switching
-> I/O representation
-> environment
-> observation
-> re-encoding
```

This is the literal electronic actuation boundary for Jarvis-X.