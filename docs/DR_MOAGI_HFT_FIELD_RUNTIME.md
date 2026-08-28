# Dr Moagi HFT Sparse Field Runtime

## Status

Experimental specialization of the canonical Dr Moagi Field Runtime v2 for bounded, deterministic, ultra-low-latency streaming decisions. The repository now contains a bit-exact Q16.16 RTL arithmetic pipeline, conservative same-cell hazard protection, and a persistent center-state reference store. This document distinguishes verified functional/synthesis properties from unmeasured FPGA timing targets.

## 1. Objective

The HFT specialization removes operations that are unsuitable for a nanosecond critical path: full encode/decode passes, iterative minimization, heap allocation, global attention, stochastic sampling, floating-point reductions, and runtime model mutation.

Each market event performs one bounded local transition:

```text
market event
  -> coordinate map
  -> state load
  -> six-neighbour field gradient
  -> one Q16.16 Euler step
  -> persistent memory/flow update
  -> fixed-weight score
  -> inventory/risk gate
  -> state commit
  -> order intent
```

The intended production target is a cut-through FPGA/ASIC pipeline. The C++ implementation remains the functional oracle.

## 2. State and arithmetic contract

For each logical lattice cell `c = (price_bin, venue, horizon)` maintain at least:

```text
Psi[c]       instantaneous field state
Omega[c]     persistent exponentially weighted memory
Flow[c]      signed event-flow state
```

The event is

```text
e_t = (price_tick, venue, side, delta_quantity, sequence)
```

All implemented scalar hot-path arithmetic is signed Q16.16. The software and RTL contract preserves:

- saturating signed add/subtract;
- 64-bit intermediate arithmetic where required;
- truncation toward zero for dyadic division/fixed-point multiplication;
- fixed left-associative reduction order;
- fail-closed inventory risk behavior.

The default coefficients are dyadic rational values (`1/2`, `1/4`, `1/8`, `1/16`, `1/32`, `7/8`, `15/16`), so the optimized RTL removes general multipliers and uses exact shift/add/subtract lowering while retaining the reference arithmetic semantics.

## 3. Local field transition

For active cell `c_t`, define

```text
Delta_6 Psi[c]
  = (((((Psi[x-] + Psi[x+]) + Psi[y-]) + Psi[y+]) + Psi[z-]) + Psi[z+])
    - 6 Psi[c]
```

where every addition/subtraction uses the same saturating Q16.16 order as the C++ reference.

Memory residual and event impulse are

```text
R_t[c] = Psi_t[c] - Omega_t[c]

u_t = +delta_quantity  for bid events
u_t = -delta_quantity  for ask events
```

and the compiled reference update is

```text
rhs_t[c]
  = -R_t[c] / 4
    + Delta_6 Psi_t[c] / 32
    + u_t / 2

Psi_(t+1)[c]
  = clamp(Psi_t[c] + rhs_t[c] / 8, -Psi_max, +Psi_max)

Omega_(t+1)[c]
  = 15 Omega_t[c] / 16 + Psi_(t+1)[c] / 16

Flow_(t+1)[c]
  = 7 Flow_t[c] / 8 + u_t
```

Only one center cell is committed per accepted event. The complete stencil requires center plus six neighbouring `Psi` values.

## 4. Decision and risk

The implemented score is evaluated in fixed order:

```text
s_t
  = Psi_(t+1)[c]
  + Omega_(t+1)[c] / 2
  + Flow_(t+1)[c] / 2
  + Delta_6 Psi_t[c] / 8
  - Inventory_t / 4
```

Decision threshold is `1/16` in Q16.16 units. Order quantity is bounded at `4`; a zero requested quantity uses the same bounded fallback. Before emission, projected inventory must satisfy

```text
|Inventory_t + signed_order_quantity| <= 64
```

otherwise the action is rejected and cleared. This score is a mechanics demonstrator, not a validated trading alpha model.

## 5. Verified RTL layers

Current RTL files include:

```text
rtl/hft_field_q16/hft_field_cell_core.sv
rtl/hft_field_q16/hft_field_cell_pow2.sv
rtl/hft_field_q16/hft_field_cell_pipeline.sv
rtl/hft_field_q16/hft_field_cell_staged.sv
rtl/hft_field_q16/hft_field_hazard_guard.sv
rtl/hft_field_q16/hft_field_guarded_pipeline.sv
rtl/hft_field_q16/hft_field_state_store.sv
rtl/hft_field_q16/hft_field_stateful_center.sv
```

The CI verification chain currently proves:

1. C++-derived golden-vector agreement for the reference RTL;
2. multiplier-free equivalence across 10,012 deterministic RTL vectors, including signed extrema/saturation cases;
3. a fixed 17-cycle valid-to-valid shell contract;
4. a genuinely staged 17-cycle arithmetic pipeline matching the multiplier-free oracle for back-to-back II=1 transactions;
5. conservative RAW hazard rejection for a coordinate already in flight while independent coordinates remain issuable on consecutive clocks;
6. exact coordinate/result alignment through the guarded pipeline;
7. persistent center-state commit/reload across repeated same-cell transactions;
8. fail-closed serialization of configuration writes against event issue;
9. Verilator lint and generic Yosys synthesis for the reference, multiplier-free, staged, guarded, and stateful-center RTL layers.

These are functional and generic synthesis results. They are not FPGA place-and-route timing results.

## 6. Stateful recurrence boundary

The current stateful engine stores and reloads center-cell `Psi`, `Omega`, and `Flow` values. A same-coordinate transaction is blocked until the previous transaction has retired and committed, so the next recurrence observes the newly committed state.

The current reference store uses a combinational center read and synchronous write. It is intentionally small and technology-neutral; generic Yosys acceptance does **not** prove BRAM/URAM inference, banking, routing, or target-device timing.

Six-neighbour `Psi` values are still supplied externally to `hft_field_stateful_center.sv`. A production implementation must therefore add a physically realizable neighbour-address and memory-banking architecture rather than assume an unrealistic seven-read single RAM.

## 7. Latency contracts and targets

### Verified logical arithmetic latency

The staged arithmetic transaction contract is:

```text
accepted input -> result = 17 clock cycles
```

The pipeline can accept independent transactions at initiation interval `II=1`. Same-coordinate traffic is conservatively stalled until commit; same-cell speculative forwarding is not implemented.

If a future FPGA implementation closes at 500 MHz, the 17-cycle arithmetic latency would correspond to:

```text
17 * 2 ns = 34 ns
```

`34 ns` is therefore a derived architectural target, **not a measured latency**.

### Broader wire-to-wire budget

The original system-level budget remains:

| Stage | Cycles |
|---|---:|
| ingress / cut-through parse | 8 |
| local book update | 4 |
| state load | 4 |
| field-gradient arithmetic | 8 |
| memory update | 4 |
| score reduction | 6 |
| risk gate | 4 |
| order encode | 6 |
| TX launch | 4 |
| **Total** | **48** |

At a hypothetical 500 MHz that is `96 ns`, also a synthesis target only. It must not be reported as measured network-to-network latency until the complete design is placed, routed, timed, and physically measured.

## 8. Complexity

For fixed stencil/channel width, one accepted local event performs a fixed amount of arithmetic and center-state work, so the incremental compute kernel is `O(1)` with respect to total logical lattice size. This does not imply that all-market inference, memory footprint, neighbour addressing, or network processing is `O(1)`.

## 9. Determinism contract

A conforming hardware/software implementation must preserve:

1. Q16.16 representation and scaling;
2. saturating overflow behavior;
3. truncation toward zero;
4. fixed arithmetic/reduction order;
5. deterministic coordinate mapping and boundary policy;
6. no hidden floating-point operations in the hot path;
7. no dynamic allocation in the event-processing path;
8. deterministic replay for an identical event stream;
9. fail-closed risk behavior;
10. explicit state-hazard semantics.

## 10. Verification ladder

Performance claims advance only through these gates:

```text
G0 mathematical boundedness
G1 bit-exact C++ replay
G2 compiler/sanitizer regression
G3 RTL functional equivalence + generic synthesis
G4 target FPGA synthesis and timing constraints
G5 post-route timing + resource utilization
G6 loopback packet latency
G7 dual-port network timestamping
G8 STAC-T1-compatible workload
G9 independent/audited benchmark
```

The current RTL has materially advanced through G3 for the arithmetic, hazard, guarded-transaction, and persistent center-state reference layers. G4 and later remain unproven.

## 11. Next hardware lowering

The next implementation boundary is:

```text
3D coordinate
 -> deterministic center/x-/x+/y-/y+/z-/z+ address generator
 -> boundary policy
 -> physically realizable bank/replica selection
 -> center + six-neighbour Psi fetch
 -> staged Q16.16 arithmetic
 -> center-state commit
```

After neighbour banking, the design still requires a named FPGA target, synchronous memory timing integration, constraints, place-and-route, packet parser/encoder, Ethernet MAC/PHY integration, and timestamped replay before any hardware latency comparison is defensible.

Production rollout should begin in shadow mode: receive live/replayed market data, calculate and timestamp intents, but emit no exchange-bound orders until deterministic equivalence, risk controls, and physical latency verification are complete.
