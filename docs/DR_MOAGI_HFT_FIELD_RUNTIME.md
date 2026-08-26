# Dr Moagi HFT Sparse Field Runtime

## Status

Experimental specialization of the canonical Dr Moagi Field Runtime v2 for bounded, deterministic, ultra-low-latency streaming decisions. This document is an engineering target, not a claim of measured FPGA latency or trading profitability.

## 1. Objective

The HFT specialization removes operations that are unsuitable for a nanosecond critical path: full encode/decode passes, iterative minimization, heap allocation, global attention, stochastic sampling, floating-point reductions, and runtime model mutation.

Each market event performs one bounded local state transition:

```text
market event
  -> coordinate map
  -> local book update
  -> six-neighbour field gradient
  -> one Q16.16 Euler step
  -> persistent memory update
  -> fixed-weight score
  -> inventory/risk gate
  -> order intent
```

The intended hardware target is a cut-through FPGA/ASIC pipeline. A CPU implementation exists only as a deterministic functional reference.

## 2. State

For each lattice cell `c = (price_bin, venue, horizon)` maintain

```text
Psi[c]       instantaneous field state
Omega[c]     persistent exponentially weighted memory
Bid[c]       bounded bid-side depth proxy
Ask[c]       bounded ask-side depth proxy
Flow[c]      signed event-flow state
```

Global risk state contains at least inventory `I_t`.

The event is

```text
e_t = (price_tick, venue, side, delta_quantity, sequence)
```

All critical-path scalar arithmetic is signed Q16.16 with saturating add/subtract/multiply and truncation-toward-zero after a 64-bit intermediate product.

## 3. Local field equation

For the active cell `c_t`, define the six-neighbour discrete Laplacian

```text
Delta_6 Psi[c]
  = sum(Psi[n] for n in face_neighbours(c)) - 6 Psi[c].
```

Define the memory residual

```text
R_t[c] = Psi_t[c] - Omega_t[c].
```

and signed event impulse

```text
u_t = +delta_quantity  for bid events
u_t = -delta_quantity  for ask events.
```

The hot-path field update is one explicit step

```text
rhs_t[c]
  = -alpha R_t[c]
    + lambda Delta_6 Psi_t[c]
    + eta u_t

Psi_(t+1)[c]
  = clamp(Psi_t[c] + dt rhs_t[c], -Psi_max, +Psi_max).
```

This is a deliberately compiled surrogate of the more general Field Runtime v2 transition. It preserves bounded local propagation and persistent state while eliminating full codec evaluation from the trading path.

Persistent memory is

```text
Omega_(t+1)[c]
  = rho Omega_t[c] + (1-rho) Psi_(t+1)[c].
```

Flow is

```text
Flow_(t+1)[c]
  = rho_flow Flow_t[c] + u_t.
```

Only one center cell is written by the reference transition; seven field values are read (center plus six neighbours). Therefore field work per event is bounded independently of total lattice size.

## 4. Decision and risk

The reference score is intentionally linear so it lowers to a fixed MAC tree:

```text
s_t
  = w_psi Psi_(t+1)[c]
  + w_omega Omega_(t+1)[c]
  + w_flow Flow_(t+1)[c]
  + w_lap Delta_6 Psi_t[c]
  - w_inventory I_t.
```

Decision:

```text
if s_t >  threshold: BUY
if s_t < -threshold: SELL
otherwise:            NONE
```

Before an intent is emitted, projected inventory must satisfy

```text
|I_t + signed_order_quantity| <= I_max.
```

The reference kernel fails closed when this bound is violated.

This score is a hardware/mechanics demonstrator, not a validated alpha model. A production strategy must be trained and evaluated separately from the latency kernel.

## 5. Complexity

For one event the field kernel performs a fixed number of operations:

```text
7 field reads
1 local depth update
1 six-neighbour Laplacian
1 explicit field step
1 memory update
1 flow update
1 fixed-width score reduction
1 bounded risk check
```

For fixed channel/stencil width,

```text
C_event = O(1)
```

with respect to total historical state and total logical field size. This does not mean all-market global inference is O(1); it means the incremental critical-path update is bounded.

## 6. Pipeline budget

Reference design budget at 500 MHz:

| Stage | Cycles |
|---|---:|
| ingress / cut-through parse | 8 |
| local book update | 4 |
| state load | 4 |
| field-gradient MACs | 8 |
| memory update | 4 |
| score reduction | 6 |
| risk gate | 4 |
| order encode | 6 |
| TX launch | 4 |
| **Total** | **48** |

At 500 MHz:

```text
48 cycles * 2 ns/cycle = 96 ns
```

`96 ns` is a synthesis target only. It must not be reported as measured latency until an RTL implementation is placed, routed, timed, and measured network-to-network.

For context, the audited STAC-T1 ADHOC HFFT-02A result published in October 2024 reported 115.07 ns mean and 140.04 ns 99th percentile SOM-to-SOF at 1x market rate, with 76.40 ns mean EOT-to-SOF. The benchmark version differs from earlier STAC-T1 versions, so cross-version comparisons must be handled carefully.

Reference: https://docs.stacresearch.com/news/ADHC240918

## 7. Determinism contract

A conforming hardware/software implementation must specify and preserve:

1. Q16.16 representation and scaling;
2. 64-bit multiply intermediate;
3. truncation toward zero after fixed-point multiplication;
4. saturating overflow behavior;
5. fixed reduction order;
6. fixed coordinate wrapping/mapping;
7. no hidden floating-point operations in the hot path;
8. no dynamic allocation in `process`;
9. deterministic replay digest for an identical event stream;
10. fail-closed risk behavior.

## 8. Verification ladder

Performance claims advance only through these gates:

```text
G0 mathematical boundedness
G1 bit-exact C++ replay
G2 compiler/sanitizer regression
G3 HLS/RTL equivalence against C++ vectors
G4 synthesis timing closure
G5 post-route timing + resource utilization
G6 loopback packet latency
G7 dual-port network timestamping
G8 STAC-T1-compatible workload
G9 independent/audited benchmark
```

A failure at any gate blocks promotion of the corresponding performance claim.

## 9. Current implementation

Files:

```text
cpp_runtime/include/jarvisx/hft_field.hpp
cpp_runtime/src/hft_field_main.cpp
cpp_runtime/tests/hft_field_tests.cpp
```

The C++ simulator reports software throughput and the 48-cycle FPGA design target separately. Software timing must never be relabeled as FPGA tick-to-trade latency.

## 10. Next hardware lowering

The first RTL should preserve the C++ state transition exactly:

```text
RX stream
 -> event parser
 -> coordinate mapper
 -> BRAM/URAM 7-cell read bank
 -> Laplacian MAC tree
 -> Q16.16 field-step pipeline
 -> Omega/Flow update
 -> score MAC tree
 -> risk comparator
 -> order-intent encoder
 -> TX stream
```

The production design should support shadow mode first: receive live/replayed market data, calculate intents, timestamp them, but emit no exchange-bound orders. Order transmission is enabled only after deterministic equivalence, risk, and latency verification.
