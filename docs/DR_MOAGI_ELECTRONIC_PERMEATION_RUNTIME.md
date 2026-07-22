# Dr Moagi Electronic Permeation Runtime

## Status

Executable deterministic substrate model for the Jarvis-X virtual machine.

The runtime maps each committed VM instruction to an auditable electronic state transition. It does **not** claim direct access to transistor counters, package power sensors, cache hardware counters, or junction-temperature sensors. All values emitted by `ElectronicSubstrate` are model outputs derived from declared configuration and observed VM state transitions.

---

## 1. Operational identity

The abstract Jarvis-X state is represented electronically as:

\[
S_e(t)=(V_t,Q_t,B_t,R_t,PC_t,M_t,C_t,T_t)
\]

where voltage, charge, bits, registers, program counter, memory, control, and thermal state form the physical execution substrate.

The software implementation uses the measurable VM projection:

\[
\widehat S_e(t)=(I_t,R_t,\Delta R_t,G_t,E_t,P_t,T_t)
\]

with:

- `I_t`: current 64-bit instruction word;
- `R_t`: register snapshot;
- `ΔR_t`: register-bit Hamming transitions;
- `G_t`: estimated gate activity;
- `E_t`: modeled switching energy;
- `P_t`: modeled cycle power;
- `T_t`: first-order thermal state.

The transition is:

\[
\boxed{\widehat S_e(t+1)=F_e(\widehat S_e(t),I_t,R_t,R_{t+1})}
\]

---

## 2. Instruction-cycle mechanics

For each `CodexVM.step()`:

```text
FETCH instruction word
DECODE opcode and operands
Λ policy authorization
SNAPSHOT register state R_t
EXECUTE instruction
SNAPSHOT candidate register state R_t+1
COUNT register and instruction-bus bit transitions
ESTIMATE opcode-specific gate activity
CALCULATE timing, switching energy, power, and thermal state
APPLY electronic Λ gate
COMMIT ledger and trace
ADVANCE instruction pointer
```

When `ElectronicConfig.enforce_limits` is false, the model observes without altering legacy VM behavior. When enforcement is true, a failed electronic Λ decision restores the pre-instruction register snapshot and rejects the commit.

---

## 3. Bit-transition mechanics

Each integer is projected into an unsigned fixed-width word:

\[
W(x)=x\bmod 2^N
\]

The transition count between two values is the Hamming distance:

\[
H(a,b)=\operatorname{popcount}(W(a)\oplus W(b))
\]

Total register activity is:

\[
\Delta R_t=\sum_{r\in\mathcal R}H(R_t[r],R_{t+1}[r])
\]

Instruction-bus activity is:

\[
\Delta I_t=H(I_{t-1},I_t)
\]

These are deterministic consequences of the actual VM state and instruction stream.

---

## 4. Gate-activity model

The current backend models these logical resources:

```text
AND OR XOR NAND NOT MUX FF
```

Opcode mappings are intentionally explicit:

- `SET`: decode, immediate bus, multiplexing, and destination flip-flop writes;
- `ADD`: full-adder XOR, AND, OR, mux, and register activity;
- `SUB`: operand inversion plus full-adder activity;
- `HALT`: bounded control-path activity;
- unknown opcode: conservative decoder and routing estimate.

For an `N`-bit adder, the baseline logical estimate includes:

\[
G_{XOR}=2N,\qquad G_{AND}=2N,\qquad G_{OR}=N
\]

These are activity proxies, not physical standard-cell counts.

---

## 5. Timing model

The clock period is:

\[
T_{clk}=\frac{1}{f}
\]

The available timing budget is:

\[
T_{usable}=T_{clk}(1-g)
\]

where `g` is the timing guard fraction.

The current ADD/SUB backend uses a conservative ripple-carry estimate:

\[
T_{critical}=(N+3)t_g
\]

and the timing margin is:

\[
M_t=T_{usable}-T_{critical}
\]

The timing decision is:

\[
V_{timing}=[M_t\ge 0]
\]

The model can later be replaced by carry-lookahead, vector-lane, FPGA, ASIC, or measured timing backends without changing the VM-facing API.

---

## 6. Energy and power model

Dynamic switching energy per cycle is:

\[
E_{dyn}=N_{toggle}CV^2
\]

where:

- `N_toggle`: total estimated switching events;
- `C`: configured switched capacitance per event;
- `V`: supply voltage.

Dynamic power is:

\[
P_{dyn}=\frac{E_{dyn}}{T_{clk}}
\]

Total modeled power is:

\[
P_{total}=P_{static}+P_{dyn}
\]

Cumulative energy includes static and dynamic terms:

\[
E_{cum,t+1}=E_{cum,t}+E_{dyn}+P_{static}T_{clk}
\]

---

## 7. Thermal model

The thermal target is:

\[
T_{target}=T_{ambient}+P_{total}R_\theta
\]

A first-order discrete update is used:

\[
T_{j,t+1}=T_{j,t}+\alpha(T_{target}-T_{j,t})
\]

with:

\[
\alpha=\min\left(1,\frac{T_{clk}}{\tau_\theta}\right)
\]

The thermal validity condition is:

\[
V_{thermal}=[T_j\le T_{max}]
\]

---

## 8. Electronic Lambda gate

The electronic gate is:

\[
\boxed{\Lambda_e=V_{timing}\land V_{thermal}}
\]

The emitted telemetry field is `lambda_accept`.

With enforcement enabled:

```text
if Λe == false:
    restore R_t
    reject instruction commit
```

This gives Jarvis-X a transactional boundary between abstract execution and declared electronic constraints.

---

## 9. Telemetry record

Each cycle emits:

```text
cycle
opcode
register_bit_transitions
instruction_bus_bit_transitions
gate_toggles
total_gate_toggles
dynamic_energy_j
cumulative_energy_j
dynamic_power_w
total_power_w
junction_temp_c
critical_path_ns
clock_period_ns
timing_margin_ns
timing_ok
thermal_ok
lambda_accept
register_checksum
source = deterministic-model
```

The `source` marker and `telemetry_is_measured = false` prevent modeled values from being misrepresented as hardware observations.

---

## 10. Runtime use

```python
from jarvisx.core import CodexVM
from jarvisx.electronic import ElectronicConfig, ElectronicSubstrate

substrate = ElectronicSubstrate(
    ElectronicConfig(
        word_bits=64,
        clock_hz=1_000_000_000.0,
        supply_voltage_v=0.90,
        enforce_limits=True,
    )
)

vm = CodexVM(electronics=substrate)
vm.load(bytecode)
vm.run()

print(vm.last_electronic_trace.to_dict())
print(vm.electronics.snapshot())
```

---

## 11. Full permeation chain

```text
Operator intent
    ↓
Source program
    ↓
Assembler and 64-bit instruction word
    ↓
Decoder and control selection
    ↓
Register-transfer execution
    ↓
Bit-transition field
    ↓
Gate-activity estimate
    ↓
Timing / energy / power / thermal projection
    ↓
Electronic Λ validation
    ↓
Ledger, trace, memory, and output
    ↓
Next instruction cycle
```

The closed-loop invariant is:

\[
\boxed{\text{No abstract VM state changes without a corresponding modeled electronic state transition.}}
\]

---

## 12. Extension points

The deterministic interface is designed for progressive replacement by real backends:

1. Linux `perf_event_open` counters;
2. Intel RAPL or AMD energy counters;
3. NVIDIA NVML power and thermal telemetry;
4. FPGA post-place-and-route timing and switching reports;
5. SPICE or standard-cell characterization;
6. cache and DRAM transaction counters;
7. eBPF kernel scheduling and syscall traces;
8. oscilloscope, logic-analyzer, or board telemetry streams.

Measured backends must preserve provenance and mark values as measured, estimated, inferred, or simulated.
