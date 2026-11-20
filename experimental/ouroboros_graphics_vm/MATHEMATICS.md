# JARVIS X — Operational and mathematical decomposition of the uploaded React artifact

## Scope and evidence

Inspected source: `React Artifact(20261120-161245).html` (bundle, rendering component, VM implementation and UI controls). The C++17 file `jarvis_x_3d_vm.cpp` is a standalone **CPU reference implementation** of the inspectable graph construction, per-neuron activations, event packets, graphics instruction set, framebuffer and feedback mechanics. It deliberately omits the React interface, browser canvas projection, hosted records storage and event handlers; it exports PGM, PLY and CSV instead. Its input validation and bounded execution are stricter than the source. It does not train or infer with a weight-matrix ANN.

## 1. Full state model

Let

\[
S_t=(PC_t,ACC_t,R_t,SP_t,Stack_t,M_t,V_t,G_t,P_t,\tau_t,C_t).
\]

- `PC`: next instruction address.
- `ACC`: accumulator, mirrored by successful destination writes.
- `R`: eight 64-bit signed general registers `R0..R7` (JavaScript original uses `Number`).
- `SP`, `Stack`: call stack / push-pop stack.
- `M`: 256-byte working memory, storing `ACC & 255` at `PC mod 256` on each completed instruction.
- `V`: 16,384-element unsigned-byte grayscale framebuffer, arranged as 128×128.
- `G=(N,E)`: toroidal-layer neuron positions, activation states and directed connections.
- `P`: transient propagating packet list (capped at 900 active packets).
- `\tau`: continuous display time; `C`: integer VM cycle counter.

The VM performs a one-instruction transition

\[
S_{c+1}=\operatorname{Execute}(\operatorname{Decode}(I_{PC_c}),S_c),
\quad PC_{c+1}=\begin{cases}j &\text{branch/call/return},\\PC_c+1 &\text{otherwise.}\end{cases}
\]

The renderer is a separate animation-time process. The six pipeline labels `FETCH → DECODE → EXECUTE → RASTERIZE → SHADE → FEEDBACK` appear in the browser UI; the code highlights them by `cycles mod 6`. They are *not* independently executed six-stage hardware pipeline instructions.

## 2. Parametric 3D neural geometry

For `L` layers, requested visible node count `N`, and loop depth `d`, let `m=ceil(N/L)` and let `j=0..m−1`, `ℓ=0..L−1`. Then

\[
\begin{aligned}
s&=1-0.055(d-1), &R&=430s,\\
\phi_j&=2\pi j/m, &\theta_{\ell j}&=2\pi\ell/L+0.12(d-1)\sin\phi_j,\\
r_\ell&=285s(1-0.42\ell/L),\\
x_{\ell j}&=(R+r_\ell\cos\phi_j)\cos\theta_{\ell j},\\
y_{\ell j}&=0.92(R+r_\ell\cos\phi_j)\sin\theta_{\ell j},\\
z_{\ell j}&=r_\ell\sin\phi_j+28\sin(d\theta_{\ell j}).
\end{aligned}
\]

The edge set includes sparse edges from layer `ℓ` to layer `ℓ+1`, and explicit return edges from layer `L−1` to layer `0`. These are geometrically plotted logical links, *not* a trained synaptic weight matrix. Default parameters: `N=6500`, `L=5`, `d=3`.

## 3. Activation and feedback dynamics

For neuron `i` with layer `ℓ_i`, phase `φ_i` and transient boost `b_i(t)`, the source uses

\[
z_i(t)=1.15\sin(1.8t-0.85\ell_i+0.4\phi_i)
       +2.2b_i(t)+0.6P_t\sin(5t-1.4\ell_i),
\]

where `P_t∈{0,1}` identifies a user-triggered pulse frame. Apply a chosen nonlinear function `σ` (ReLU, sigmoid with slope 2, tanh, or tanh-approximated GELU) and a source-specific rescaling `T_σ`. Then

\[
a_i(t)=\operatorname{clip}_{[0,1]}\big(T_\sigma(\sigma(z_i(t)))\big),
\quad b_i\leftarrow 0.985b_i.
\]

Activity in telemetry is counted when `a_i > 0.58`. VM `DRAW_PIXEL` raises one node's `boost` to 1 by a pixel index mapping; `DRAW_TRI` boosts six indexed nodes. When a packet completes edge `i→j`, the target's boost rises by `0.45` for an ordinary edge and by `0.45g` for a returning edge, capped at 1. During the VM `FEEDBACK` instruction, **all neurons' boosts increase by `0.12g`**, capped at 1. The source creates packets probabilistically per browser frame and injects 28 more upon a pulse. Packet progress evolves as

\[
p_{k,t+\Delta t}=p_{k,t}+v_k\,\Delta t\,s_{\rm speed}.
\]

The node/edge values described above support a dynamic visualization, but there is no parameter-gradient update, supervised target, or learned model in the inspected code.

## 4. Graphics VM: actual instruction semantics

| Class | Instructions | Implementation |
|---|---|---|
| Arithmetic | `LOAD MOV ADD MUL SIN` | Read operands; write destination and `ACC` |
| Flow | `JMP JZ CALL RET HALT NOP` | Modify `PC`; `JZ` checks `ACC == 0` |
| Stack | `PUSH POP` | Bounded stack, checked for underflow |
| Raster | `DRAW_PIXEL DRAW_TRI CLEAR NOISE` | Modify the 128×128 8-bit framebuffer |
| Feedback | `FEEDBACK` | Scale each framebuffer pixel; increase node boosts |

Raster address of pixel `(x,y)`:

\[
x'=((x\bmod128)+128)\bmod128,\quad y'=((y\bmod128)+128)\bmod128,
\quad\text{index}=128y'+x'.
\]

`DRAW_TRI` uses a barycentric point-in-triangle test over a clipped bounding box and writes an 8-bit intensity. The framebuffer's feedback transformation is

\[
V_{c+1}[k]=\operatorname{clip}_{[0,255]}
\big(\operatorname{round}[(0.92+0.08g)V_c[k]]\big).
\]

For `g=0`, a 100-intensity pixel becomes 92; `g=1` preserves 100; `g=2` becomes 108. Repeated application can damp, preserve or saturate values depending on rounding and clipping. This is **multiplicative feedback, not a loss-optimized encoder-decoder**. The source increments its VM `frame` every 48 executed instructions. CPU simulation instructions and visual animation frames are distinct clocks.

## 5. Logical scale versus physical complexity

The UI title is `1,000,000^(1,000,000)` pixels, which algebraically equals `10^6,000,000`. Its *displayed* simulated pixel indicator is an increasing exponent `e(t)=round(6,000,000u(t))`, with `u` initially about `0.012` and incrementing at a fixed UI-clock rate. Thus a label like `10^74,160` is a **progress indicator over a logarithmic exponent scale**, not a real allocation of that many pixels or a proof of encoding capacity.

For `N` graph nodes, `E` directed edges, `P` active packets, a framebuffer size `B=128²`:

- Graph initialization: `O(N + E)` time and space.
- Per display frame: `O(N + P)` for activity and packets in the headless C++ runtime; browser drawing additionally traverses edges and depth-sorts nodes (`O(E + N log N)`).
- `DRAW_PIXEL`, scalar arithmetic: `O(1)` per instruction.
- `FEEDBACK`: `O(B+N)` per instruction, with 16,384 pixel updates and one boost per node.
- `DRAW_TRI`: at most `O(B)` pixel tests, bounded by clipped triangle extent.
- `NOISE #n`: `O(n)` random writes (bounded to 1,000,000 per instruction in this reference).
- Typical working allocation: thousands of nodes/edges plus the fixed framebuffer; nothing comparable to the logical headline exponent.

## 6. Reproducible run

```bash
g++ -std=c++17 -O2 -Wall -Wextra -pedantic jarvis_x_3d_vm.cpp -o jarvis_x_3d_vm
./jarvis_x_3d_vm --self-test
./jarvis_x_3d_vm --program pulse --cycles 2400 --frames 120 --out jarvis_x_demo
./jarvis_x_3d_vm --program triangle --cycles 2400 --frames 120 --depth 5 --gain 0.8
./jarvis_x_3d_vm --asm my_code.asm --cycles 1000 --frames 60
```

Outputs: `_framebuffer.pgm` (8-bit grayscale image), `_neural_field.ply` (3D coordinates + edges, view in a compatible mesh viewer), and `_telemetry.csv` (measured counters, mean activation, **frame-to-frame normalized framebuffer MSE**, and FNV-1a hash). The MSE is calculated from consecutive rendered buffers; it is **not an ANN learning loss**.

### Local validation executed on the supplied source reconstruction

- Built successfully with GCC C++17 (`-O2 -Wall -Wextra -pedantic`).
- Passed 11 source-derived/self-consistency test groups, including ACC/register arithmetic, branching/subroutines, drawing, feedback gains, seeded noise, assembler failure paths, graph bounds, pulses and 48-cycle frames.
- With the default pulse program, `--cycles 2400 --frames 120`: **6500 geometry nodes, 8738 graph edges, 2400 executed instructions, 50 VM frames**, framebuffer FNV-1a64 **`d8bee218252570a5`**. The final field saturated at 6500/6500 active nodes under repeated all-node feedback (a property of the reference dynamics, not a neural-learning achievement).

## 7. Differences from original HTML/JS

1. C++ is intentionally **headless** and exports files; canvas rendering, camera orbit and mouse interactions remain browser-specific.
2. It uses signed 64-bit registers and checked overflow; the JS source uses IEEE-754 `Number` and permissive input parsing.
3. Invalid operands, unknown symbols, stack underflow, excessive values and invalid jumps raise explicit errors instead of silently becoming zero or running unpredictably.
4. Fixed seeds enable repeatable runs within the same compiler/standard-library environment. Exact cross-platform RNG streams are not guaranteed.
5. It exports actual state-change MSE in place of the source UI's cosmetic synthetic `loss ≈ 2.45 exp(−epoch/11) + 0.09 + noise`. No backpropagation or optimizer has been invented.
6. Its 3D field positions and connections are equivalent formulas rather than a byte-for-byte recreation of JavaScript random draws; floating-point, sampling and output colors may differ.

**Status:** Verified C++ prototype of the explicit operational logic; not a pretrained ANN, universal 3D AI processor, lossless autoencoder, or experimentally established extreme-scale computing system.
