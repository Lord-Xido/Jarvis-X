# JARVIS X — 3D Ouroboros Graphics VM (C++17 research reference)

This isolated prototype reconstructs the operational core of the supplied 2026-11-20 **React Artifact** in headless C++17. It implements a deterministic instruction interpreter, 128×128 single-channel framebuffer, raster operations, feedback gain and a finite layered 3D neuron visualization field.

**Scope:** The 3D nodes are analytically activated; there are no learned weights, optimizer, data-driven gradients, verified generalization or cross-platform bit-exact rendering promises. UI labels such as `1,000,000^(1,000,000)` are *logical/simulated* and no such tensor is allocated. The output MSE is consecutive-frame pixel difference, not model training loss.

## Build and execute

```bash
cmake -S experimental/ouroboros_graphics_vm -B build/ouroboros -DCMAKE_BUILD_TYPE=Release
cmake --build build/ouroboros --parallel 2
ctest --test-dir build/ouroboros --output-on-failure
build/ouroboros/jx_ouroboros_vm --program pulse --cycles 2400 --frames 120 --out /tmp/jx_pulse
python3 experimental/ouroboros_graphics_vm/audit.py /tmp/jx_pulse 120 2400
```

Invoke from the repository root. Output files are `_framebuffer.pgm`, `_neural_field.ply`, `_telemetry.csv`. PLY points and directed edges are a **3D logical/visual field**; rasterization occurs in the 2D framebuffer. `--help` lists all command-line options. Four presets: `pulse`, `triangle`, `plasma`, `spiral`. Custom `--asm FILE` programs use the embedded assembler.

## State transition

Let `s_c=(pc,acc,R,stack,M,V,G,P,c)`, where `R` contains eight signed 64-bit registers, `M` is 256 bytes, `V` is a 16384-byte framebuffer, `G` is the 3D graph, and `P` is a bounded packet list.

- Interpreter: `s_(c+1) = Execute(Decode(I_pc),s_c)` with checked 64-bit ADD/MUL and bounded stack.
- Framebuffer address: `idx = 128*((y mod 128+128) mod 128)+((x mod 128+128) mod 128)`.
- Feedback: `V'[k] = clamp(round((0.92+0.08*g)*V[k]),0,255)`; gain `g` is validated in `[0,2]`.
- Node activation: `a_i = clamp(T_sigma(sigma(1.15*sin(1.8*t-0.85*layer_i+0.4*phase_i)+2.2*boost_i+pulse_i)),0,1)`.
- VM frame count increments once per 48 executed instructions. Display ticks are separate.

Read [MATHEMATICS.md](MATHEMATICS.md) for geometry, full instruction set, feedback interpretation, complexity and limitations.

## Verification and governance

- `--self-test`: 11 test groups for arithmetic, register state, branch/call/return, coordinate wrap, triangle fill, gain scaling, deterministic seeded noise, assembler rejection, graph edge bounds, pulse and cycle framing.
- `audit.py`: independent stdlib checks for framebuffer dimensions, PLY header/index ranges and CSV timing/invariant consistency.
- Scoped GitHub workflow: GCC/Clang strict C++17 build, CTest, sample run/audit and Clang ASan/UBSan.
- No changes to existing Jarvis-X applications, deployment, permissions, existing VM protocols or cloud services.

**Hash note:** FNV-1a64 is a *diagnostic* checksum, not a cryptographic integrity proof. This repository addition should remain an unmerged draft until CI and code review succeed.
