# DMSO Autonomous OS-Style Runtime

## Status

Executable C++17 reference subsystem for PR #84.

This layer turns the DMSO research runtime into one self-contained userspace lifecycle environment containing:

- fixed register arrays;
- a 64-bit instruction format;
- bytecode compiler/decompiler functions;
- a procedural geometry field;
- a shading field;
- forward execution;
- numerical reverse optimization;
- transactional candidate commit/rollback;
- an asynchronous lifecycle supervisor;
- bounded convergence and telemetry reporting;
- CTest regression coverage.

It deliberately uses only the C++ standard library. No NumPy, scripting wrapper, external configuration file, or generated runtime dependency is required.

## Important boundary

`AutonomousSystemOS` is an **OS-style userspace machine environment**, not a bootable operating-system kernel and not a hardware-isolation boundary. Its VRAM, registers, bytecode and scheduler are software abstractions executing inside the host process.

The name describes the lifecycle role: it owns its virtual hardware state, program image, target buffer, execution loop, optimizer, validation gates and shutdown report.

## 64-bit instruction layout

```text
63                            56 55              40 39              24 23       0
+------------------------------+------------------+------------------+-----------+
| opcode: 8 bits               | alpha: 16 bits   | beta: 16 bits    | reserved  |
+------------------------------+------------------+------------------+-----------+
```

The reference instruction set is intentionally small:

```text
0x80  RECURSE_SPACE
0x90  EVAL_FIELD_GEO
0xA0  EVAL_FIELD_TEX
```

Unsupported opcodes are rejected before execution.

## Virtual registers

`VirtualVRAMRegisters` owns:

```text
REG_PC   program counter
REG_F    RGB frame buffer
Z_BUF    depth surface
THETA_G  geometry parameters
THETA_T  shading parameters
```

Resolution is bounded to `[1, 512]` to prevent accidental unbounded allocation in the reference implementation.

## Forward execution

The default program is assembled internally at boot:

```text
RECURSE_SPACE -> EVAL_FIELD_GEO -> EVAL_FIELD_TEX
```

`EVAL_FIELD_GEO` samples a bounded procedural signed-distance-like field over sixteen fixed depth planes. `EVAL_FIELD_TEX` shades occupied samples into the RGB register buffer.

The geometry field is:

```math
f(x,y,z)
=
\sqrt{x^2+y^2+z^2}-r
+
a\sin(\omega x)\cos(\omega y)\sin(\omega z).
```

The shading field is a bounded procedural RGB map controlled by `THETA_T`.

## Reverse optimization

The prototype equation used numerical central differences. The operational reference preserves that mechanic but makes it transactional.

For each parameter `theta_i`:

```math
\frac{\partial L}{\partial \theta_i}
\approx
\frac{L(\theta_i+\varepsilon)-L(\theta_i-\varepsilon)}{2\varepsilon}.
```

A shadow candidate is then formed from clipped gradients.

The update is committed only when:

```text
candidate loss is finite
AND
candidate loss <= baseline loss
```

Otherwise `THETA_G` and `THETA_T` are restored exactly to the pre-optimization snapshot.

This means the authoritative optimization trajectory is monotone non-worsening with respect to the measured MSE objective for each accepted epoch.

## Lifecycle

The complete machine loop is:

```text
BOOT
  -> allocate virtual registers
  -> assemble internal 64-bit program
  -> construct internal target frame
  -> validate program and buffers
  -> FORWARD
  -> finite-difference SHADOW OPTIMIZATION
  -> validate candidate
  -> COMMIT or ROLLBACK
  -> check convergence
  -> repeat within bounded epoch budget
  -> emit lifecycle report
  -> HALT
```

The asynchronous entry point uses `std::async(std::launch::async, ...)`. The authoritative lifecycle method is protected by a mutex so concurrent callers cannot mutate the same virtual register bank simultaneously.

## Telemetry

`LifecycleReport` records:

- epochs executed;
- accepted optimization updates;
- rejected updates;
- initial loss;
- final committed loss;
- elapsed host time;
- convergence status;
- final geometry parameters;
- final shading parameters.

Elapsed time is telemetry only. It does not participate in correctness or commit decisions.

The runtime reports `converged=true` only when the committed loss reaches the configured threshold. Exhausting the epoch budget is not mislabeled as convergence.

## Build and run

From the repository root:

```bash
cmake -S cpp_runtime -B build/cpp-runtime -DCMAKE_BUILD_TYPE=Release
cmake --build build/cpp-runtime --parallel
ctest --test-dir build/cpp-runtime --output-on-failure
```

Run the self-contained environment with defaults:

```bash
./build/cpp-runtime/jarvisx-dmso-os
```

Or provide resolution and epoch count:

```bash
./build/cpp-runtime/jarvisx-dmso-os 32 20
```

The executable prints a compact JSON lifecycle receipt.

## Corrections relative to the Python sketch

The C++ operational form intentionally fixes several prototype hazards:

1. the supplied Python import line was syntactically collapsed;
2. `asyncio.run(runtime_environment...)` was outside the `if __name__ == "__main__"` guard and would fail when imported;
3. optimization updated authoritative parameter arrays before proving the candidate improved the objective;
4. the final diagnostic always printed `FUNCTIONAL / STABLE` even when the convergence threshold had not been reached;
5. unsupported opcodes and unbounded resolutions/epoch counts were not rejected;
6. multiple concurrent lifecycle calls could race on shared VRAM state;
7. NumPy made the claimed self-contained runtime dependent on an external numerical package.

The native reference addresses those issues without changing the central architectural idea.

## Capability boundary

This implementation demonstrates an integrated virtual machine and optimization lifecycle. It does not claim:

- a bootable general-purpose OS;
- physical GPU/VRAM isolation;
- exact differentiable rendering through discontinuous hit masks;
- analytic backpropagation;
- real-time guarantees;
- unrestricted self-modifying native code;
- autonomous safety or correctness beyond its explicit invariants;
- consciousness or general intelligence.

The current reverse path is numerical finite-difference optimization. A later production acceleration layer could replace it with analytic/automatic differentiation while preserving the same shadow-state commit contract.
