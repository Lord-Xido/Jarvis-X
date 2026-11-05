# G3M isolated module — local validation record

Date: 2026-11-05. **Local tests only; GitHub CI is separately reported.**

## Bytecode plan

- `python3 bytecode/emit_plan.py` generated 272-byte `G3MBC001` sample: 32-byte header + 15 instructions × 16 bytes.
- `g++ -std=c++17 -O2 -Wall -Wextra -pedantic bytecode/g3m_disasm.cpp` compiled.
- Valid plan accepted; wrong CRC, illegal opcode with recomputed CRC, and invalid point bounds with recomputed CRC are rejected.
- The expanded unittest script also checks register dependencies, mutation of kNN cardinality, instruction ordering, and truncated files.

## LibTorch C++ reference

- Previous build: CPU LibTorch, C++17, Clang 17; package-level build in original local artifact.
- Fresh standalone `tests/test_engine.py` executed against the previously compiled binary and passed checkpoint train/reload output equality, truncated/non-finite input rejection, and bounded-support 1,000,000-input / 1,000,000-output run (12,000,000-byte float32 XYZ output).
- The neural path is **lossy** and uses a selected support. The data are synthetic and results do not establish generalization.

## Not yet proved

- Bytecode-driven numerical execution (dispatcher)
- GitHub-hosted LibTorch training/inference validation
- Geometric reconstruction quality on independent datasets
- Speedup versus a quality-matched baseline
- Authenticated distributed evidence, multi-tenant sandbox or cloud production deployment
