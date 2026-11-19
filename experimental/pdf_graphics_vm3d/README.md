# JARVIS-X PDF Graphics VM3D (bounded research reference)

This isolated C++20 module operationalizes a narrow, testable portion of the proposed **Dr. Moagi 3D PDF-bytecode equation**. It reads a **decompressed PDF page content stream** (not a complete `.pdf` file), lexes and parses a defined subset, lowers operators into 16-byte records with external operands, constructs an ordered geometry command plan, renders a 128x128 RGB PPM, and emits deterministic instruction + virtual 3D tile traces. A measured and pixel-checked decision chooses the specialized plan or the reference interpreter.

## Reproduce

```sh
cmake -S experimental/pdf_graphics_vm3d -B build/pdf_vm3d -DCMAKE_BUILD_TYPE=Release
cmake --build build/pdf_vm3d --parallel
ctest --test-dir build/pdf_vm3d --output-on-failure
build/pdf_vm3d/jx_pdf_graphics_vm3d --demo out.ppm trace.csv
build/pdf_vm3d/jx_pdf_graphics_vm3d --content experimental/pdf_graphics_vm3d/sample.content out.ppm trace.csv
python3 experimental/pdf_graphics_vm3d/audit.py out.ppm trace.csv
```

`out.ppm` is an uncompressed P6 RGB image; most image programs can open it. `trace.csv` has a unified header for opcode events and emitted polygon coordinates. Benchmark results (ns per complete 128x128 rendering execution) are observed on the executing machine, not portable guarantees.

## Supported PDF content operators

| Operator | Reference semantics |
| --- | --- |
| `q`, `Q` | Bounded graphics-state stack, max depth 64 |
| `cm` | Concatenate current transformation matrix (CTM) |
| `BT`, `ET`, `Tm` | Text object lifecycle and text matrix |
| `Tf`, `Tj` | Font-size selection and text placement using **demonstration 5x7 font** |
| `re`, `f` | Build rectangle path under CTM at construction; nonzero fill for simple rectangle |
| `rg` | Set opaque nonstroking RGB fill color |

Strictly rejects unknown PDF operators, malformed operand types/counts, nonfinite numbers, and unbalanced text/graphics states. Limits: source <= 1 MiB, <= 100,000 operators, <= 300,000 polygons, <= 16,384 bytes per text token, fixed 128x128 canvas. This parser does not implement all PDF syntax or a PDF file's Xref/stream/object infrastructure. It is **not appropriate for untrusted PDF ingestion without a complete PDF security review and additional isolation**.

## 3D geometry and typed IR

A 128-bit `Instruction` is a 16-byte, fixed-width **descriptor** with opcode, operand count, number-pool offset, string-pool offset, and reserved field. Six double-precision operands do **not** fit inline in 128 bits. The operand pools keep the required values outside the descriptor. The original content stream is retained byte-for-byte by the `Program` object; no file-size compression is asserted.

A rendered polygon is indexed at logical tile coordinate `(floor(min_x/16), floor(min_y/16), draw_sequence)`, with coordinates bounded in `[0,10^6)`. A 10^18-coordinate lattice is **virtual**: only emitted polygons have addresses; no dense 10^18 voxel array is allocated. The `z` axis denotes execution/layer order, **not physical depth**. Pixel painting preserves the original emission order.

The coordinate rule for a PDF 2D matrix is `x'=a*x+c*y+e`, `y'=b*x+d*y+f`. The CTM and text matrix remain separate state; `CTM * Tm` is computed at text drawing. The rectangle path's geometry is frozen under the CTM that existed at `re`, even if a later `cm` precedes `f`. Thus no unsafe global `cm + Tm` conflation is performed.

## Compiler feedback, without unsupported claims

- **Reference tier:** execute typed operators and lower them to polygons for every frame.
- **Specialized tier:** precompute the polygon sequence once from typed IR, then execute its rasterizer without repeated PDF operator dispatch. This is **static plan specialization**, *not* native machine-code JIT compilation.
- **Gate:** compare framebuffer bytes and input identity; measure the median of three baseline/specialized timing batches (100 repetitions by default). Promote the specialized plan only if bitwise-equal pixels and a lower measured time are observed. Microbenchmark results are workload-dependent and noisy.
- **Trace:** record per-operator provenance, FNV-1a 64-bit hashes, emitted polygons, and virtual `(x,y,z)` tile coordinates. FNV is not a cryptographic integrity or authenticity primitive.

`(S_{k+1}, F_k) = M_3D(S_k,B_k)`, with typed IR, virtual geometry, measured timings, and a conditional optimization decision. Full JIT code emission, GPU dispatch, 3D learned autoencoding, PDF xref parsing, predictive prefetch, glyph-atlas rasterization, and hardware bandwidth metrics remain **future integration work**. Neither 0.8 ms latency, 1 million operations per second, nor 10^18 physical ops/sec is claimed.

## Attribution and integration boundary

Research implementation of the user-proposed Dr. Moagi / JARVIS-X system model (2026-11-19). All files are confined to `experimental/pdf_graphics_vm3d/` and a path-scoped CI workflow. Existing Jarvis-X PDF-transport, DM3D bytecode VM, and canonical codecs are not modified. Review and testing must precede any merger into `main`.
