# DM-vOmegaXi+ Recursive Cube Interpreter

Status: deterministic software reference layered on VMAD128 World Engine

The Recursive Cube Interpreter turns the inward/outward cube abstraction into a bounded executable hierarchy. It does not allocate a literal `1GB x 1GB x 1GB` physical cube. The canonical logical cube extent is `1,000,000,000` byte coordinates per axis inside the existing `2^33`-per-axis VMAD128 sparse address domain.

## 1. Operational recurrence

The interpreter executes:

```text
sparse source bytes
  -> 1024-byte tile
  -> ENC_LAT_VOL
  -> 32-byte latent
  -> shadow latent/output
  -> CALC_DELTA
  -> PROPOSE_BIAS
  -> VALIDATE
  -> authoritative commit or rollback
  -> next recursive level
  -> outward latent decode
  -> final sparse world output
```

The governing transition is:

```text
Describe -> Encode -> Reconstruct -> Compare -> Propose -> Validate -> Commit/Rollback -> Recurse
```

Accepted refinement is non-worsening at the VMAD byte-delta level, and repeated refinement is bounded by both an epsilon target and a maximum pass count. There is no unbounded `while(true)` execution path.

## 2. DMCUBE1 execution buffer

Control flow is carried in a dedicated execution buffer, not inferred from decoded data.

Header:

| Field | Bytes |
|---|---:|
| magic `DMCUBE1\0` | 8 |
| version | 2 |
| command count | 2 |
| reserved | 4 |

Every command is exactly 96 bytes:

| Field | Bytes |
|---|---:|
| opcode | 1 |
| hierarchy level | 1 |
| flags | 2 |
| tile count | 4 |
| maximum passes | 4 |
| epsilon | 4 |
| source VMAD128 | 16 |
| authoritative latent VMAD128 | 16 |
| authoritative output VMAD128 | 16 |
| shadow latent VMAD128 | 16 |
| shadow output VMAD128 | 16 |

An 8-byte FNV-1a digest terminates the entire buffer.

Implemented cube opcodes:

- `0x10 ENCODE_REFINE`
- `0x20 DECODE`
- `0xFF HALT`

Unknown opcodes, invalid magic/version, digest corruption, overlapping transaction spans, out-of-cube addresses, excessive tile counts, excessive refinement counts, missing `HALT`, commands after `HALT`, and aggregate step-budget overflow all fail closed before execution.

## 3. Inward tile transaction

For each 1024-byte source tile, the interpreter composes the existing VMAD128 micro-ops:

```text
LOAD_VMAD     source
LOAD_VMAD     shadow_latent
LOAD_VMAD     shadow_output
TILE_IN_VEC   1024 B
ENC_LAT_VOL   1024 B -> 32 B
STORE_VEC     32 B shadow latent
FUSE_ATTN     deterministic fixed-point fusion
DEC_PIX_VOL   32 B -> 1024 B shadow output
CALC_DELTA    source - reconstruction
PROPOSE_BIAS  shadow candidate
VALIDATE      previous-delta threshold
HALT
```

No authoritative latent or output bytes are written by this candidate program.

If Lambda accepts the candidate, the interpreter first stages the complete shadow spans in host memory, copies them to the authoritative latent/output VMAD ranges, and then executes `COMMIT_IF` for the adaptive bias candidate.

If the candidate fails Lambda or fails to improve after the first accepted pass, `COMMIT_IF` is issued against a reserved zero gate and the adaptive candidate is rolled back. Authoritative latent/output spans remain unchanged.

This preserves the operational invariant:

```text
candidate -> validate -> commit/rollback
```

The software reference is transactionally ordered but is not yet a crash-atomic persistent store; WAL/journaling remains a separate hardening boundary.

## 4. Recursive hierarchy

The byte hierarchy uses the already implemented `1024 -> 32` latent contraction, giving a 32:1 byte reduction per level.

For level `l`:

```text
N_l source tiles -> N_l latent vectors
N_(l+1) = ceil(N_l / 32)
```

Thirty-two 32-byte child latent vectors form the 1024-byte source quantum for the next level. Missing tail bytes are sparse zero/default state.

Conceptually:

```text
V0 --encode--> V1 --encode--> ... --encode--> VL
 |                                         |
 +<--decode--- ... <---------decode--------+
```

The outward pass starts from the highest committed latent stream and repeatedly expands each 32-byte latent vector into a 1024-byte lower-level representation until the final base-world output is produced.

## 5. Fixed-point rule

Let `d_t` be the mean absolute byte reconstruction delta for one tile.

The first candidate is admissible when the lower VMAD engine validates `d_t <= 255`. Subsequent passes must satisfy strict improvement:

```text
d_(t+1) < d_t
```

Refinement stops when either:

```text
d_t <= epsilon
```

or no strict improvement occurs, or `max_passes` is reached.

Thus the machine implements bounded admissible refinement rather than claiming infinite computation or guaranteed zero error.

## 6. Data/code separation

Decoded output is data. It is never interpreted as executable control merely because it contains opcode-like bytes or the `DMCUBE1` magic.

Only a separately supplied execution buffer that passes full `DMCUBE1` validation controls the interpreter. This is the executable form of:

```text
decoded bytes != executable instructions
```

unless they are explicitly staged as a control buffer and pass all validation gates.

## 7. CLI

CMake target:

```text
jarvisx-recursive-cube
```

Windows executable:

```text
DrMoagi-Recursive-Cube.exe
```

Example:

```powershell
.\DrMoagi-Recursive-Cube.exe --tiles 32 --levels 2 --state-dir .\cube-state
```

The demo seeds deterministic sparse bytes, constructs a disjoint recursive VMAD layout, executes the validated hierarchy, unfolds it outward, and reports execution-buffer validation, tile/pass counts, accepted/rejected refinements, committed latent bytes, world output bytes, recursive output MAE, and lower-engine commit/rollback counts.

## 8. Production claim boundary

Implemented and testable:

- sparse virtual cube semantics;
- VMAD128 addressing inherited from the World Engine;
- integrity-protected recursive execution buffer;
- explicit data/code separation;
- bounded multilevel encoding and outward decoding;
- candidate-first shadow latent/output state;
- Lambda-gated adaptive commit/rollback;
- finite convergence/step guards;
- deterministic telemetry and cross-platform regression tests.

Not established by this software reference:

- literal `1GB^3` resident physical memory;
- one physical processor per byte cell;
- silicon-photonic or TSV timing;
- sub-nanosecond pipeline execution;
- physical energy efficiency;
- SOTA superiority;
- patentability.

Those claims remain separate evidence gates.
