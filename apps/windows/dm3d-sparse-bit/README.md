# DM3D sparse-bit Windows integration candidate

This isolated Windows x64 console application implements bounded sparse bits on
three axes of 8,000,000,000,000 positions, synchronous seven-point GF(2) parity,
129-bit Morton addressing, bit-register bytecode, snapshots and local command routing.

It is an integration candidate with a separate D3VM format, not a replacement for
the canonical Python VM or C++ DM3D runtime. No cross-backend compatibility is
claimed and no private source document is distributed.

The default build is compact. The explicit -PadToOneGB option produces exactly
1,000,000,000 bytes (decimal 1 GB). The additional zero-filled reserve adds no
computation, AI model, functionality or memory. The executable is unsigned and
requires a compatible Windows x64 .NET Framework runtime.

## Build and test

From the repository root, on Windows with the .NET Framework x64 C# compiler:

    ./apps/windows/dm3d-sparse-bit/build.ps1
    python apps/windows/dm3d-sparse-bit/test_executable.py --exe apps/windows/dm3d-sparse-bit/build/DM3D_JARVIS_X.exe
    ./apps/windows/dm3d-sparse-bit/build.ps1 -PadToOneGB
    python apps/windows/dm3d-sparse-bit/test_executable.py --exe apps/windows/dm3d-sparse-bit/build/DM3D_JARVIS_X.exe --expected-size 1000000000

The script compiles and tests the executable, copies examples, creates a checksum,
and records executable and reserve sizes in build-manifest.json. Python 3.10+ is
required only for independent acceptance tests. Tests use a disposable directory
and never touch saved user state.

The Windows workflow tests both sizes and uploads a compressed artifact containing
the 1 GB executable, examples, checksum and acceptance reports. Download it from a
successful Actions run and extract it to a writable folder. Generated binaries and
snapshots are excluded from Git.

## Start

Double-click `DM3D_JARVIS_X.exe` for the interactive console. Enter `help` to list commands and `quit` to exit. Keep the executable in a writable folder; it creates `DM3D_Data` beside itself. One running instance per data folder is supported.

```text
write 2 2 2 1
step
status
cells 7
hash
save my-state.dm3d
```

This creates seven active bits after one step. `load my-state.dm3d` restores them. State is held in memory until explicitly saved; closing the application discards unsaved changes. A failed instruction preserves previous successfully executed instructions. A failed STEP preserves the state before that STEP. A multi-step command may commit earlier successful steps before a later step fails.

From PowerShell, in the executable folder:

```powershell
.\DM3D_JARVIS_X.exe --self-test
.\DM3D_JARVIS_X.exe --batch .\DM3D_Demo.txt
```

The demo requires the included `DM3D_Data/demo.asm`. A normal interactive session and a batch each share one in-memory VM for their duration. Separate invocations start fresh and must load a snapshot when needed. Batch mode stops on the first error and returns exit code 1. Success returns 0. Interactive errors are displayed and the console remains open.

## Routing scope

All commands entered into this application go through its local dispatcher: reads, writes, STEP, snapshots, assembly, bytecode execution and hashes. The broker validates command operands, file names and resource limits and writes BEGIN plus OK or FAILED audit records. A BEGIN without a completion record identifies an interrupted operation; it is not automatically replayed. Logs omit payload contents.

No external application, ChatGPT conversation, operating system traffic, email account or cloud service has been rerouted. No network listener, startup hook, proxy, credential access or external adapter is installed. Connecting an external operation requires its exact source application, action and integration mechanism. This application implements the document's local virtual computer, not a general AI assistant or a whole-machine replacement.

## Commands

| Command | Result |
|---|---|
| `status` | Axis size, iteration and active bit count |
| `reset` | Clear state and iteration |
| `read x y z` | Read a zero-default bit |
| `write x y z bit` | Store 0 or 1 |
| `step [count]` | Run synchronous parity updates |
| `cells [limit]` | Show sorted active coordinates, default 20, maximum 1,000 |
| `hash` | SHA-256 of canonical state bytes |
| `save name` / `load name` | Atomically save or validate then load a snapshot |
| `assemble source.asm output.d3b` | Assemble a source file from DM3D_Data |
| `run program.d3b` | Verify and execute bytecode from DM3D_Data |
| `morton x y z` / `unmorton integer` | Reversible 129-bit Morton address mapping |
| `selftest` | Execute built-in verification tests |

File commands accept plain file names within `DM3D_Data`, not arbitrary paths. The router rejects direct linked files and a linked data folder. This is a trusted local-user application; file scoping is not an operating-system sandbox against another process modifying the directory concurrently. Keep untrusted users from writing to the application data folder.

## ISA and binary format

Sixteen one-bit registers R0 to R15 initialize to zero for each run. Instructions are LOAD r x y z, STORE x y z r, XOR destination source1 source2, MOV r bit, JNZ r target, STEP, HASH and HALT. Source is one instruction per line, with optional `#` comments. Targets are zero-based instruction indices. HASH stores the most recent digest for the final HALT report. Program registers are ephemeral; spatial memory persists for the session.

The bytecode file starts with four ASCII bytes `D3VM` and a little-endian unsigned-compatible positive Int32 instruction count. This signature denotes the initial format. Each 32-byte instruction consists of opcode/A/B/C as four bytes, four reserved zero bytes, and X/Y/Z as three little-endian UInt64 values. Opcodes 1 to 8 correspond to the instruction list above. LOAD and STORE use A and XYZ. XOR uses ABC. MOV uses A and X, where X is 0 or 1. JNZ uses A and X as the target. STEP, HASH and HALT have no operands. All unused fields must be zero. The verifier checks every instruction and rejects reachable fallthrough or a program with no reachable HALT. Reachable HALT alone does not establish termination; execution limits stop loops.

Coordinates are stored without 64-bit address packing; each axis fits in a signed 64-bit value. Morton indices use arbitrary-precision BigInteger values. Snapshot serialization is compatible with the document: ASCII `DM3D`, version byte 1, 6-byte big-endian axis, 4-byte big-endian iteration, 8-byte big-endian active count, then lexicographically sorted triples of 6-byte big-endian coordinates. Total size is `23 + 18 * active_count`. The loader rejects noncanonical order, duplicates, out-of-bounds coordinates, trailing or truncated bytes, and quota violations. Unlike the document's prototype decoder, it never raises resource quotas based on untrusted file contents.

## Limits and failure behavior

- 100,000 active cells; 100 cumulative iterations per reset; axes at most 8 trillion.
- STEP toggles contributions into a fresh set with at most 700,000 transient entries. It commits only if the result fits the active quota.
- 100,000 instructions per program; 1,000,000 executed instructions per run.
- Cooperative ten-second deadlines for STEP and program execution; checks occur at bounded loop checkpoints, not hard real-time process termination.
- Snapshot inputs at most 1,800,023 bytes; bytecode at most 3,200,008 bytes; assembly input at most 4,000,000 bytes.
- Batch input at most 1,000,000 bytes and 10,000 lines. These limits do not guarantee that a batch of many expensive commands completes quickly.
- New saves require fewer than 256 existing data files and account for a 256 MiB data-folder budget including temporary save bytes. Audit records stop new logged commands after approximately 4 MB; archive the audit file with the application closed.
- Snapshot saves use a flushed temporary file and atomic replacement on the tested local filesystem. SHA-256 detects changes but is not publisher authentication. Snapshots are not encrypted.

## Validation and implemented boundary

The built-in suite has 3,112 assertions including dense-reference parity comparisons,
Morton round trips, serialization, bytecode validation and quotas. Python tests
independently check canonical bytes and SHA-256, malformed inputs, file scoping,
batch termination and atomic rejection of excessive STEP growth. Package length is
checked explicitly when testing the padded build.

This candidate does not implement GPU processing, paging, a graphical viewer,
parallel scheduling, external adapters or automatic crash recovery. Audit timestamps
and cooperative deadlines depend on the host clock. Guest state evolution and
canonical snapshots are deterministic when execution completes within those limits.
