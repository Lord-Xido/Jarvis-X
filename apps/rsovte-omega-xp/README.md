# RSOVTE-Ω/XP: sparse 3D emulator

A deterministic, bounded geometric computation module for Jarvis-X. This preserves the uploaded RSOVTE research concept while making its assumptions inspectable. It is an independent app, not a replacement for the VM, ANN, financial, vehicle or cloud subsystems.

## Run

Browser: from repository root run python -m http.server 8000 and open http://localhost:8000/apps/rsovte-omega-xp/ . A local server is needed for ES module imports. Drag to orbit; scroll to zoom; export source packets, audit JSON, and preview/4K/8K PNG.

CLI: node apps/rsovte-omega-xp/cli.mjs --count 20000 --seed 42 --iterations 1000 --out rsovte-audit.json --packet rsovte.rs3d

Replay: node apps/rsovte-omega-xp/cli.mjs --input rsovte.rs3d --iterations 1000

Tests: node --test apps/rsovte-omega-xp/test_core.mjs

Requires Node.js 20+ for CLI/tests, no npm dependencies.

## Operational contract

Input: n integer XYZ triples with each coordinate in [0,999999], n <= 200000.

Virtual address capacity: G³ = (1,000,000)³ = 10¹⁸ possible coordinate locations, not allocated memory. Number of folded blocks per axis is ceil(1,000,000 / 1024) = 977 including the partial terminal block. Memory is O(n), not O(G³).

RS3D source-coordinate packet is exactly 24 header bytes plus 8 bytes per point. Header (little endian): ASCII RS3D [0..3], u16 version 1 [4..5], u16 header size 24 [6..7], u32 count [8..11], u32 G [12..15], u32 block [16..19], u32 FNV-1a payload checksum [20..23]. Each 8-byte point stores x(20 bits), y(20 bits), z(20 bits), and 4 required-zero reserved bits. The two u32 words are lo = x | (y << 20) and hi = (y >>> 12) | (z << 8).

FNV-1a is only accidental-corruption detection; it is not a MAC, encryption, or adversarial tamper proof. Reversibility applies to source-coordinate packet, not geometric contraction or PNG. Packet size for n=20,000 is 160,024 bytes.

Geometric target is center of the original 1024-cell, clamped into the domain:

    c[a] = floor(source[a] / 1024)
    t[a] = min(999999, (c[a] + 0.5) * 1024)

Recurrent evolution:

    v[k+1,a] = q * w[a] * (t[a] - p[k,a]) + beta * v[k,a]
    p[k+1,a] = p[k,a] + v[k+1,a]

The characteristic polynomial of the per-axis error dynamics is:

    lambda² - (1 - q * w[a] + beta) * lambda + beta = 0

Bounded search scans q=0.20..1.55 by 0.01 and beta=0..0.20 by 0.005, minimizing maximum spectral radius across axes while rho < 1. A particle is pruned only after both positional error and transient velocity are <= 1e-7; snapping to target is disclosed with maximum snap error. Snapping is a deliberate deviation from exact trajectory, not lossless simulation.

Exactly the configured logical passes are evaluated. The audit reports physical point updates, skipped updates, fixed-point residual, preserved folded cells, finite-state and bounds checks, exact codec roundtrip mismatch count, and reproducible non-cryptographic state digest.

## Reference regression

Default: 20,000 particles, seed 42, weights (1,.8,1.25), 1,000 passes. A test asserts digest 3DC62705, zero codec errors, bounded positions, original-cell preservation and fixed point. The digest is a correctness regression rather than a security or performance claim. Investigate a failing reference rather than silently changing its expectation.

## Explicit limits

There is no trained neural network, autonomous reasoning, learned semantic codec, universal convergence theorem, hardware acceleration, or physical allocation of 10^18 cells. The kernel does not access external services, user credentials or devices. 8K PNG may require substantial browser memory. This module does not modify existing Jarvis-X subsystems.
