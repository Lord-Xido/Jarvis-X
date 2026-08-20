# Changelog

All notable user-visible changes to Jarvis-X will be documented here.

The project follows semantic versioning where practical during alpha development.

## [Unreleased]

### Added

- verifiable JSON-native Omega ledger entries;
- atomic persistent-ledger writes;
- deterministic clock injection for replay fixtures;
- explicit VM lifecycle and bytecode validation;
- expanded VM and ledger regression tests;
- enforced CI quality, test and dependency-audit jobs;
- architecture, project-status, roadmap, governance, security and contribution documents;
- repository templates, ownership, dependency automation and citation metadata;
- dependency-free C++17 processor laboratory with a sparse virtual `8192³` lattice;
- signed 3-bit feature encoding, sparse scatter/diffusion, decoding and residual correction;
- bounded deterministic genome and bytecode-schedule candidate evaluation;
- atomic genome, ROM and evolution-journal checkpoints;
- cross-platform GCC, Clang sanitizer and MSVC validation;
- dependency-free periodic fractional 3D smoothing reference;
- public forward/inverse 3D DFT and periodic Laplacian spectral primitives;
- analytic constant-forcing updates for `du/dt = -D(-Delta)^alpha u + Omega`;
- `2×2×2` restriction, prolongation and coarse-to-fine fusion;
- ordered mechanistic traces, mass drift, gradient energy, residual and convergence telemetry;
- independent direct-DFT, spatial-stencil and semigroup validation tests;
- bounded deterministic 3D multiparallel package pipeline with sequential and process backends;
- versioned `JXMP` framing, bounded zlib decode and per-chunk SHA-256 verification;
- read-only Python code geometry, immutable branch snapshots and ordered typed merging;
- seeded candidate-first topology search with wall-clock timing excluded from fitness;
- `jarvisx-multiparallel` run, map-code and evolve command-line surface.

### Changed

- ordinary VM runs use an in-memory journal unless persistence is explicitly requested;
- reflex stabilization is opt-in so ordinary assembly semantics remain authoritative;
- packaging metadata is authoritative in `pyproject.toml`;
- supported Python versions begin at Python 3.10;
- CI uses current Node 24-compatible major versions of core actions;
- project documentation distinguishes stable code, reference laboratories, numerical references, integration candidates, demonstrations and specifications;
- C++ candidate selection excludes wall-clock latency from both fitness and equal-fitness tie handling;
- C++ checkpoint loading requires all versioned fields and a matching deterministic fingerprint;
- fractional smoothing documentation states dense storage and separable `O(N⁴)` cost for cubic grids explicitly.

### Fixed

- raw `bytes` objects in ledger entries were not JSON serializable;
- pytest used the invalid coverage reporter `term-local`;
- tests could create or mutate an `omega_ledger.json` file in the working directory;
- VM loading accepted empty and non-64-bit programs;
- invalid instruction-pointer states were not rejected explicitly;
- negative C++ genome mutations could wrap unsigned fields to their maximum bounds;
- the earlier C++ direct-build documentation referenced a nonexistent source file;
- C++ ROM and CSV state writes could leave partial output after interruption;
- the earlier fractional branch exposed a dead `_dft3` helper that always raised instead of a testable spectral API.

## [0.1.0] — Initial alpha

### Added

- Jarvis-X Python package and CLI entry point;
- assembly parser, assembler, decoder, registers and minimal VM executor;
- policy, sandbox, trace, reflex and ledger components;
- FastAPI and Uvicorn integration dependencies;
- sparse fractal-octree reference implementation;
- Hugging Face model configuration and safetensors export utilities;
- foundational Dr Moagi and reality-grounded architecture documents.

[Unreleased]: https://github.com/Lord-Xido/Jarvis-X/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Lord-Xido/Jarvis-X/releases/tag/v0.1.0
