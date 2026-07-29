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
- repository templates, ownership, dependency automation and citation metadata.

### Changed

- ordinary VM runs use an in-memory journal unless persistence is explicitly requested;
- reflex stabilization is opt-in so ordinary assembly semantics remain authoritative;
- packaging metadata is authoritative in `pyproject.toml`;
- supported Python versions begin at Python 3.10;
- CI uses current Node 24-compatible major versions of core actions;
- project documentation distinguishes stable code, integration candidates, demonstrations and specifications.

### Fixed

- raw `bytes` objects in ledger entries were not JSON serializable;
- pytest used the invalid coverage reporter `term-local`;
- tests could create or mutate an `omega_ledger.json` file in the working directory;
- VM loading accepted empty and non-64-bit programs;
- invalid instruction-pointer states were not rejected explicitly.

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
