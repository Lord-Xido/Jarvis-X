# ROM Forge legacy-source provenance

This directory records the audit identity and disposition of the 16 top-level
artifacts supplied for the 2026-09-01 universal-bitcode permeation.

The original archives and scripts are intentionally not part of the live Python,
JavaScript, C++, deployment, or workflow namespaces. The audit found explicit
placeholder modules, eager import-time execution, a trillion-iteration prototype,
stubbed policy/routing behavior, unreviewed deployment templates, an empty image,
and a file named as a notebook that is not valid notebook JSON.

`manifest.json` preserves the exact top-level byte lengths and SHA-256 digests so
the supplied inputs can be identified later. The promoted, testable common layer
is implemented in `src/jarvisx/universal_bitcode.py`.

Disposition values:

- `promoted-concept`: a bounded primitive informed the canonical implementation;
- `hash-only-reference`: retained as provenance but not imported or executed;
- `excluded-live-code`: unsafe, placeholder, invalid, or deployment-specific code
  was deliberately kept out of canonical runtime paths.

The manifest is provenance metadata, not an authenticity signature or license
grant for the underlying artifact contents.
