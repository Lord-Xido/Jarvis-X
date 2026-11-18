# DM-UGRE Omega3 — public integration facade (experimental)

A **public-safe, standalone demonstration** of generic federated connector interfaces, pre-publication permission/provenance gates, 3D state coordinates (phase, source, evidence), deterministic Morton addressing, bounded source calls, and trace events.

**Not the DM-UGRE v1.0/v1.1 confidential engine.** This directory intentionally excludes all confidential source, optimizer logic and policy parameters, iteration records, invention disclosures, private datasets, account data, credentials, and proprietary connector implementations. It is a minimal independent compatibility interface for reviewing CI/build integration without publishing trade secrets.

## Build and verify

    cmake -S experimental/dm_ugre_public_facade -B build/dm_ugre_facade -DCMAKE_BUILD_TYPE=Release
    cmake --build build/dm_ugre_facade --parallel
    ctest --test-dir build/dm_ugre_facade --output-on-failure
    ./build/dm_ugre_facade/dm_ugre_facade --demo

## Executable behavior

A single synthetic in-memory connector returns mock documents. The coordinator rejects unpermitted sources, ownership mismatches, missing evidence metadata and source mismatches **before publishing hits**. Search calls are budgeted and deterministic. Tests include a basic Morton-coordinate uniqueness/bounds check. The trace uses event codes, not private document bodies.

`--self-test` runs six local invariant assertions; CTest exercises the assertions and demo command.

## Scope and caveats

- Does **not** connect to live Gmail, private files, internet search or other third-party accounts.
- Does **not** implement neural intelligence, adaptive policies, source-code self-evolution or the confidential 1,200-step optimizer.
- The ACL strings and metadata flags are **synthetic fixtures**, not a production identity, cryptographic attestation, or end-to-end security mechanism.
- Morton indexing here is a generic spatial encoding and is not proof of patent novelty.
- No patent application, ownership right, filing date or legal priority is established by GitHub commits or this module.
- The confidential project archive should be held in **non-public controlled storage** and reviewed with patent counsel before disclosure.

This module is deliberately isolated from existing Jarvis-X components, does not modify prior engines and should remain a **draft PR** until reviewed.
