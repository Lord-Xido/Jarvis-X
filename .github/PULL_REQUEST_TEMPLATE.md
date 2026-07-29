## Purpose

<!-- What problem does this solve? Why does it belong in Jarvis-X? -->

## Classification

- [ ] Canonical implementation
- [ ] Integration candidate
- [ ] Bug fix
- [ ] Demonstration
- [ ] Specification or documentation
- [ ] Infrastructure or test repair

## What changed

<!-- List the narrow, concrete changes. -->

## Authoritative state and contracts

<!-- What state, API, format, equation or invariant changes? -->

## Implemented boundary

<!-- State what this PR implements and what it explicitly does not claim. -->

## Determinism and resource bounds

<!-- Identify seeds, clocks, environmental inputs, cycle limits and resident-memory bounds. -->

## Failure, rollback and persistence

<!-- What happens on invalid input, partial failure, rejected candidates or corrupt state? -->

## Validation

<!-- Paste the exact commands and summarize the results. -->

```text
python -m compileall -q src tests
pytest
```

## Performance evidence

<!-- Required only for performance claims: hardware, workload, repetitions, baseline, latency, throughput and memory. -->

## Security considerations

<!-- Untrusted inputs, native code, model loading, external scripts, secrets or data exposure. -->

## Documentation

- [ ] README or usage documentation updated
- [ ] Project status updated
- [ ] Changelog updated
- [ ] ADR added or updated where required
- [ ] No documentation change required

## Merge dependencies and successors

<!-- Name prerequisite, superseded or follow-up PRs/issues. -->

## Final checklist

- [ ] Scope is focused and reviewable
- [ ] Tests cover normal, invalid and boundary behavior
- [ ] CI is green
- [ ] Claims match measured or executable evidence
- [ ] No secrets or sensitive personal information are included
- [ ] Persistent formats and public APIs are documented
