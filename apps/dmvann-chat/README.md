# DMVANN Chat — DM-vΩΞ⁺ Neural Interface

This directory registers the externally deployed DMVANN Chat interface with the Jarvis-X application tree.

## Live deployment

- URL: https://dmvannchat-utfg9t9j.manus.space/
- Observed page title: `DM–vΩΞ⁺ Neural Interface`
- Registered: 2026-08-25
- Deployment host: Manus

## Relationship to Jarvis-X

Jarvis-X already contains `apps/dr-moagi-cognitive`, the production-oriented DM-vΩΞ⁺ browser control plane. DMVANN Chat is tracked separately here until its deployable source tree, build configuration, and runtime contract are imported and verified.

This registration intentionally does **not** claim that the current Manus deployment is byte-for-byte identical to `apps/dr-moagi-cognitive`.

## Integration boundary

Before treating this deployment as a first-class Jarvis-X runtime, verify and import:

1. application source and dependency lockfile;
2. build/start commands and environment variables;
3. client/server API contract;
4. model/provider configuration with secrets kept server-side;
5. deterministic health endpoint and readiness semantics;
6. CSP, CORS, authentication, rate limiting, and input-boundary controls;
7. test suite covering chat state, failure recovery, and any DM-vΩΞ⁺ numerical transforms;
8. CI build/test/deploy workflow;
9. provenance/version metadata tying a deployment to a Git commit;
10. deployment rollback procedure.

## Verification status

The public endpoint is reachable through the browsing layer and identifies itself as `DM–vΩΞ⁺ Neural Interface`. The dynamic deployment source was not recoverable from the public page during this registration, so no source equivalence or backend-capability claim is made here.

## Target repository layout

When source is available, converge on:

```text
apps/dmvann-chat/
├── README.md
├── package.json
├── package-lock.json
├── src/
├── public/
├── tests/
└── deployment/
```

The acceptance criterion is **working → robust → portable → elegant → advanced**: first reproduce the deployed behavior locally, then harden, test, containerize, and optimize it.
