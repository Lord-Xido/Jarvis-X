# Jarvis X Commerce OS — Closed-loop orchestrator

This application is a **fully runnable local simulation**, not a connected autonomous business.

Its self-contained `index.html` runs offline without hosting, dependencies, accounts or data exports. It includes an adjustable 3D warehouse, interactive four-P product controls, a 1–52-week end-to-end commercial operating simulation, 7/14/28-day forecast baseline, procurement constraints, supplier lead-time arrivals, synthetic sales, accounts receivable, cash receipts, expansion HOLD conditions and downloadable proposals.

The headless Node.js 20+ engine uses the **same deterministic calculation modules**:

```
node apps/retail-wholesale-orchestrated/run.mjs --weeks 12 --seed 20261103 --output /tmp/jx-ops
node apps/retail-wholesale-orchestrated/verify_audit.mjs /tmp/jx-ops
node --test apps/retail-wholesale-orchestrated/test_orchestrator.mjs
```

Outputs include orchestration.json, weekly_kpis.csv, hash-chained audit_ledger.ndjson and audit_manifest.json. A matching manifest detects changes to historical events but is not a digital signature or an encrypted log.

Business logic conserves physical stock, batches replenishment with lead times, caps distribution and procurement against capacity, and separates accrual margin from cash collection. No causal sales uplift is assumed for promotional spending. The baseline is not a trained ANN. Simulated virtual replenishment can continue during expansion HOLD for sensitivity testing; it never sends real purchase orders. Real payments, advertisements, customer communications, market deployments and financial trading are disabled.

Synthetic scenario outputs are **not** commercial traction, real customer ROI, or production security evidence. Live commercial integration requires permissioned data, tenant isolation, explicit scoped approval, cybersecurity/privacy review, and separate supervised production adapters.

The existing shared GitHub Pages deployment may be disabled at the repository level; this artifact remains independently usable offline.

## Geographic stress tests

Run `node apps/retail-wholesale-orchestrated/scale_test.mjs`. All five scale tiers are synthetic; an unfunded cash deficit now halts that scenario instead of assuming credit. Do not infer global market feasibility from a multipliers-only scenario.
