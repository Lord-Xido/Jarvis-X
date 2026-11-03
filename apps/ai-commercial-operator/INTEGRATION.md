# Integration scope — AI Commercial Auto Operator

This directory is an **isolated demonstration**, not a production control plane, and is not part of the existing VM/runtime deployment paths.

## What is included

- Deterministic 5-market synthetic data generator, 10→14→1 MLP with chronological validation and holdout, last-week baseline and interval-coverage diagnostic.
- Enumeration-based four-P subscription pricing and marketing-plan optimisation.
- Request-capacity, cash-reserve, marketing-budget and operating-contribution checks.
- A proposal-only JSON/CSV audit report and offline HTML dashboard, generated when the program runs.
- Targeted standard-library CI tests.

## Operational boundary

This feature **does not** provision computing resources, change prices, send customer messages, initiate marketing campaigns, place securities trades, handle payments, or access customer data. Every reported market size and business assumption in the demonstration is synthetic. The ANN response curve has not been causally established.

Source files and configuration are committed; generated output and private customer data are not. Use a separate output directory for each execution in production environments.

## Review / rollout conditions

1. Replace synthetic data with permissioned records governed by an agreed data contract, minimum group sizes, and appropriate privacy controls.
2. Confirm forecast calibration and compare against multiple robust chronological baselines.
3. Verify marketing effects with controlled tests; ANN correlation alone cannot justify capital expenditure.
4. Measure actual GPU/service throughput, latency, cost and cash collection.
5. Keep all real-world actions under separately authorised, scoped integrations with audit, replay and rollback.
6. Complete product, cybersecurity, data protection, financial and legal reviews before any market expansion.

Do not infer global readiness or production ROI from demo backtests.
