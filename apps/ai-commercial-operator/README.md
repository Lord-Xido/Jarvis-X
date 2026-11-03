# Jarvis X Auto Operator — Local-to-global AI commercial controller

A **runnable, Python standard-library-only** 4P planning and governance prototype. It forecasts **new paying business customers**, proposes subscription prices and promotional budgets by market, checks weekly inference capacity and working capital, and generates an auditable operating plan. It does **not** execute real market actions.

## Quick start

```bash
python3 jarvis_x_auto_operator.py --demo
python3 test_operator.py
```

Outputs (in `outputs/`):
- `synthetic_observations.csv` — generated labelled data for 5 market tiers over 52 weeks;
- `dashboard.html` — offline view of recommended 4P market allocation and financial results;
- `latest_plan.json` — machine-readable demand, commercial and cash-flow proposals;
- `market_plan.csv` — market-by-market proposed pricing and marketing budgets;
- `pending_approvals.json` — all external actions blocked unless separately authorised;
- `audit.jsonl` — local append-only *file* audit events (not cryptographically immutable).

## Connect your own observations

```bash
python3 jarvis_x_auto_operator.py --data observations.csv --output outputs/live_analysis
```

Required CSV fields:

`week,market_id,price_zar,promotion_zar,place_reach,product_quality_index,affordability_index,sentiment,previous_new_paid,new_paid,uncensored`

Use a consistent weekly time unit and market identifiers matching `config.json`. `new_paid` must mean verified new subscriptions sold, and `uncensored=true` only where unmet demand isn't obscured by acquisition, compute or payment constraints. Market-specific base price, size, pricing assumptions, company cash and compute capacity are in `config.json`. CSV imports do **not** prove truth or causal validity of labels.

Optional watch mode polls local files for a change and runs **only when the input content changes**:

```bash
python3 jarvis_x_auto_operator.py --data observations.csv --watch-seconds 3600
```

This requires **you to keep the process running**. It does not create a cloud service, scheduled task or background job in ChatGPT.

## What is genuinely operational?

1. Deterministic synthetic market-data generator with constant seed.
2. Real 10→14→1 neural network trained with gradient descent/Adam; chronological training/validation/test splits.
3. Out-of-sample MAE compared with a last-week naïve forecast; interval-coverage diagnostic.
4. Finite 4P pricing and marketing menu optimised across 5 markets using exact enumeration of combinations.
5. Per-market new-paid, active-paid, inference-request, subscription-revenue, marketing and variable-cost calculations.
6. Portfolio budget, compute and minimum-cash-reserve hard constraints.
7. **Safety gates**: negative operating contribution or poor predictive performance blocks scale; all real executions are disabled.
8. Versionable JSON/CSV audit and HTML dashboard plus runnable unit tests.

## What is NOT yet operational?

- Actual customer/CRM, POS, payment or advertising integrations.
- Real cloud provisioning, security certification, geographic data residency controls.
- Causally established price and promotion responses.
- Accurate GPU concurrency, token-load, latency and energy benchmarks.
- Automated financial commitments, marketing launch, client messaging or legal approval.

**The existing `jarvis_x_demand_ann_industrial.html` is a separate industrial-forecasting demonstrator**, not automatically connected to this service. Production rollout requires real permissioned records, data contracts, held-out calibration, incremental campaign experiments, service instrumentation and explicit authorisation.

## Core equations

For market \(j\), estimated weekly new paying clients \(\widehat n_j=f_\theta(\text{product},\text{place},\text{price},\text{promotion},\ldots)\).

\[
A_{j,t+1}=A_{j,t}(1-\chi_j)+\widehat n_j,
\qquad W_j=100A_{j,t+1},
\]

\[
R_j=A_{j,t}(1-\chi_j)P_j^0+\widehat n_jP_j,
\]

\[
\max_{P_j,M_j}\sum_j (R_j-C_{\text{compute},j}-C_{\text{support},j}-M_j)-F,
\]

subject to \(\sum_j W_j\le K_{\text{compute}}\), \(\sum_jM_j\le B_{\text{marketing}}\), and closing cash \(\ge C_{\text{reserve}}\).

Subscription pricing for existing subscribers is held at the reference price; proposed new prices apply only to **new customers**. This prototype does not model multi-period customer lifetime value, cohort churn, sales tax or capital investment payback.