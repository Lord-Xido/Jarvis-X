# Jarvis X Retail & Wholesale Intelligence — public demonstration MVP

**Status:** executable, stateless **decision-support demonstration**; not connected to real customer systems or financial execution.

## Features
- Responsive browser dashboard with simulated SKU history and a visual forecast.
- Strict aggregate-only planning request schema, 28–365 daily observations per SKU and up to 12 SKUs.
- Price, place/reach, wholesale channel mix, inventory and promotional-spend scenario controls.
- Deterministic demand estimates with a chronological rolling test and naïve baseline.
- Safety-stock and order-up-to proposals, subject to procurement cash and production capacity.
- Separate wholesale/retail blended sales revenue, COGS, accounts receivable and cash collection.
- Browser CSV import of aggregates only; plan exported locally as JSON.
- Proposals only; no orders, payment processing, email, live advertisements or automation.

## Run

Requires Python 3.12+. No packages to install.

    PORT=8080 python server.py

Open http://localhost:8080 and verify http://localhost:8080/healthz.

    python -m unittest -v test_engine

Optional Docker:

    docker build -t jarvis-x-commerce .
    docker run --rm -p 8080:8080 jarvis-x-commerce

## Input contract

POST /api/plan with Content-Type: application/json. Body contains:
- products: array of 1–12 products
- cash_available_zar: nonnegative ZAR amount
- production_capacity_units: integer, maximum available procurement/production capacity
- collection_fraction: portion of forecast sales receipts collected during the same period (0–1)

Each product requires: sku, daily_sales (28–365 integer daily observations), unit_cost_zar, retail_price_zar and wholesale_price_zar. Optional: on_hand, reserved, incoming, wholesale_fraction, lead_days, review_days, promotion_zar, place_reach, stockout_days. Unknown keys, including personal customer identifiers, are rejected.

## Important modelling limitations

- A weighted 7/14/28-day average is the reference model; **it is not an ANN**. The previously merged CLI ANN is a separate research component. A production system should perform a validated forecast-model selection rather than claim the demo is a trained model.
- Demand estimates depend on observed sales, which can be censored by stockouts. Stockout flags cause explicit warnings.
- The service is stateless and does not retain request payloads; the browser still sends product-level aggregates to the server when recalculating. Use **synthetic data only on public hosting**. Do not upload private financial or personal customer records.
- Promotion spending is treated as expenditure. No causal incremental demand is assumed.
- Orders proposed today are **not** counted as saleable stock during the 7-day forecast; fulfillment uses available on-hand units.
- The safety-stock formula assumes fixed lead time and roughly independent demand fluctuations. Its nominal factor is a heuristic, not a validated service-level guarantee.
- This is a single-period scenario: procurement cost is an immediate cash outflow, revenue may be partly receivable; it does not model tax, debt, supplier terms, refunds or margin over multiple periods.
- External integrations, tenant identity, authentication, privacy certification, observability, deployment hardening and regulatory reviews are prerequisites for real customer operation.

## Deployment
Deploy this folder as a Docker service or execute `python server.py`, binding `0.0.0.0:$PORT`. The GET /healthz route is provided for host health probes. Do not enable live customer data, remote writes or autonomous execution without an independently reviewed secure backend and appropriate authorisation.
