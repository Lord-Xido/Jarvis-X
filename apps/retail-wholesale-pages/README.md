# Jarvis X Retail & Wholesale Commerce — Browser-only live fallback

This directory contains the standalone, client-side build of the retail and wholesale demo for GitHub Pages. It is an alternative to the Replit deployment whose production API routes returned HTTP 404.

The original server API and tests remain in apps/retail-wholesale-mvp. This fallback uses the same four-P controls and a locally executed planning module to avoid a network API dependency.

## Features and boundaries
- Responsive demand and inventory dashboard with synthetic SKU data.
- 7-day baseline forecast, wholesale/retail pricing mix, replenishment proposals, physical stock limits and financial reconciliation.
- Local-only CSV import and JSON export.
- No accounts, remote database, payments, orders, advertising, tracking, personal customer data or remote business actions.
- No trained ANN in this fallback. The weighted baseline is transparent and deterministic.

The in-browser response layer emulates /api/demo and /api/plan calls for the dashboard only. These are not real public HTTP API endpoints; calculations run entirely inside the browser.

Run Node 20+ tests using: node --test apps/retail-wholesale-pages/test_static_engine.mjs

Planning proposals only. Forecasts are not calibrated for real-world replenishment decisions. New procurement cannot be counted as available before receipt.
