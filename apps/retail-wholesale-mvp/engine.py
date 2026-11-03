"""Deterministic, proposal-only inventory and cash-planning engine (stdlib).
Inputs are aggregate product metrics. It is intentionally not a causal ANN
and never performs external commerce, advertising or purchasing actions.
"""
import math
from statistics import mean, pstdev

MAX_ITEMS = 12
MAX_HISTORY = 365

class InputError(ValueError):
    pass

def number(value, label, minimum=0, maximum=1e12):
    if type(value) not in (float, int) or not math.isfinite(value) or not minimum <= value <= maximum:
        raise InputError(f"{label} must be a finite number between {minimum} and {maximum}")
    return float(value)

def whole(value, label, minimum=0, maximum=1000000000):
    x = number(value, label, minimum, maximum)
    if x != int(x):
        raise InputError(f"{label} must be a whole number")
    return int(x)

def compute(payload):
    if not isinstance(payload, dict) or set(payload)-{"products", "cash_available_zar", "production_capacity_units", "collection_fraction"}:
        raise InputError("Expected products and permitted aggregate planning fields only")
    products = payload.get("products")
    if not isinstance(products, list) or not 1 <= len(products) <= MAX_ITEMS:
        raise InputError(f"Provide between 1 and {MAX_ITEMS} products")
    cash = number(payload.get("cash_available_zar", 50000), "cash_available_zar")
    capacity = whole(payload.get("production_capacity_units", 10000), "production_capacity_units")
    collection = number(payload.get("collection_fraction", 0.8), "collection_fraction", 0, 1)
    rows = []
    seen = set()
    for idx, p in enumerate(products):
        if not isinstance(p, dict):
            raise InputError("Each product must be an object")
        allowed = {"sku","daily_sales","on_hand","reserved","incoming","unit_cost_zar","retail_price_zar","wholesale_price_zar",
                   "wholesale_fraction","lead_days","review_days","stockout_days","promotion_zar","place_reach"}
        if set(p)-allowed:
            raise InputError("Unknown product fields; personal data and identifiers other than SKU are not accepted")
        sku = p.get("sku")
        if not isinstance(sku, str) or not (1 <= len(sku) <= 32) or not all(c.isascii() and (c.isalnum() or c in "-_") for c in sku):
            raise InputError("sku must contain only ASCII letters, numbers, - and _ (1-32 characters)")
        if sku in seen:
            raise InputError("Duplicate SKU")
        seen.add(sku)
        history = p.get("daily_sales")
        if not isinstance(history, list) or not 28 <= len(history) <= MAX_HISTORY:
            raise InputError("daily_sales requires 28-365 observations per SKU")
        sales = [whole(x, "daily_sales item", 0, 1000000) for x in history]
        onhand = whole(p.get("on_hand", 0), "on_hand")
        reserved = whole(p.get("reserved", 0), "reserved", 0, onhand)
        incoming = whole(p.get("incoming", 0), "incoming")
        cost = number(p.get("unit_cost_zar"), "unit_cost_zar", 0.01, 1e8)
        rp = number(p.get("retail_price_zar"), "retail_price_zar", 0.01, 1e8)
        wp = number(p.get("wholesale_price_zar"), "wholesale_price_zar", 0.01, 1e8)
        wholesale = number(p.get("wholesale_fraction", 0.3), "wholesale_fraction", 0, 1)
        lead = whole(p.get("lead_days", 7), "lead_days", 1, 90)
        review = whole(p.get("review_days", 7), "review_days", 1, 90)
        reach = number(p.get("place_reach", 1), "place_reach", 0, 1)
        promo = number(p.get("promotion_zar", 0), "promotion_zar", 0, 100000000)
        stockout_days = p.get("stockout_days", [])
        if not isinstance(stockout_days, list) or len(stockout_days)>len(sales) or any(type(j)!=int or j<0 or j>=len(sales) for j in stockout_days):
            raise InputError("stockout_days must be valid zero-based history indexes")
        # Chronological holdout: last seven observations are not used to fit this error estimate.
        rolling_errors = []
        baseline_errors = []
        for t in range(21, len(sales)):
            if t in stockout_days or any(k in stockout_days for k in range(t-7,t)):
                continue
            prior = sales[:t]
            prediction = .5*mean(prior[-7:])+.3*mean(prior[-14:])+.2*mean(prior[-21:])
            baseline = mean(prior[-7:])
            rolling_errors.append(abs(prediction-sales[t]))
            baseline_errors.append(abs(baseline-sales[t]))
        # If sales are censored, avoid interpreting observed units as unconstrained demand.
        reliable = not stockout_days and len(rolling_errors)>=7
        daily = .5*mean(sales[-7:])+.3*mean(sales[-14:])+.2*mean(sales[-28:])
        observed_sigma = pstdev(sales[-28:])
        demand_7 = daily*7*reach
        # Normal approximation, fixed lead time and independent daily errors: planning heuristic, not calibrated service level.
        safety = math.ceil(1.645*observed_sigma*math.sqrt(lead+review))
        target = math.ceil(daily*(lead+review)*reach + safety)
        inventory_position = onhand-reserved+incoming
        requested = max(0, target-inventory_position)
        blended_price = rp*(1-wholesale)+wp*wholesale
        row = dict(
          sku=sku, daily_forecast=round(daily*reach,2), weekly_forecast=round(demand_7,2),
          forecast_mae=round(mean(rolling_errors),2) if rolling_errors else None,
          naive_mae=round(mean(baseline_errors),2) if baseline_errors else None,
          stockout_censored=bool(stockout_days), forecast_reliable=reliable,
          safety_stock=safety, order_up_to=target, inventory_position=inventory_position,
          requested_replenishment=requested, on_hand=onhand, reserved=reserved, incoming=incoming,
          projected_sales_units=min(onhand-reserved, math.floor(demand_7)),
          unit_cost_zar=cost, blended_sale_price_zar=round(blended_price,2),
          promotion_zar=promo, wholesale_fraction=wholesale,
          available_unit_margin_zar=round(blended_price-cost,2),
          place_reach=reach,
        )
        rows.append(row)
    # Conservatively order from higher margin-weighted, demand-weighted shortfalls first.
    rows.sort(key=lambda p: (-max(0,p["available_unit_margin_zar"])*p["weekly_forecast"]/max(1,p["on_hand"]+p["incoming"]), p["sku"]))
    budget_remaining = cash
    units_remaining = capacity
    for p in rows:
        affordable = math.floor((budget_remaining+1e-8)/p["unit_cost_zar"])
        approved = min(p["requested_replenishment"],units_remaining,affordable)
        p["proposed_replenishment"] = max(0,approved)
        p["procurement_cash_zar"] = round(approved*p["unit_cost_zar"],2)
        budget_remaining -= p["procurement_cash_zar"]
        units_remaining -= approved
        # Sales forecast is a conservative 7-day quantity with physical inventory + proposed replenishment.
        p["projected_sales_units"] = min(
            math.floor(p["weekly_forecast"]), p["on_hand"]-p["reserved"])
        p["projected_revenue_zar"]=round(p["projected_sales_units"]*p["blended_sale_price_zar"],2)
        p["projected_cogs_zar"]=round(p["projected_sales_units"]*p["unit_cost_zar"],2)
        p["projected_gross_margin_zar"]=round(
            p["projected_revenue_zar"]-p["projected_cogs_zar"],2)
        p["proposed_action"]="HUMAN_REVIEW_REQUIRED"
    total = lambda key: round(sum(p[key] for p in rows),2)
    revenue=total("projected_revenue_zar")
    cost=total("projected_cogs_zar")
    procurement=total("procurement_cash_zar")
    promo=total("promotion_zar")
    cash_end=round(cash - procurement - promo + collection*revenue,2)
    warnings = ["Scenario planning only: no live marketing, pricing, purchase or payment execution.",
       "Promotion does not increase forecast sales without experimentally validated uplift.",
       "No personal customer records are retained or required."]
    if any(p["stockout_censored"] for p in rows):
        warnings.append("Historical sales include stockout-censored periods: forecasts may underestimate demand.")
    if any(not p["forecast_reliable"] for p in rows):
        warnings.append("Forecast calibration is incomplete: review suggested replenishment manually.")
    if cash_end<0:
        warnings.append("Expected ending cash is negative; do not execute proposal.")
    return dict(
      mode="simulation", currency="ZAR", time_horizon_days=7, execution_enabled=False,
      requires_human_approval=True, deployable_as_public_demo=True,
      products=rows, summary={
          "forecast_weekly_revenue_zar":revenue, "forecast_weekly_cogs_zar":cost,
          "forecast_gross_margin_zar":round(revenue-cost,2),
          "proposed_procurement_cash_zar":procurement, "promotion_spend_zar":promo,
          "cash_collected_zar":round(collection*revenue,2),
          "forecast_ending_cash_zar":cash_end,
          "accounts_receivable_created_zar":round((1-collection)*revenue,2),
          "planned_replenishment_units":sum(p["proposed_replenishment"] for p in rows),
          "planned_sales_units":sum(p["projected_sales_units"] for p in rows),
          "cash_budget_zar":cash,
       }, warnings=warnings)
