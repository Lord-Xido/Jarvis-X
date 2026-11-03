/**
 * Jarvis X Commerce - standalone, deterministic, browser-only simulation.
 * No network requests for plan/health/demo; no customer data persistence.
 */
const MAX_PRODUCTS = 12;
const round2 = x => Math.round((x + Number.EPSILON) * 100) / 100;
const avg = values => values.reduce((a, b) => a + b, 0) / values.length;
const validNum = (n, name, lo = 0, hi = 1e12) => {
  if (typeof n !== "number" || !Number.isFinite(n) || n < lo || n > hi) {
    throw new RangeError(name + " must be a finite number between " + lo + " and " + hi);
  }
  return n;
};
const validInt = (n, name, lo = 0, hi = 1e9) => {
  validNum(n, name, lo, hi);
  if (!Number.isInteger(n)) throw new RangeError(name + " must be a whole number");
  return n;
};
const parseProduct = p => {
  if (p === null || typeof p !== "object" || Array.isArray(p)) throw new TypeError("Each product must be an object");
  const allowed = new Set(["sku", "daily_sales", "on_hand", "reserved", "incoming", "unit_cost_zar",
    "retail_price_zar", "wholesale_price_zar", "wholesale_fraction", "lead_days", "review_days",
    "stockout_days", "promotion_zar", "place_reach"]);
  for (const key of Object.keys(p)) if (!allowed.has(key)) {
    throw new TypeError("Unsupported product field: " + key);
  }
  if (typeof p.sku !== "string" || !/^[A-Za-z0-9_-]{1,32}$/.test(p.sku))
    throw new TypeError("SKU must contain 1-32 letters, numbers, dashes or underscores");
  if (!Array.isArray(p.daily_sales) || p.daily_sales.length < 28 || p.daily_sales.length > 365)
    throw new TypeError("daily_sales must have 28-365 daily observations");
  const sales = p.daily_sales.map((n, i) => validInt(n, "daily_sales[" + i + "]", 0, 1e6));
  const onHand = validInt(p.on_hand ?? 0, "on_hand");
  const reserved = validInt(p.reserved ?? 0, "reserved", 0, onHand);
  const incoming = validInt(p.incoming ?? 0, "incoming");
  const cost = validNum(p.unit_cost_zar, "unit_cost_zar", .01, 1e8);
  const retail = validNum(p.retail_price_zar, "retail_price_zar", .01, 1e8);
  const wholesalePrice = validNum(p.wholesale_price_zar, "wholesale_price_zar", .01, 1e8);
  const share = validNum(p.wholesale_fraction ?? .30, "wholesale_fraction", 0, 1);
  const lead = validInt(p.lead_days ?? 7, "lead_days", 1, 90);
  const review = validInt(p.review_days ?? 7, "review_days", 1, 90);
  const reach = validNum(p.place_reach ?? 1, "place_reach", 0, 1);
  const promotion = validNum(p.promotion_zar ?? 0, "promotion_zar", 0, 1e8);
  const censored = p.stockout_days ?? [];
  if (!Array.isArray(censored) || censored.length > sales.length
      || censored.some(x => !Number.isInteger(x) || x < 0 || x >= sales.length))
    throw new TypeError("stockout_days must list valid zero-based observation indexes");
  const banned = new Set(censored);
  const errors = [], naiveErrors = [];
  for (let t = 21; t < sales.length; t++) {
    if (banned.has(t) || sales.slice(t-7, t).some((_, j) => banned.has(t - 7 + j))) continue;
    const prior = sales.slice(0, t);
    const prediction = .5*avg(prior.slice(-7)) + .3*avg(prior.slice(-14)) + .2*avg(prior.slice(-21));
    errors.push(Math.abs(prediction-sales[t]));
    naiveErrors.push(Math.abs(avg(prior.slice(-7))-sales[t]));
  }
  const last28 = sales.slice(-28), mu = avg(last28);
  const sigma = Math.sqrt(avg(last28.map(x => (x-mu)**2)));
  const daily = .5*avg(sales.slice(-7)) + .3*avg(sales.slice(-14)) + .2*mu;
  const safety = Math.ceil(1.645*sigma*Math.sqrt(lead+review));
  const target = Math.ceil(daily*(lead+review)*reach + safety);
  const position = onHand-reserved+incoming;
  const weightedPrice = retail*(1-share) + wholesalePrice*share;
  const weekDemand = daily*7*reach;
  return {
    sku:p.sku, daily_forecast:round2(daily*reach), weekly_forecast:round2(weekDemand),
    forecast_mae:errors.length?round2(avg(errors)):null,
    naive_mae:naiveErrors.length?round2(avg(naiveErrors)):null,
    stockout_censored:banned.size>0, forecast_reliable:banned.size===0 && errors.length>=7,
    safety_stock:safety, order_up_to:target, inventory_position:position,
    requested_replenishment:Math.max(0,target-position), on_hand:onHand, reserved,
    incoming, unit_cost_zar:cost, blended_sale_price_zar:round2(weightedPrice),
    promotion_zar:promotion, wholesale_fraction:share,
    available_unit_margin_zar:round2(weightedPrice-cost), place_reach:reach,
    projected_sales_units:Math.min(onHand-reserved,Math.floor(weekDemand))
  };
};
export function computePlan(payload) {
  if (payload === null || typeof payload !== "object" || Array.isArray(payload))
    throw new TypeError("Expected a planning object");
  for (const key of Object.keys(payload))
    if (!["products","cash_available_zar","production_capacity_units","collection_fraction"].includes(key))
      throw new TypeError("Unsupported planning field: "+key);
  if (!Array.isArray(payload.products) || payload.products.length < 1 || payload.products.length > MAX_PRODUCTS)
    throw new TypeError("Provide 1 to 12 products");
  const cash = validNum(payload.cash_available_zar ?? 50000, "cash_available_zar");
  const capacity = validInt(payload.production_capacity_units ?? 10000, "production_capacity_units");
  const collection = validNum(payload.collection_fraction ?? .8, "collection_fraction", 0, 1);
  const products = payload.products.map(parseProduct);
  if (new Set(products.map(x => x.sku)).size !== products.length)
    throw new TypeError("Duplicate SKU");
  products.sort((a,b) => {
    const priority = x => -(Math.max(0,x.available_unit_margin_zar)*x.weekly_forecast/Math.max(1,x.on_hand+x.incoming));
    return priority(a)-priority(b) || a.sku.localeCompare(b.sku);
  });
  let budget = cash, units = capacity;
  for (const p of products) {
    const affordable = Math.floor((budget+1e-8)/p.unit_cost_zar);
    const approved = Math.max(0,Math.min(p.requested_replenishment,units,affordable));
    p.proposed_replenishment = approved;
    p.procurement_cash_zar = round2(approved*p.unit_cost_zar);
    budget -= p.procurement_cash_zar;
    units -= approved;
    // Goods ordered today are NOT assumed to arrive before this week's sales.
    p.projected_sales_units = Math.min(Math.floor(p.weekly_forecast),p.on_hand-p.reserved);
    p.projected_revenue_zar = round2(p.projected_sales_units*p.blended_sale_price_zar);
    p.projected_cogs_zar = round2(p.projected_sales_units*p.unit_cost_zar);
    p.projected_gross_margin_zar = round2(p.projected_revenue_zar-p.projected_cogs_zar);
    p.proposed_action = "HUMAN_REVIEW_REQUIRED";
  }
  const sum = key => round2(products.reduce((s,p)=>s+p[key],0));
  const revenue = sum("projected_revenue_zar"), cogs = sum("projected_cogs_zar");
  const procurement = sum("procurement_cash_zar"), promo = sum("promotion_zar");
  const endingCash = round2(cash-procurement-promo+collection*revenue);
  const warnings = [
    "Standalone browser-based proposal only. No customer payments, orders or advertising are executed.",
    "Promotion is charged as spending, but no unverified demand uplift is assumed.",
    "No customer identifiers or personal data are required. All calculations are local to this browser."
  ];
  if (products.some(p=>p.stockout_censored))
    warnings.push("Some historical sales are stockout-censored; forecasts may underestimate demand.");
  if (products.some(p=>!p.forecast_reliable))
    warnings.push("One or more forecasts have incomplete calibration: review replenishment carefully.");
  if (endingCash < 0)
    warnings.push("Expected ending cash is negative; do not execute procurement proposal.");
  return {
    mode:"simulation",currency:"ZAR",time_horizon_days:7,
    execution_enabled:false,requires_human_approval:true,
    deployable_as_public_demo:true,products,
    summary:{
      forecast_weekly_revenue_zar:revenue,forecast_weekly_cogs_zar:cogs,
      forecast_gross_margin_zar:round2(revenue-cogs),
      proposed_procurement_cash_zar:procurement,promotion_spend_zar:promo,
      cash_collected_zar:round2(collection*revenue),
      forecast_ending_cash_zar:endingCash,
      accounts_receivable_created_zar:round2((1-collection)*revenue),
      planned_replenishment_units:products.reduce((n,p)=>n+p.proposed_replenishment,0),
      planned_sales_units:products.reduce((n,p)=>n+p.projected_sales_units,0),
      cash_budget_zar:cash
    },warnings
  };
}
export function syntheticDemo() {
  const configs = [
    ["WATER-FILTER",42,320,38,6,220,100,.30,.85,11],
    ["SOLAR-LAMP",18,180,12,4,430,260,.45,.72,17],
    ["CLEANING-KIT",64,570,54,5,145,92,.20,.92,23]
  ];
  const products=configs.map(([sku,base,stock,cost,lead,retail,wholesale,share,reach,seed])=>{
    const daily_sales=Array.from({length:35},(_,d)=>{
      const seasonal=1+.16*Math.sin(2*Math.PI*d/7);
      const drift=1+.002*d;
      const noise=(((d*19+seed*17)%11)-5)*.55;
      return Math.max(0,Math.round(base*seasonal*drift+noise));
    });
    return {sku,daily_sales,on_hand:stock,reserved:8,incoming:0,unit_cost_zar:cost,
      retail_price_zar:retail,wholesale_price_zar:wholesale,wholesale_fraction:share,
      lead_days:lead,review_days:7,stockout_days:[],promotion_zar:0,place_reach:reach};
  });
  return {products,cash_available_zar:55000,production_capacity_units:800,collection_fraction:.8};
}
export function installStaticAPI() {
  if (typeof window === "undefined" || typeof window.fetch !== "function") return;
  const original=window.fetch.bind(window);
  window.fetch=async (resource, options={})=>{
    const url=typeof resource==="string"?resource:resource.url;
    const method=(options.method||resource?.method||"GET").toUpperCase();
    const path=new URL(url,window.location.href).pathname;
    const json=(data,status=200)=>new Response(JSON.stringify(data),{
      status,headers:{"Content-Type":"application/json; charset=utf-8","Cache-Control":"no-store"}
    });
    if (method==="GET" && path==="/api/demo") {
      const input=syntheticDemo();
      return json({input,plan:computePlan(input)});
    }
    if (method==="GET" && path==="/healthz") return json({ok:true,mode:"local-browser",version:"1.0.0"});
    if (method==="POST" && path==="/api/plan") {
      try {
        const input=JSON.parse(options.body??"null");
        return json(computePlan(input));
      } catch (e) {
        return json({error:"invalid_request",message:String(e.message||e).slice(0,200)},400);
      }
    }
    return original(resource,options);
  };
}
