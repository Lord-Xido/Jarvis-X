import test from "node:test";
import assert from "node:assert/strict";
import {computePlan,syntheticDemo,installStaticAPI} from "./static_engine.mjs";

test("synthetic example has three complete products",()=>{
 const d=syntheticDemo();
 assert.equal(d.products.length,3);
 assert.ok(d.products.every(p=>p.daily_sales.length===35));
});
test("deterministic replay and no execution",()=>{
 const d=syntheticDemo();
 const a=computePlan(d);
 assert.deepEqual(a,computePlan(structuredClone(d)));
 assert.equal(a.execution_enabled,false);
 assert.equal(a.requires_human_approval,true);
});
test("cash and receivable conservation",()=>{
 const d=syntheticDemo(),s=computePlan(d).summary;
 const close=(x,y)=>assert.ok(Math.abs(x-y)<=.02,"values differ");
 close(s.forecast_ending_cash_zar,d.cash_available_zar-s.proposed_procurement_cash_zar-s.promotion_spend_zar+s.cash_collected_zar);
 close(s.forecast_weekly_revenue_zar,s.cash_collected_zar+s.accounts_receivable_created_zar);
 close(s.forecast_gross_margin_zar,s.forecast_weekly_revenue_zar-s.forecast_weekly_cogs_zar);
});
test("sales cannot include incoming or newly ordered stock",()=>{
 const d=syntheticDemo();
 d.products[0].on_hand=0;d.products[0].reserved=0;d.products[0].incoming=100000;
 const p=computePlan(d).products.find(x=>x.sku===d.products[0].sku);
 assert.equal(p.projected_sales_units,0);
});
test("zero cash prohibits ordering",()=>{
 const d=syntheticDemo();d.cash_available_zar=0;
 assert.equal(computePlan(d).summary.planned_replenishment_units,0);
});
test("zero production capacity prohibits ordering",()=>{
 const d=syntheticDemo();d.production_capacity_units=0;
 assert.equal(computePlan(d).summary.planned_replenishment_units,0);
});
test("promotion has no assumed causal demand uplift",()=>{
 const d=syntheticDemo(),p=d.products[0];
 const before=computePlan(d).products.find(x=>x.sku===p.sku).weekly_forecast;
 p.promotion_zar=5000;
 const after=computePlan(d);
 assert.equal(after.products.find(x=>x.sku===p.sku).weekly_forecast,before);
 assert.equal(after.summary.promotion_spend_zar,5000);
});
test("stockout-censored observations trigger warning",()=>{
 const d=syntheticDemo();d.products[0].stockout_days=[3,7];
 const r=computePlan(d);
 assert.ok(r.warnings.some(x=>x.includes("stockout")));
 assert.equal(r.products.find(x=>x.sku===d.products[0].sku).forecast_reliable,false);
});
test("reject extraneous personally identifying fields",()=>{
 const d=syntheticDemo();d.products[0].email="person@example.com";
 assert.throws(()=>computePlan(d),/Unsupported product field/);
});
test("reject nonfinite numbers and malformed SKU",()=>{
 const d=syntheticDemo();d.products[0].retail_price_zar=NaN;
 assert.throws(()=>computePlan(d),/finite/);
 d.products[0].retail_price_zar=220;d.products[0].sku="../secret";
 assert.throws(()=>computePlan(d),/SKU/);
});
test("reject over-reservation and duplicate SKUs",()=>{
 const d=syntheticDemo();d.products[0].reserved=d.products[0].on_hand+1;
 assert.throws(()=>computePlan(d),/reserved/);
 d.products[0].reserved=0;d.products[1].sku=d.products[0].sku;
 assert.throws(()=>computePlan(d),/Duplicate SKU/);
});
test("in-browser API responds to planning without network access",async()=>{
 const old=globalThis.window;
 try {
  globalThis.window={
   location:{href:"https://example.test/Jarvis-X/commerce/"},
   fetch:async()=>{throw new Error("Unexpected network access")}
  };
  installStaticAPI();
  const data=await globalThis.window.fetch("/api/demo");
  assert.equal(data.status,200);
  const {input,plan}=await data.json();
  assert.equal(plan.execution_enabled,false);
  const posted=await globalThis.window.fetch("/api/plan",{
   method:"POST",body:JSON.stringify(input)
  });
  assert.equal(posted.status,200);
  const result=await posted.json();
  assert.deepEqual(result.summary,computePlan(input).summary);
  const rejected=await globalThis.window.fetch("/api/plan",{
   method:"POST",body:JSON.stringify({...input,customer_email:"invalid"})
  });
  assert.equal(rejected.status,400);
  const rejectedJson=await rejected.json();
  assert.equal(rejectedJson.error,"invalid_request");
 }finally{globalThis.window=old}
});
