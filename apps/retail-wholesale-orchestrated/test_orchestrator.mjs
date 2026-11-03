import test from "node:test";
import assert from "node:assert/strict";
import {calculate,makeDemo} from "./engine.mjs";
import {simulateOperations} from "./orchestrator.mjs";
const approx=(a,b)=>assert.ok(Math.abs(a-b)<.06);
const run=(options={})=>simulateOperations({weeks:12,seed:20261103,...options});
test("deterministic 12-week replay",()=>{
 const r=run();assert.deepEqual(r,run());assert.equal(r.weeks.length,12);
});
test("live execution is disabled in every event",()=>{
 const r=run();assert.equal(r.externalExecutionEnabled,false);
 assert.equal(r.summary.externalActionsExecuted,0);
 assert.ok(r.events.every(e=>e.simulated===true&&e.externalExecution===false));
});
test("weekly cash flow and roll forward reconcile",()=>{
 const r=run();for(let i=0;i<r.weeks.length;i++){
  const w=r.weeks[i];
  approx(w.closingCashZAR,w.openingCashZAR+w.priorCollectionsZAR+w.collectedCurrentZAR-w.procurementPaidZAR-w.marketingZAR-w.overheadZAR);
  if(i>0)approx(w.openingCashZAR,r.weeks[i-1].closingCashZAR);
 }
});
test("invoiced sales reconcile to collections and receivables",()=>{
 const r=run();
 for(let i=0;i<r.weeks.length;i++){
  const w=r.weeks[i];
  approx(w.outstandingReceivablesZAR,w.realisedRevenueZAR-w.collectedCurrentZAR);
  if(i>0)approx(w.priorCollectionsZAR,r.weeks[i-1].outstandingReceivablesZAR);
 }
});
test("inventory mass balance per SKU",()=>{
 const initial=makeDemo(),r=run({initial});
 for(const p of initial.products){
  const receipts=r.events.filter(e=>e.type==="RECEIVE_VIRTUAL"&&e.sku===p.sku).reduce((s,e)=>s+e.units,0);
  const shipments=r.events.filter(e=>e.type==="SIMULATED_SALE"&&e.sku===p.sku).reduce((s,e)=>s+e.units,0);
  assert.equal(r.summary.endingState.onHand.find(q=>q.sku===p.sku).units,p.onHand+receipts-shipments);
 }
});
test("lead time and distribution bounds",()=>{
 const d=makeDemo(),r=run({initial:d});
 assert.ok(r.weeks.every(w=>w.unitsSold<=d.distributionCap));
 assert.ok(r.events.filter(e=>e.type==="RECEIVE_VIRTUAL").every(e=>e.week>e.sourceWeek));
});
test("cash budget and production zero prevent all orders",()=>{
 const d=makeDemo();d.productionCap=0;
 const r=run({weeks:3,initial:d});assert.equal(r.summary.unitsOrdered,0);
});
test("out-of-budget promotion blocked",()=>{
 const d=makeDemo();d.marketingCap=10;
 const r=run({weeks:1,initial:d});
 assert.equal(r.weeks[0].marketingZAR,0);
 assert.ok(r.events.some(e=>e.type==="MARKETING_BUDGET_BLOCKED"));
});
test("expansion hold does not authorise external procurement",()=>{
 const r=run();assert.ok(r.weeks.some(w=>w.expansionGate==="HOLD"));
 assert.equal(r.requiresHumanApproval,true);
});
test("reject invalid seed, weeks, input data",()=>{
 assert.throws(()=>run({weeks:53}),/weeks/);
 assert.throws(()=>run({seed:-3}),/seed/);
 const d=makeDemo();d.products[0].reserved=d.products[0].onHand+1;
 assert.throws(()=>run({initial:d}),/reserved/);
});
test("no mutation of source and baseline remains reproducible",()=>{
 const d=makeDemo(),snap=structuredClone(d);
 run({initial:d,weeks:2});
 assert.deepEqual(d,snap);
 assert.deepEqual(calculate(makeDemo()),calculate(makeDemo()));
});

test("all geographic tiers halt cleanly rather than invent overdraft financing",()=>{
 for(let tier=0;tier<5;tier++){
  const d=makeDemo();d.tier=tier;
  const r=run({initial:d});
  assert.ok(r.summary.simulationWeeks>=1&&r.summary.simulationWeeks<=12);
  assert.equal(r.summary.externalActionsExecuted,0);
  if(r.summary.terminalHalt){
   assert.equal(r.summary.terminalHalt.code,"HALT_UNFUNDED_CASH_DEFICIT");
   assert.ok(r.summary.terminalHalt.unfundedCashZAR>0);
   assert.equal(r.summary.completedRequestedHorizon,false);
  }else assert.equal(r.summary.completedRequestedHorizon,true);
 }
});
