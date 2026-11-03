import {calculate, makeDemo} from './engine.mjs';
/**
 * Jarvis X Commerce OS: deterministic, finite-horizon, multi-step simulation.
 * It calls the SAME reference 4P engine as the interactive dashboard.
 * All warehouse orders and shipments are VIRTUAL, not authorised actions.
 * No HTTP, credentials, payments, advertising, or external integrations.
 */


const sum = (arr) => arr.reduce((a,b)=>a+b,0);
const money2 = n => Math.round((n+Number.EPSILON)*100)/100;
const safeWeek = v => Number.isInteger(v) && v>=1 && v<=52;
const safeSeed = v => Number.isInteger(v) && v>=0 && v<=0xffffffff;
const invariant = (ok,msg) => {if(!ok) throw new Error('SIM_INVARIANT: '+msg)};
const rngFrom = seed => {let x=(seed>>>0)||0x6d2b79f5;return ()=>{x^=x<<13;x^=x>>>17;x^=x<<5;return (x>>>0)/4294967296}};

/** @param {{weeks?:number,seed?:number,initial?:object}} options */
export function simulateOperations(options={}) {
  const weeks=options.weeks??12, seed=options.seed??20261103;
  if(!safeWeek(weeks)||!safeSeed(seed)) throw Error('weeks must be 1..52 and seed a uint32');
  const rng=rngFrom(seed),state=structuredClone(options.initial??makeDemo());
  // Validate input before touching state. The model can accept only aggregate SKU fields.
  calculate(state);
  let cash=state.cash, priorCash=state.cash, virtualOut=0;
  let terminalHalt=null;
  const receivables=[], pending=[], weeksData=[], events=[];
  let accumulatedContribution=0,accumulatedRevenue=0,accumulatedCOGS=0;
  const log=(week,type,payload)=>events.push({seq:events.length+1,week,type,...payload,simulated:true,externalExecution:false});
  for(let week=1;week<=weeks;week++) {
    let receivedQty=0, collectedPrior=0;
    // Step A: only due virtual deliveries become on-hand goods.
    for(let i=pending.length-1;i>=0;i--) if(pending[i].dueWeek<=week) {
      const shipment=pending.splice(i,1)[0],p=state.products.find(p=>p.sku===shipment.sku);
      invariant(!!p,'shipment SKU');
      p.incoming-=shipment.qty;p.onHand+=shipment.qty;receivedQty+=shipment.qty;
      log(week,'RECEIVE_VIRTUAL', {sku:shipment.sku,units:shipment.qty,sourceWeek:shipment.placedWeek});
    }
    // Step B: prior weekly credit settles before new procurement planning.
    for(let i=receivables.length-1;i>=0;i--) if(receivables[i].dueWeek<=week) {
      const receipt=receivables.splice(i,1)[0];collectedPrior=money2(collectedPrior+receipt.amount);
      log(week,'COLLECT_VIRTUAL_AR',{amountZAR:receipt.amount,invoiceWeek:receipt.invoiceWeek});
    }
    cash=money2(cash+collectedPrior);
    const cashAtPlan=cash;
    state.cash=cash;
    // Step C: forecast, profitability, constrained cash allocation and expansion gate.
    const plan=calculate(state);
    log(week,'FORECAST_AND_OPTIMISE',{
      expansionGate:plan.scaleGate,proposedProcurementZAR:plan.totals.procurement,
      forecastRevenueZAR:plan.totals.revenue,issueCount:plan.issues.length
    });
    const authorisedReal=!!plan.executionEnabled;
    invariant(!authorisedReal,'real-world actions are prohibited');
    // Step D: simulated procurement commits cash now, enters pending delivery queue.
    let procurementPaid=0,newOrdered=0;
    for(const q of plan.products) if(q.proposed>0) {
      const p=state.products.find(p=>p.sku===q.sku);
      const amount=money2(q.proposed*q.unitCost);
      pending.push({sku:q.sku,qty:q.proposed,dueWeek:week+Math.max(1,Math.ceil(q.lead/7)),placedWeek:week});
      p.incoming+=q.proposed;
      procurementPaid=money2(procurementPaid+amount);newOrdered+=q.proposed;
      log(week,'VIRTUAL_ORDER_PROPOSAL',{sku:q.sku,units:q.proposed,cashCommitmentZAR:amount,
        dueWeek:week+Math.max(1,Math.ceil(q.lead/7)),approvedExternally:false});
    }
    invariant(Math.abs(procurementPaid-plan.totals.procurement)<0.02,'procurement reconciliation');
    const requestedMarketing=plan.totals.marketing;
    const marketing=requestedMarketing<=state.marketingCap?requestedMarketing:0;
    const overhead=plan.totals.overhead;
    if(marketing!==requestedMarketing)log(week,'MARKETING_BUDGET_BLOCKED',{requestedZAR:requestedMarketing,actualZAR:0});
    const paid=money2(procurementPaid+marketing+overhead);
    // Step E: realise deterministic *synthetic* orders from a separate seeded demand process.
    let remainingDelivery=state.distributionCap;
    let revenue=0,cogs=0,unitsSold=0,unmet=0;
    const allocated=[];
    const demandPriority=[...plan.products].sort((a,b)=>{
      const score=p=>Math.max(0,p.sellingPrice-p.unitCost)*p.weeklyDemand/Math.max(1,p.available);
      return score(b)-score(a)||a.sku.localeCompare(b.sku);
    });
    for(const q of demandPriority) {
      const p=state.products.find(p=>p.sku===q.sku);
      const multiplier=0.86+0.28*rng();
      const syntheticDemand=Math.max(0,Math.round(q.weeklyDemand*multiplier));
      const physicallyAvailable=p.onHand-p.reserved;
      const sold=Math.max(0,Math.min(syntheticDemand,physicallyAvailable,remainingDelivery));
      const shortfall=syntheticDemand-sold;
      invariant(sold<=physicallyAvailable && sold<=remainingDelivery,'no negative physical inventory');
      p.onHand-=sold;remainingDelivery-=sold;
      const saleZAR=money2(sold*q.sellingPrice),soldCOGS=money2(sold*q.unitCost);
      revenue=money2(revenue+saleZAR);cogs=money2(cogs+soldCOGS);
      unitsSold+=sold;unmet+=shortfall;
      allocated.push({sku:q.sku,forecastUnits:Math.round(q.weeklyDemand),demandUnits:syntheticDemand,
        soldUnits:sold,shortfallUnits:shortfall,endingOnHand:p.onHand,saleZAR,cogsZAR:soldCOGS});
      // Add seven daily *observed* stock sales to the next training window.
      const originalLen=p.daily.length;
      const base=Math.floor(sold/7),extra=sold%7;
      const newObs=Array.from({length:7},(_,i)=>base+(i<extra?1:0));
      p.daily.push(...newObs);
      if(shortfall>0) for(let i=0;i<7;i++)p.stockoutDays.push(originalLen+i);
      if(p.daily.length>365){const trim=p.daily.length-365;p.daily=p.daily.slice(trim);
        p.stockoutDays=p.stockoutDays.filter(i=>i>=trim).map(i=>i-trim);
      }
      log(week,'SIMULATED_SALE',{sku:q.sku,units:sold,unmetUnits:shortfall,
        salesZAR:saleZAR,cogsZAR:soldCOGS});
    }
    // Step F: collect current invoices partially, maintain future dated receivables.
    const collectedNow=money2(revenue*state.collectionFraction);
    const newlyReceivable=money2(revenue-collectedNow);
    if(newlyReceivable>0)receivables.push({invoiceWeek:week,dueWeek:week+1,amount:newlyReceivable});
    const closingCash=money2(cashAtPlan-paid+collectedNow);
    const contribution=money2(revenue-cogs-marketing-overhead);
    const outstandingAR=money2(sum(receivables.map(x=>x.amount)));
    const inventoryValue=money2(sum(state.products.map(p=>p.onHand*p.unitCost)));
    const weekly={week,openingCashZAR:priorCash,priorCollectionsZAR:collectedPrior,
      availableCashZAR:cashAtPlan,forecastRevenueZAR:plan.totals.revenue,
      realisedRevenueZAR:revenue,realisedCOGSZAR:cogs,operatingContributionZAR:contribution,
      procurementPaidZAR:procurementPaid,marketingZAR:marketing,overheadZAR:overhead,
      collectedCurrentZAR:collectedNow,closingCashZAR:closingCash,
      outstandingReceivablesZAR:outstandingAR,inventoryBookCostZAR:inventoryValue,
      unitsSold,unitsOrdered:newOrdered,unitsReceived:receivedQty,unmetDemandUnits:unmet,
      expansionGate:plan.scaleGate,issues:plan.issues,products:allocated,externalActions:0};
    invariant(Math.abs(closingCash-(priorCash+collectedPrior-procurementPaid-marketing-overhead+collectedNow))<.03,'cash accounting');
    invariant(unitsSold<=state.distributionCap,'delivery throughput');
    invariant(state.products.every(p=>p.onHand>=p.reserved&&p.incoming>=0),'inventory accounting');
    invariant(Math.abs(revenue-(collectedNow+newlyReceivable))<.02,'credit sales accounting');
    events.push({seq:events.length+1,week,type:'WEEK_RECONCILED',
      cashZAR:closingCash,receivablesZAR:outstandingAR,inventoryZAR:inventoryValue,
      simulated:true,externalExecution:false});
    cash=closingCash;priorCash=closingCash;
    accumulatedContribution=money2(accumulatedContribution+contribution);
    accumulatedRevenue=money2(accumulatedRevenue+revenue);
    accumulatedCOGS=money2(accumulatedCOGS+cogs);
    weeksData.push(weekly);
    // A negative cash balance implies missing financing, not an overdraft licence.
    // Stop rather than injecting credit or passing an invalid cash balance to the next model step.
    if(closingCash<0){
      terminalHalt={week,code:'HALT_UNFUNDED_CASH_DEFICIT',unfundedCashZAR:money2(-closingCash)};
      log(week,'HALT_UNFUNDED_CASH_DEFICIT',{unfundedCashZAR:terminalHalt.unfundedCashZAR,
        requiresHumanIntervention:true});
      break;
    }
  }
  const endingState={cashZAR:cash,inventoryValueZAR:money2(sum(state.products.map(p=>p.onHand*p.unitCost))),
    receivablesZAR:money2(sum(receivables.map(p=>p.amount))),pendingShipments:structuredClone(pending),
    onHand:state.products.map(p=>({sku:p.sku,units:p.onHand,alreadyReserved:p.reserved,incoming:p.incoming}))};
  const summary={simulationWeeks:weeksData.length,requestedWeeks:weeks,seed,
    completedRequestedHorizon:terminalHalt===null,terminalHalt,startingCashZAR:options.initial?.cash??125000,
    endingCashZAR:cash,accrualRevenueZAR:accumulatedRevenue,cogsZAR:accumulatedCOGS,
    cumulativeOperatingContributionZAR:accumulatedContribution,
    unitsSold:sum(weeksData.map(w=>w.unitsSold)),unitsOrdered:sum(weeksData.map(w=>w.unitsOrdered)),
    unitsReceived:sum(weeksData.map(w=>w.unitsReceived)),
    unmetDemandUnits:sum(weeksData.map(w=>w.unmetDemandUnits)),
    expansionHoldWeeks:weeksData.filter(w=>w.expansionGate==='HOLD').length,
    events:events.length,externalActionsExecuted:virtualOut,
    endingState};
  invariant(virtualOut===0,'no real-world execution');
  return {schema:'jarvis-x-commerce-orchestration/v1',mode:'SYNTHETIC_SIMULATION_ONLY',
    generatedBy:'deterministic 7/14/28-day reference model, not trained ANN',
    externalExecutionEnabled:false,requiresHumanApproval:true,
    assumptions:['Weekly simulation; actual demand is an independent deterministic synthetic draw.',
      'Virtual procurement is paid at placement and received after lead-time weeks.',
      '75% (configurable) of invoices collected immediately; remainder next week.',
      'Inventory COGS is recognised in operating contribution, but not subtracted again from cash.',
      'A negative cash forecast ends the simulation immediately; no credit facilities are invented.',
      'No tax, wages beyond fixed overhead, loan amortisation, FX movements, supplier terms or cash financing.',
      'Forecasting uses stockout-censored sales; validation HOLD is not a live purchasing authorisation.'],
    summary, weeks:weeksData, events};
}
