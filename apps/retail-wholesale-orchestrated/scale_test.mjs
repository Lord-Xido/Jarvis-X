#!/usr/bin/env node
import fs from "node:fs";
import {makeDemo,TIERS} from "./engine.mjs";
import {simulateOperations} from "./orchestrator.mjs";
const rows=[];
for(let tier=0;tier<TIERS.length;tier++){
 const initial=makeDemo();initial.tier=tier;
 const r=simulateOperations({initial,weeks:12,seed:20261103});
 rows.push({market:TIERS[tier].name,assumedDemandMultiplier:TIERS[tier].factor,
  simulatedRevenueZAR:r.summary.accrualRevenueZAR,
  simulatedOperatingContributionZAR:r.summary.cumulativeOperatingContributionZAR,
  closingCashZAR:r.summary.endingCashZAR,
  executedWeeks:r.summary.simulationWeeks,
  unfundedCashHalt:r.summary.terminalHalt,
  expansionHoldWeeks:r.summary.expansionHoldWeeks,
  externalActions:r.summary.externalActionsExecuted});
}
console.log(JSON.stringify(rows,null,2));
if(rows.some(r=>r.externalActions!==0))process.exit(1);
