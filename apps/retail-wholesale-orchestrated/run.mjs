#!/usr/bin/env node
import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";
import {simulateOperations} from "./orchestrator.mjs";
const argv=process.argv.slice(2);
function option(name, fallback) {const i=argv.indexOf(name);return i<0?fallback:argv[i+1]}
const weeks=Number(option("--weeks","12"));
const seed=Number(option("--seed","20261103"));
const out=path.resolve(option("--output","./outputs"));
const result=simulateOperations({weeks,seed});
fs.mkdirSync(out,{recursive:true});
fs.writeFileSync(path.join(out,"orchestration.json"),JSON.stringify(result,null,2)+"\n");
const cols=["week","expansionGate","realisedRevenueZAR","realisedCOGSZAR","operatingContributionZAR","procurementPaidZAR","closingCashZAR","outstandingReceivablesZAR","inventoryBookCostZAR","unitsSold","unitsOrdered","unitsReceived","unmetDemandUnits"];
fs.writeFileSync(path.join(out,"weekly_kpis.csv"),cols.join(",")+"\n"+result.weeks.map(w=>cols.map(c=>w[c]).join(",")).join("\n")+"\n");
let prev="0".repeat(64);
const lines=result.events.map(e=>{
 const raw=JSON.stringify(e);
 const sha=crypto.createHash("sha256").update(prev+"\n"+raw).digest("hex");
 const row=JSON.stringify({...e,previousSHA256:prev,eventSHA256:sha});
 prev=sha;
 return row
});
fs.writeFileSync(path.join(out,"audit_ledger.ndjson"),lines.join("\n")+"\n");
const manifest={schema:"jarvis-x-commerce-audit-manifest/v1",events:lines.length,terminalHash:prev,externalExecutionEnabled:false,mode:result.mode};
fs.writeFileSync(path.join(out,"audit_manifest.json"),JSON.stringify(manifest,null,2)+"\n");
console.log(JSON.stringify({mode:result.mode,summary:result.summary,manifest},null,2));
