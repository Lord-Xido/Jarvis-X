#!/usr/bin/env node
import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";
const out=path.resolve(process.argv[2]||"./outputs");
const lines=fs.readFileSync(path.join(out,"audit_ledger.ndjson"),"utf8").trim().split("\n");
const manifest=JSON.parse(fs.readFileSync(path.join(out,"audit_manifest.json"),"utf8"));
let previous="0".repeat(64);
for(let i=0;i<lines.length;i++){
 const row=JSON.parse(lines[i]);
 const {previousSHA256,eventSHA256,...event}=row;
 if(event.seq!==i+1||previousSHA256!==previous)throw Error("audit sequence break at "+(i+1));
 const sha=crypto.createHash("sha256").update(previous+"\n"+JSON.stringify(event)).digest("hex");
 if(sha!==eventSHA256)throw Error("audit digest mismatch at "+(i+1));
 previous=sha;
}
if(manifest.terminalHash!==previous||manifest.events!==lines.length||manifest.externalExecutionEnabled!==false)throw Error("audit manifest mismatch");
console.log("AUDIT PASS",lines.length,previous);
