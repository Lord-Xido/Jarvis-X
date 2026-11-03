/* Jarvis X Commerce v1.0 — deterministic, local-only reference model.
   All forecasts and unit economics are proposals, not trade or order execution. */
const finite = (x,name,min=0,max=1e12) => {
  if(typeof x!== 'number' || !Number.isFinite(x) || x<min || x>max) throw Error(`${name} must be finite, ${min}–${max}`);
  return x;
};
const integer=(x,name,min=0,max=1e9)=>{ finite(x,name,min,max); if(!Number.isInteger(x))throw Error(`${name} must be an integer`);return x;};
const mean=a=>a.reduce((s,x)=>s+x,0)/a.length;
const round=n=>Math.round((n+Number.EPSILON)*100)/100;
export const TIERS = Object.freeze([
 {name:'Local pilot',factor:1,overhead:1800},
 {name:'South Africa',factor:3,overhead:8000},
 {name:'Southern Africa',factor:7,overhead:26000},
 {name:'Pan-African',factor:15,overhead:85000},
 {name:'Global',factor:30,overhead:250000}
]);
export function makeDemo(){
 const catalog=[
  ['WATER-FILTER','Water filtration cartridge',43,420,28,80,185,150,.32,.87,7,13],
  ['SOLAR-LAMP','Solar lantern',20,240,12,115,275,225,.42,.72,12,31],
  ['CLEANING-KIT','Household cleaning kit',63,740,45,52,145,119,.26,.93,4,17],
  ['FIRST-AID','First aid replenishment kit',14,160,20,130,295,245,.38,.77,10,23]
 ];
 return {tier:0,cash:125000,minReserve:12000,marketingCap:18000,distributionCap:2000,productionCap:1200,collectionFraction:.75,
   products:catalog.map(([sku,name,base,onHand,reserved,cost,retail,wholesale,mix,reach,lead,seed])=>({
     sku,name,daily:Array.from({length:56},(_,i)=>Math.max(0,Math.round(base*(1+.14*Math.sin(2*Math.PI*i/7))*(1+.0015*i)+(((i*17+seed*19)%13)-6)*.38))),
     onHand,reserved,incoming:0,unitCost:cost,retail,wholesale,mix,reach,promo:sku==='WATER-FILTER'?1500:500,
     lead,review:7,stockoutDays:[]
   }))};
}
function validate(src){
 if(!src || typeof src!=='object' || !Array.isArray(src.products) ||src.products.length<1||src.products.length>12)throw Error('Expected 1–12 products');
 integer(src.tier,'tier',0,4);
 finite(src.cash,'cash',0,1e9);finite(src.minReserve,'minReserve',0,1e9);
 finite(src.marketingCap,'marketingCap',0,1e9);
 integer(src.distributionCap,'distributionCap',0,1e8);integer(src.productionCap,'productionCap',0,1e8);
 finite(src.collectionFraction,'collectionFraction',0,1);
 const ids=new Set();
 for(const p of src.products){
  if(!/^[A-Z0-9-]{1,32}$/.test(p.sku)||ids.has(p.sku))throw Error('Invalid or duplicate SKU');ids.add(p.sku);
  if(typeof p.name!=='string'||p.name.length>60)throw Error('Invalid product name');
  if(!Array.isArray(p.daily)||p.daily.length<28||p.daily.length>365)throw Error('Daily history must be 28–365 values');
  p.daily.forEach((v,i)=>integer(v,`daily[${i}]`,0,1e6));
  integer(p.onHand,'onHand',0,1e8); integer(p.reserved,'reserved',0,p.onHand);
  integer(p.incoming,'incoming',0,1e8);finite(p.unitCost,'unitCost',.01,1e7);
  finite(p.retail,'retail',.01,1e8);finite(p.wholesale,'wholesale',.01,1e8);
  finite(p.mix,'mix',0,1);finite(p.reach,'reach',0,1);
  finite(p.promo,'promo',0,1e8);integer(p.lead,'lead',1,90);integer(p.review,'review',1,90);
  if(!Array.isArray(p.stockoutDays) ||p.stockoutDays.some(x=>!Number.isInteger(x)||x<0||x>=p.daily.length))throw Error('Invalid stockout indexes');
 }
}
function forecast(p){
 const history=p.daily,errors=[],baselineErrors=[];
 const out=new Set(p.stockoutDays);
 for(let t=28;t<history.length;t++){
  if(out.has(t) || Array.from({length:28},(_,i)=>t-i-1).some(i=>out.has(i)))continue;
  const prev=history.slice(0,t);
  const pred=.5*mean(prev.slice(-7))+.3*mean(prev.slice(-14))+.2*mean(prev.slice(-28));
  errors.push(Math.abs(pred-history[t]));
  baselineErrors.push(Math.abs(mean(prev.slice(-7))-history[t]));
 }
 const day=.5*mean(history.slice(-7))+.3*mean(history.slice(-14))+.2*mean(history.slice(-28));
 const latest=history.slice(-28);const std=Math.sqrt(mean(latest.map(v=>(v-mean(latest))**2)));
 const safety=Math.ceil(1.645*std*Math.sqrt(p.lead+p.review));
 return {day,mae:errors.length?mean(errors):null,baselineMae:baselineErrors.length?mean(baselineErrors):null,
    safety,calibrated:errors.length>=7&&!out.size,sd:std,censored:!!out.size};
}
export function calculate(source){
 const src=structuredClone(source);validate(src);const tier=TIERS[src.tier];
 const products=src.products.map(p=>{
  const f=forecast(p),wDemand=f.day*7*p.reach*tier.factor;
  const target=Math.ceil(f.day*(p.lead+p.review)*p.reach*tier.factor+f.safety);
  const position=p.onHand-p.reserved+p.incoming;
  return {...p,forecast:round(f.day*p.reach*tier.factor),weeklyDemand:round(wDemand),safety:f.safety,
   mae:f.mae===null?null:round(f.mae),baselineMae:f.baselineMae===null?null:round(f.baselineMae),
   reliable:f.calibrated,censored:f.censored,orderUpTo:target,position,
   requested:Math.max(0,target-position),sellingPrice:round(p.retail*(1-p.mix)+p.wholesale*p.mix),
   available:p.onHand-p.reserved,proposed:0,sold:0,revenue:0,cogs:0,grossMargin:0};
 });
 const proposedMarketing=round(products.reduce((s,p)=>s+p.promo,0));
 const marketingHold=proposedMarketing>src.marketingCap;
 const freeCash=Math.max(0,src.cash-src.minReserve-tier.overhead-proposedMarketing);
 let remainingCash=freeCash,production=src.productionCap,delivery=src.distributionCap;
 const priority=[...products].sort((a,b)=>{
  const score=x=>Math.max(0,x.sellingPrice-x.unitCost)*x.weeklyDemand/Math.max(1,x.available+x.incoming);
  return score(b)-score(a)||a.sku.localeCompare(b.sku);
 });
 for(const p of priority){
  const units=Math.min(p.requested,production,Math.floor((remainingCash+1e-8)/p.unitCost));
  p.proposed=Math.max(0,units);production-=p.proposed;remainingCash-=p.proposed*p.unitCost;
 }
 for(const p of priority){
  const units=Math.min(p.available,Math.floor(p.weeklyDemand),delivery);
  p.sold=Math.max(0,units);delivery-=p.sold;
  p.revenue=round(p.sold*p.sellingPrice);p.cogs=round(p.sold*p.unitCost);
  p.grossMargin=round(p.revenue-p.cogs);
 }
 const sum=k=>round(products.reduce((s,p)=>s+p[k],0));
 const revenue=sum('revenue'),cogs=sum('cogs'),procurement=round(products.reduce((s,p)=>s+p.proposed*p.unitCost,0));
 const collections=round(revenue*src.collectionFraction);
 const contribution=round(revenue-cogs-proposedMarketing-tier.overhead);
 const closingCash=round(src.cash-procurement-proposedMarketing-tier.overhead+collections);
 const paidThisWeek=round(procurement+proposedMarketing+tier.overhead);
 const issues=[];
 if(marketingHold)issues.push('Promotional requests exceed campaign budget; do not execute marketing.');
 if(src.cash-paidThisWeek<src.minReserve)issues.push('Minimum pre-collection cash reserve would be breached.');
 if(closingCash<src.minReserve)issues.push('Closing cash forecast below required reserve.');
 if(contribution<=0)issues.push('Operating contribution is not positive.');
 if(products.some(p=>!p.reliable))issues.push('Forecast validation insufficient or stockout-censored history.');
 if(products.some(p=>p.weeklyDemand>p.available))issues.push('Expected stockouts: new procurement is not assumed available this week.');
 if(delivery===0)issues.push('Distribution capacity limit has been reached.');
 const scaleGate=issues.length===0?'REVIEW_READY':'HOLD';
 return {tier:tier.name,factor:tier.factor,products,issues,scaleGate,
  totals:{revenue,cogs,grossMargin:round(revenue-cogs),contribution,cash:closingCash,collections,
   receivables:round(revenue-collections),procurement,marketing:proposedMarketing,
   overhead:tier.overhead,unitsSold:products.reduce((s,p)=>s+p.sold,0),unitsOrdered:products.reduce((s,p)=>s+p.proposed,0),
   supplyCapacityUsed:src.distributionCap-delivery,availableCashForProcurement:round(freeCash)},
  executionEnabled:false,requiresHumanApproval:true,synthetic:true,
  methods:'7/14/28-day weighted moving-average; stockout censor and rolling MAE; no causal marketing/price elasticity'};
}

