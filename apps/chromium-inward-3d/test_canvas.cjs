'use strict';
// A real software Canvas with a deliberately small DOM adapter. Native browser
// layout, event delivery, and CSS are validated separately by test_browser.cjs.
const fs=require('fs'),vm=require('vm'),assert=require('assert/strict'),path=require('path');
const {spawnSync}=require('child_process'),{performance}=require('perf_hooks');
const {createCanvas}=require('@napi-rs/canvas');
const OUT=path.join(__dirname,'test-results','canvas');
fs.mkdirSync(OUT,{recursive:true});
const fragment=fs.readFileSync(path.join(__dirname,'index.html'),'utf8');
const script=fragment.match(/<script>\s*([\s\S]*?)<\/script>/)[1];
const palettes={
 light:{background:'rgb(255, 255, 255)',foreground:'rgb(26, 28, 31)',card:'rgb(244, 244, 244)',border:'rgba(26, 28, 31, 0.15)','muted-foreground':'rgb(98, 99, 102)','viz-series-1':'rgb(51, 156, 255)','viz-series-2':'rgb(243, 136, 59)','viz-series-3':'rgb(93, 201, 119)','viz-series-4':'rgb(235, 119, 177)'},
 dark:{background:'rgb(24, 24, 24)',foreground:'rgb(255, 255, 255)',card:'rgb(38, 38, 38)',border:'rgba(255, 255, 255, 0.18)','muted-foreground':'rgb(162, 162, 162)','viz-series-1':'rgb(131, 195, 255)','viz-series-2':'rgb(245, 154, 86)','viz-series-3':'rgb(116, 213, 139)','viz-series-4':'rgb(240, 143, 192)'}
};
function mount({theme='light',width=736,faultEdge=0,roundCopies=false,blockedStorage=false,persisted=null}={}){
 let virtualTime=0,timerID=0,rafID=0;const timers=new Map(),rafs=new Map(),resizeCallbacks=[],storage=new Map();
 if(persisted)storage.set('chromium-3d-inward-policy-v1',persisted);
 class E{
  constructor(tag,attrs={},children=[]){
   this.tagName=tag.toUpperCase();this.nodeType=1;this.attrs=attrs;this.style={};this.events={};this.parent=null;this.childNodes=[];this.value='';this.disabled=false;this.checked=Object.hasOwn(attrs,'checked');
   this.classList={add:s=>{this.className=[...new Set([...this.className.split(/\s+/),s])].join(' ').trim();},remove:s=>{this.className=this.className.split(/\s+/).filter(x=>x!==s).join(' ');},contains:s=>this.className.split(/\s+/).includes(s)};
   children.forEach(c=>this.append(c));
   if(tag==='canvas'){
    this.canvas=createCanvas(Number(attrs.width)||1,Number(attrs.height)||1);const context=this.canvas.getContext('2d');
    this.proxy=new Proxy(context,{get:(target,key)=>{const v=target[key];if(typeof v!=='function')return v;if(key==='drawImage')return(image,...args)=>{
      const native=image.canvas||image;if(faultEdge && native.width===faultEdge && native.height<=faultEdge)return;
      const result=target.drawImage(native,...args);
      if(roundCopies && native.width<=128 && native.height<=128 && this.canvas.width===320 && this.canvas.height===226){
       const x=Math.max(0,Math.min(319,args[0])),y=Math.max(26,Math.min(225,args[1])),pixel=target.getImageData(x,y,1,1);
       pixel.data[0]^=1;target.putImageData(pixel,x,y);
      }
      return result;
     };if(key==='putImageData')return(pixels,...args)=>{if(faultEdge && pixels.width===faultEdge && pixels.height<=faultEdge)return;return target.putImageData(pixels,...args);};return v.bind(target);},set:(target,key,value)=>{target[key]=value;return true;}});
   }
  }
  append(c){this.childNodes.push(c);c.parent=this;}
  get children(){return this.childNodes.filter(c=>c.nodeType===1);}
  get className(){return this.attrs.class||'';}set className(v){this.attrs.class=v;}
  get textContent(){return this.childNodes.map(c=>c.textContent).join('');}set textContent(v){this.childNodes=[{nodeType:3,textContent:String(v),parent:this}];}
  get width(){return this.canvas.width;}set width(v){this.canvas.width=v;}get height(){return this.canvas.height;}set height(v){this.canvas.height=v;}
  getContext(){return this.proxy;}getBoundingClientRect(){return {width,height:width<520?440:470,left:0,top:0};}setPointerCapture(){}
  matches(selector){if(selector[0]==='#')return this.attrs.id===selector.slice(1);if(selector[0]==='.')return this.className.split(/\s+/).includes(selector.slice(1));if(selector[0]==='['){const m=selector.match(/^\[([^=]+)="?([^"\]]+)"?\]$/);return !!m&&this.attrs[m[1]]===m[2];}return this.tagName.toLowerCase()===selector.toLowerCase();}
  querySelectorAll(selector){const parts=selector.trim().split(/\s+/),out=[];const visit=e=>{for(const c of e.children){if(c.matches(parts.at(-1))){let p=c.parent,i=parts.length-2;while(p&&i>=0){if(p.matches(parts[i]))i--;p=p.parent;}if(i<0)out.push(c);}visit(c);}};visit(this);return out;}
  querySelector(s){return this.querySelectorAll(s)[0]||null;}
  addEventListener(name,fn){(this.events[name] ||= []).push(fn);}emit(name,extra={}){for(const fn of this.events[name]||[])fn({target:this,...extra});}
 }
 function parse(html){const res=spawnSync(process.env.PYTHON || 'python3',[path.join(__dirname,'test-support','parse_dom.py')],{input:html,encoding:'utf8'});assert.equal(res.status,0,res.stderr || String(res.error || ''));const build=n=>Object.hasOwn(n,'text')?{nodeType:3,textContent:n.text}:new E(n.tag,n.attrs,(n.children||[]).map(build));return build(JSON.parse(res.stdout));}
 const document=parse(fragment);document.getElementById=id=>document.querySelector('#'+id);document.createElement=tag=>new E(tag);document.hidden=false;
 class DOMParser{parseFromString(html){return parse(html);}}class ResizeObserver{constructor(cb){this.cb=cb;resizeCallbacks.push(cb);}observe(){this.cb();}}class IntersectionObserver{constructor(cb){this.cb=cb;}observe(){this.cb([{isIntersecting:true}]);}}
 const media={};
 const window={devicePixelRatio:1,matchMedia:q=>(media[q] ||= {matches:false,events:[],addEventListener(_,fn){this.events.push(fn);}}),localStorage:{getItem:key=>{if(blockedStorage)throw new Error('Opaque origin');return storage.get(key)||null;},setItem:(key,value)=>{if(blockedStorage)throw new Error('Opaque origin');storage.set(key,value);}}};
 const sandbox={document,window,DOMParser,ResizeObserver,IntersectionObserver,console,TextEncoder,performance:{now:()=>performance.now()+virtualTime},getComputedStyle:e=>({color:palettes[theme][(e.style.color||'').replace(/^var\(--|\)$/g,'')]||palettes[theme].foreground,fontSize:e.classList.contains('text-small')?'12px':'14px',fontFamily:'Arial'}),requestAnimationFrame:fn=>{rafs.set(++rafID,fn);return rafID;},setTimeout:(fn,delay)=>{timers.set(++timerID,{at:virtualTime+delay,fn});return timerID;},clearTimeout:id=>timers.delete(id)};
 vm.runInNewContext(script,sandbox,{filename:'inward-feedback.js'});
 const root=document.getElementById('chromium-rendering-3d'),api=root.crEmulator;
 const click=action=>root.querySelector(`[data-action="${action}"]`).emit('click');
 function frame(){const callbacks=[...rafs.values()];rafs.clear();callbacks.forEach(fn=>fn(performance.now()+virtualTime));}
 function flush(){let rounds=0;while(api.snapshot().playing || api.snapshot().optimising || api.snapshot().feedbackQueued){assert.ok(rounds++<100,'Every pass terminates.');if(api.snapshot().playing){virtualTime+=900;frame();}else {const next=[...timers.entries()].sort((a,b)=>a[1].at-b[1].at)[0];assert.ok(next,'A pending probe has a timer.');timers.delete(next[0]);virtualTime=Math.max(virtualTime,next[1].at);next[1].fn();frame();}}return api.snapshot();}
 function auto(on){const c=root.querySelector('[data-action="auto"]');c.checked=on;c.emit('change');}
 async function capture(name){const scene=root.querySelector('.cr-scene'),image=createCanvas(scene.width,scene.height),c=image.getContext('2d');c.fillStyle=palettes[theme].background;c.fillRect(0,0,image.width,image.height);c.drawImage(scene.canvas,0,0);await fs.promises.writeFile(OUT+'/'+name+'.png',await image.encode('png'));}
 return {api,root,document,click,flush,auto,capture,storage,timers,theme:next=>{theme=next;window.matchMedia('(prefers-color-scheme: dark)').events.forEach(fn=>fn());},resize:next=>{width=next;resizeCallbacks.forEach(fn=>fn());}};
}
(async()=>{
 const m=mount(),initial=m.api.snapshot();assert.equal(initial.pixelDiff,0);assert.equal(initial.frame,1);const originalHash=initial.frameHash;
 const settled=m.flush(),f=m.api.feedback();
 assert.equal(settled.pixelDiff,0);assert.equal(settled.frameHash,originalHash,'Optimisation preserves the actual presented image.');
 assert.ok(f.converged,'The controller reaches a fixed policy within its finite neighbourhood.');
 assert.ok(f.best.work.drawCalls<f.baseline.work.drawCalls,'Accepted policies reduce counted drawing calls.');
 assert.ok(f.best.work.stageExecutions<=f.baseline.work.stageExecutions);
 assert.ok(f.accepted>=2);assert.ok(f.iterations<=8);assert.ok(f.radius<f.initialRadius/50,'The feedback coordinates actually contract in three dimensions.');
 assert.ok(f.history.every(r=>!r.accepted || r.mismatchedPixels===0));assert.equal(f.best.checkedPixels,433920);
 assert.equal(f.policy.reuseDocument,true);assert.ok(f.omega.some(v=>v>0));assert.ok(m.storage.size>0);
 await m.capture('inward-3d-light');
 const before=m.api.snapshot();const audit=m.api.probePolicy(before.policy);assert.equal(audit.valid,true);assert.equal(m.api.snapshot().frameHash,before.frameHash,'Isolated probes cannot alter the live framebuffer.');
 m.auto(false);m.click('run');const replay=m.flush();assert.deepEqual(Array.from(replay.plan),[0,8,9]);assert.equal(replay.work.stageExecutions,3);assert.equal(replay.rasterized,0);assert.equal(replay.pixelDiff,0);
 m.click('mutate');const mutation=m.flush();assert.equal(mutation.revision,1);assert.notEqual(mutation.frameHash,originalHash);assert.equal(mutation.pixelDiff,0);assert.ok(mutation.reused>0 && mutation.rasterized>0);
 m.click('scroll');const scroll=m.flush();assert.equal(scroll.scroll,24);assert.equal(scroll.pixelDiff,0);assert.equal(scroll.rasterized,0);assert.deepEqual(Array.from(scroll.plan),[8,9]);
 const reference=mount();reference.auto(false);reference.click('mutate');reference.flush();reference.click('scroll');reference.flush();assert.equal(scroll.frameHash,reference.api.snapshot().frameHash,'Optimised mutation and scroll match the untuned renderer.');
 const picker=m.root.querySelector('#cr-stage-select');picker.value='2';picker.emit('change');assert.equal(m.api.snapshot().stage,2);picker.value='9';picker.emit('change');assert.equal(m.api.snapshot().pixelDiff,0);
 const scene=m.root.querySelector('.cr-scene'),yaw=m.api.snapshot().yaw;scene.emit('pointerdown',{pointerId:1,clientX:180,clientY:190,pointerType:'mouse'});scene.emit('pointermove',{pointerId:1,clientX:240,clientY:205});scene.emit('pointerup',{pointerId:1,clientX:240,clientY:205});assert.notEqual(m.api.snapshot().yaw,yaw);
 m.theme('dark');assert.equal(m.api.snapshot().pixelDiff,0);m.click('optimise');m.flush();await m.capture('inward-3d-dark');m.resize(320);await m.capture('inward-3d-mobile');assert.equal(scene.width,320);
 const persisted=m.storage.get('chromium-3d-inward-policy-v1'),reload=mount({persisted});
 const {tile,tightBounds,visibleOnly,reuseDocument}=JSON.parse(persisted);
 assert.deepEqual(JSON.parse(JSON.stringify(reload.api.snapshot().policy)),{tile,tightBounds,visibleOnly,reuseDocument});assert.equal(reload.api.snapshot().pixelDiff,0);reload.auto(false);
 for(const corrupt of ['{broken',JSON.stringify({tile:0}),JSON.stringify({tile:129,omega:[NaN]}),JSON.stringify({tile:'128',tightBounds:true})]){
  const invalid=mount({persisted:corrupt});invalid.auto(false);assert.equal(invalid.api.snapshot().policy.tile,64);assert.equal(invalid.api.snapshot().pixelDiff,0);
 }
 const faulty=mount({faultEdge:128});faulty.flush();const failure=faulty.api.feedback();assert.ok(failure.history.some(r=>r.policy.tile===128 && r.mismatchedPixels>0 && !r.accepted),'A candidate that drops tile pixels is rejected.');assert.notEqual(failure.policy.tile,128);assert.equal(faulty.api.snapshot().pixelDiff,0);assert.equal(faulty.api.snapshot().frameHash,originalHash);
 const poisoned=mount({faultEdge:128,persisted:JSON.stringify({tile:128,tightBounds:true,visibleOnly:true,reuseDocument:true,omega:[0,0,0,0]})});assert.equal(poisoned.api.snapshot().pixelDiff,0);assert.equal(poisoned.api.snapshot().policy.tile,64);assert.equal(poisoned.api.feedback().rollbacks,1);poisoned.auto(false);
 const blocked=mount({blockedStorage:true});blocked.flush();assert.equal(blocked.api.snapshot().pixelDiff,0);assert.ok(blocked.api.feedback().converged,'Blocked device storage does not stop the renderer.');
 const rounded=mount({roundCopies:true,faultEdge:128});const roundedInitial=rounded.api.snapshot();
 assert.equal(roundedInitial.compositionBackend,'rgba-copy');assert.equal(roundedInitial.pixelDiff,0);assert.equal(roundedInitial.frameHash,originalHash,'Byte-copy fallback preserves every reference pixel without tolerances.');
 assert.ok(rounded.api.inspect(9).data.tileCopyFallback.mismatchedPixels>0);rounded.flush();
 assert.ok(rounded.api.feedback().history.some(r=>r.policy.tile===128 && r.mismatchedPixels>0 && !r.accepted),'Byte-copy fallback still rejects dropped candidate tiles.');
 assert.equal(rounded.api.snapshot().pixelDiff,0);assert.equal(rounded.api.snapshot().frameHash,originalHash);
 rounded.auto(false);rounded.click('mutate');rounded.flush();rounded.click('scroll');const roundedScroll=rounded.flush();
 assert.equal(roundedScroll.frameHash,reference.api.snapshot().frameHash,'Byte-copy dirty rows preserve viewport clipping and browser chrome on scroll.');
 const cancellation=mount();cancellation.click('optimise');assert.equal(cancellation.api.snapshot().optimising,true);cancellation.click('optimise');assert.equal(cancellation.api.snapshot().optimising,false);assert.equal(cancellation.api.snapshot().pixelDiff,0);cancellation.auto(false);
 const report={schemaVersion:1,status:'PASS',environment:{node:process.version,canvas:require('@napi-rs/canvas/package.json').version},initial,settled,feedback:f,replay,mutation,scroll,faultRejection:failure.history.filter(r=>r.mismatchedPixels>0),byteCopyFallback:roundedInitial,byteCopyFaultRejection:true,byteCopyScroll:true,blockedStorage:true,policyPersistence:true,corruptStorage:true,livePolicyRollback:true,pause:true,isolatedProbes:true,scope:'Native software Canvas with HTMLParser-backed DOM adapter; native browser behavior is reported by the separate browser suite.'};
 fs.writeFileSync(OUT+'/inward-validation.json',JSON.stringify(report,null,2));
 console.log(JSON.stringify({status:'PASS',drawingCalls:[f.baseline.work.drawCalls,f.best.work.drawCalls],stageExecutions:[f.baseline.work.stageExecutions,f.best.work.stageExecutions],accepted:f.accepted,rejected:f.rejected,policy:f.policy,pixelsPerProbe:f.best.checkedPixels,livePixelDiff:settled.pixelDiff,contraction:[f.initialRadius,f.radius],faultsRejected:true,rollback:true,storage:true,nativeBrowser:false},null,2));
})().catch(error=>{console.error(error);process.exitCode=1;});
