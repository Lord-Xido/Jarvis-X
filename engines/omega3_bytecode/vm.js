'use strict';
function omega3(){
  const SIDE=1000000000000n, MASK=(1n<<40n)-1n, TILE=512, LIMIT=1048576;
  const orders=[[0,1,2],[0,2,1],[1,0,2],[1,2,0],[2,0,1],[2,1,0]];
  const check=(v,m)=>{if(!v)throw Error(m);};
  const eq=(a,b)=>a.length===b.length&&a.every((v,i)=>v===b[i]);
  const hash=a=>{let h=2166136261;for(const v of a)h=Math.imul(h^v,16777619)>>>0;return h;};
  const pad=n=>Math.ceil(n/TILE)*TILE;
  function address(x,y,z){const p=[x,y,z].map(BigInt);check(p.every(v=>v>=0n&&v<SIDE),'coordinate');return p[0]|p[1]<<40n|p[2]<<80n;}
  function coords(a){a=BigInt(a);check(a>=0n&&a<1n<<120n,'address');const p=[a&MASK,(a>>40n)&MASK,(a>>80n)&MASK];check(p.every(v=>v<SIDE),'unused address');return p;}
  function voxel(i){i=BigInt(i);check(i>=0n&&i<SIDE**3n,'voxel');const s=SIDE/8n,t=i/512n,j=Number(i%512n),p=[t%s,t/s%s,t/(s*s)].map(v=>v*8n);for(let b=0;b<3;b++)for(let a=0;a<3;a++)p[a]+=BigInt((j>>(3*b+a))&1)<<BigInt(b);return p;}
  function transform(data,id,inverse=false){check(Number.isInteger(id)&&id>=0&&id<=48&&data.length%TILE===0,'transform');const out=data.slice();if(!id)return out;const axes=orders[(id-1)%6],r=Math.floor((id-1)/6),bits=inverse?[2,1,0]:[0,1,2];for(const b of bits)for(const a of (inverse?[...axes].reverse():axes)){const m=1<<(3*b+a);for(let base=0;base<out.length;base+=TILE)for(let j=0;j<TILE;j++){if(j&m)continue;const u=base+j,v=u+m;const rot=x=>((x<<r)|(x>>>(8-r)))&255;if(inverse){if(r)out[u]^=rot(out[v]);out[v]^=out[u];}else{out[v]^=out[u];if(r)out[u]^=rot(out[v]);}}}return out;}
  function pack(latent,id,original){const z=[];for(let i=0;i<latent.length;){const zero=latent[i]===0;let n=1;while(n<128&&i+n<latent.length&&(latent[i+n]===0)===zero)n++;z.push((zero?128:0)|(n-1));if(!zero)for(let j=0;j<n;j++)z.push(latent[i+j]);i+=n;}const codec=Number(z.length<latent.length),p=codec?Uint8Array.from(z):latent,b=new Uint8Array(16+p.length),d=new DataView(b.buffer);b.set([74,88,51,68,1,id,codec,0]);d.setUint32(8,original.length,true);d.setUint32(12,hash(original),true);b.set(p,16);return b;}
  function unpack(b){check(b instanceof Uint8Array&&b.length>=16&&b.length<=LIMIT+16&&eq(b.slice(0,5),Uint8Array.of(74,88,51,68,1))&&b[7]===0,'header');const d=new DataView(b.buffer,b.byteOffset,b.byteLength),n=d.getUint32(8,true),id=b[5],codec=b[6];check(n<=LIMIT&&id<=48&&codec<=1,'header fields');const p=b.subarray(16),out=new Uint8Array(pad(n));if(!codec){check(p.length===out.length,'raw length');out.set(p);}else{let i=0,j=0;while(i<p.length){const tag=p[i++],k=(tag&127)+1;check(j+k<=out.length,'run overflow');if(!(tag&128)){check(i+k<=p.length,'literal truncation');out.set(p.subarray(i,i+k),j);i+=k;}j+=k;}check(j===out.length,'run truncation');}return {latent:out,id,n,checksum:d.getUint32(12,true)};}
  function decode(e){const p=transform(e.latent,e.id,true);check(p.subarray(e.n).every(x=>x===0),'padding');const x=p.slice(0,e.n);check(hash(x)===e.checksum,'checksum');return x;}
  const words=[[1,0,0,0,0],[2,1,0,0,0],[3,2,1,1,0],[4,3,2,0,0],[5,4,3,0,0],[6,5,4,0,0],[7,6,5,0,0],[8,7,3,6,0],[9,1,5,0,1],[10,8,7,5,0]];
  function program(iterations=1000){check(Number.isInteger(iterations)&&iterations>=1&&iterations<=1000,'iteration bound');const b=new Uint8Array(80),v=new DataView(b.buffer);words.forEach((w,i)=>{b.set(w.slice(0,4),8*i);v.setUint32(8*i+4,i===0?iterations:w[4]);});return b;}
  function run(input,bytecode=program(),adaptive=true){check(input instanceof Uint8Array&&input.length<=LIMIT,'input bound');check(bytecode instanceof Uint8Array&&bytecode.length===80,'program length');const d=new DataView(bytecode.buffer,bytecode.byteOffset,bytecode.byteLength),iterations=d.getUint32(4);check(iterations>=1&&iterations<=1000,'iteration bound');check(pad(input.length)*iterations<=16*1024*1024,'workload bound');for(let i=0;i<10;i++){const w=words[i];check(w.slice(0,4).every((x,j)=>bytecode[8*i+j]===x)&&(!i||d.getUint32(8*i+4)===w[4]),'unsupported instruction operands');}
    const original=input.slice(),R=Array(9),trace=[],scores=new Float64Array(48),counts=new Uint32Array(48);let pc=0,t=0,id=0,best,accepted=0,executed=0,verified=false;
    // A trainable 1-input / 48-output linear neural policy, with exhaustive warmup.
    // Squared prediction error SGD updates the selected output. No claim of generalization.
    const feature=1,baselinePadded=new Uint8Array(pad(original.length));baselinePadded.set(original);
    while(pc<10){const off=pc*8,op=bytecode[off],dst=bytecode[off+1],src=bytecode[off+2],aux=bytecode[off+3],imm=d.getUint32(off+4);executed++;let next=pc+1;
      switch(op){
        case 1:R[dst]=original.slice();best=pack(baselinePadded,0,original);check(eq(decode(unpack(best)),original),'baseline');break;
        case 2:R[dst]=new Uint8Array(pad(R[src].length));R[dst].set(R[src]);verified=false;break;
        case 3:id=1+t%48;if(adaptive&&t>=48){let k=0;for(let j=1;j<48;j++)if(scores[j]>scores[k])k=j;id=k+1;}R[dst]=transform(R[src],id);break;
        case 4:R[dst]=pack(R[src],id,original);break;
        case 5:R[dst]=unpack(R[src]);break;
        case 6:R[dst]=decode(R[src]);break;
        case 7:verified=eq(R[src],original);R[dst]=verified;check(verified,'exact verification failed');break;
        case 8:{check(verified&&R[aux]===true,'commit gate');const candidate=R[src],accept=candidate.length<best.length;if(accept){best=candidate.slice();accepted++;}R[dst]=best;const k=id-1,reward=1-candidate.length/Math.max(16,original.length+16);counts[k]++;scores[k]+=(reward-scores[k]*feature)*feature/counts[k];trace.push({iteration:t+1,id,bytes:candidate.length,best:best.length,accept,hash:hash(R[5]).toString(16).padStart(8,'0')});break;}
        case 9:check(verified,'feedback gate');R[0]=R[src];R[dst]=R[src];t++;if(t<iterations)next=imm;break;
        case 10:R[dst]={container:R[src].slice(),output:R[aux].slice()};break;
        default:throw Error('unknown opcode');
      }pc=next;
    }
    check(eq(R[8].output,original)&&eq(decode(unpack(R[8].container)),original),'emit gate');return {...R[8],trace,accepted,executed,iterations:t,baseline:pack(baselinePadded,0,original).length};
  }
  function selftest(){let checks=0;const test=(v,m)=>{check(v,m);checks++;};const throws=f=>{let failed=false;try{f();}catch{failed=true;}test(failed,'expected rejection');};let seed=0x12345678;const random=n=>Uint8Array.from({length:n},()=>{seed^=seed<<13;seed^=seed>>>17;seed^=seed<<5;return seed&255;});
    for(const n of [0,1,7,511,512,513,8192]){const a=random(n),p=new Uint8Array(pad(n));p.set(a);for(let id=0;id<=48;id++)test(eq(decode(unpack(pack(transform(p,id),id,a))),a),'codec roundtrip');}
    const a=Uint8Array.from({length:8192},(_,i)=>Math.floor(i/512)*9),r=run(a,program(1000));test(eq(r.output,a),'VM roundtrip');test(r.executed===8002,'instruction count');test(r.trace.every((v,i)=>!i||v.best<=r.trace[i-1].best),'monotonic');test(r.container.length===110,'fixture compatibility');test(r.iterations===1000,'iterations');test(r.trace.slice(0,48).every((v,i)=>v.id===i+1),'warmup');test(r.trace.slice(48).every(v=>v.id===1),'policy');
    for(const p of [[0n,0n,0n],[SIDE-1n,SIDE-1n,SIDE-1n],[123n,456n,789n]])test(coords(address(...p)).every((v,i)=>v===p[i]),'address');
    test(voxel(SIDE**3n-1n).every(v=>v===SIDE-1n),'voxel');const corrupt=r.container.slice();corrupt[12]^=1;throws(()=>decode(unpack(corrupt)));const bad=program(1);bad[16]=255;throws(()=>run(a,bad));throws(()=>program(0));test(eq(run(new Uint8Array(0),program(1)).output,new Uint8Array(0)),'empty');
    return {checks,iterations:r.iterations,instructions:r.executed,original:a.length,baseline:r.baseline,best:r.container.length,accepted:r.accepted,hash:hash(a).toString(16),roundTrip:true};
  }
  return {address,coords,voxel,transform,pack,unpack,decode,program,run,selftest};
}
const vm=omega3();
module.exports=vm;
if(require.main===module){const fs=require('node:fs');const [cmd,input,output,iterations]=process.argv.slice(2);if(!cmd||cmd==='selftest'){console.log(JSON.stringify(vm.selftest(),null,2));}else if(cmd==='encode'&&input&&output){const r=vm.run(new Uint8Array(fs.readFileSync(input)),vm.program(iterations===undefined?1:Number(iterations)));fs.writeFileSync(output,r.container,{flag:'wx'});console.log(JSON.stringify({iterations:r.iterations,instructions:r.executed,trace:r.trace}));}else if(cmd==='decode'&&input&&output){const x=vm.decode(vm.unpack(new Uint8Array(fs.readFileSync(input))));fs.writeFileSync(output,x,{flag:'wx'});}else{throw Error('usage: node vm.js selftest | encode INPUT OUTPUT [iterations] | decode INPUT OUTPUT');}}
