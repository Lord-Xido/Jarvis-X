"use strict";
// Experimental S3D-BCx reference: bounded 3D Q16.16 stencil, not trained ANN.
function reference({side=8,limit=1000,source,adapter=null,corruptResidual=false,badCandidate=false,tamperCipher=false}={}) {
  if(!Number.isInteger(side)||side<4||side>32||side%4||!Number.isInteger(limit)||limit<1||limit>1000) throw Error("BAD_CONFIG");
  const count=side**3*3, Q=65536;
  if(!(source instanceof Uint8Array)||source.length!==count) throw Error("BAD_SOURCE");
  const src=source.slice(), words=[
    [1,0,0,0],[0x20,1,0x1000,0],[0xb2,1,0x1100,0],
    [0x30,2,0x2000,0],[0x41,3,0x3000,0],[0x44,4,0x4100,0],
    [0xb6,4,0x9000,0],[0xb3,7,0x7300,0],[0xb4,8,0x7300,0],
    [0xb7,9,0x9000,0],[0x15,9,0x2000,3],[0x52,9,0x5000,0],
    [0x82,9,0x8100,0],[0x8f,9,0,0]
  ].map(([op,ph,addr,imm])=>[op,ph,0,255,255,addr>>8,addr&255,imm]);
  const trace=[], deltas=[], idx=(x,y,z,c)=>(((z*side+y)*side+x)*3+c), wrap=x=>(x+side)%side;
  let initial,residual,state,target,candidate,refined,out,sealed,packet,omega="0".repeat(64),pc=0,iterations=0,delta=0,shards=0;
  const distance=(a,b)=>a.reduce((s,v,i)=>s+Math.abs(v-b[i]),0);
  function stencil(a) {
    const b=new Int32Array(count);
    for(let sh=0;sh<4;sh++) for(let z=sh*side/4;z<(sh+1)*side/4;z++)
    for(let y=0;y<side;y++) for(let x=0;x<side;x++) for(let c=0;c<3;c++) {
      const i=idx(x,y,z,c), get=(dx,dy,dz)=>a[idx(wrap(x+dx),wrap(y+dy),wrap(z+dz),c)];
      b[i]=Math.floor((2*a[i]+get(1,0,0)+get(-1,0,0)+get(0,1,0)+get(0,-1,0)+get(0,0,1)+get(0,0,-1)+4)/8);
    }
    return b;
  }
  function verify() {
    for(let i=0;i<count;i++) {
      if(((initial[i]/Q)^residual[i])!==src[i]) throw Error("SOURCE_INTEGRITY");
      if(candidate[i]<0||candidate[i]>255*Q) throw Error("BOUND");
    }
    const before=distance(state,target),after=distance(candidate,target);
    if(after>before||(before>0&&after===before)) throw Error("OBJECTIVE");
  }
  while(pc<words.length) {
    const word=words[pc], op=word[0], address=word[5]*256+word[6];
    const hex=word.map(x=>x.toString(16).padStart(2,"0")).join("");
    trace.push({pc,word:hex,phase:word[1],descriptor:address,iteration:iterations});
    switch(op) {
      case 1: break;
      case 0x20: initial=Int32Array.from(src,v=>(v&240)*Q);state=initial.slice();break;
      case 0xb2: residual=Uint8Array.from(src,(v,i)=>v^(initial[i]/Q));if(corruptResidual)residual[0]^=1;break;
      case 0x30: target=stencil(initial);break;
      case 0x41: candidate=Int32Array.from(state,(v,i)=>{const d=target[i]-v;return v+Math.sign(d)*Math.ceil(Math.abs(d)/2);});break;
      case 0x44: for(let i=0;i<count;i++)candidate[i]=Math.max(0,Math.min(255*Q,candidate[i]));if(badCandidate)candidate[0]=-1;break;
      case 0xb6: verify();break;
      case 0xb3: packet=JSON.stringify({version:1,side,iteration:iterations,state:Array.from(candidate),sourceLatent:Array.from(initial),residual:Array.from(residual),omega});if(adapter)sealed=adapter.seal(packet,iterations);break;
      case 0xb4: if(adapter&&adapter.open(sealed,iterations,tamperCipher)!==packet)throw Error("CHECKPOINT_INTEGRITY");break;
      case 0xb7:
        delta=0;for(let i=0;i<count;i++)delta=Math.max(delta,Math.abs(candidate[i]-state[i]));
        state=candidate;iterations++;deltas.push(delta);
        if(adapter)omega=adapter.hash(omega+":"+hex+":"+packet);
        if(delta===0){pc++;continue;}
        if(iterations>=limit)throw Error("NOT_CONVERGED");
        pc=3;continue;
      case 0x15: shards=4;break;
      case 0x52: refined=stencil(state);break;
      case 0x82: out=Uint8Array.from(refined.slice(0,side*side*3),v=>Math.round(v/Q));break;
      case 0x8f: return {side,iterations,deltas,omega:adapter?omega:null,shards,trace,output:Array.from(out),state:Array.from(state),words:words.map(w=>w.map(v=>v.toString(16).padStart(2,"0")).join(""))};
      default: throw Error("BAD_OPCODE");
    }
    pc++;
  }
  throw Error("NO_HALT");
}
function selftest(adapterFactory=null) {
  const source=Uint8Array.from({length:8**3*3},(_,i)=>(i*73+Math.floor(i/7)*19)%256);
  const run=(opts={})=>reference({source,adapter:adapterFactory?adapterFactory():null,...opts});
  const a=run(),b=run();
  if(a.deltas.at(-1)!==0||a.shards!==4||a.output.length!==192||JSON.stringify(a.state)!==JSON.stringify(b.state)||a.omega!==b.omega||a.trace.at(-1).word.slice(0,2)!=="8f")throw Error("REPLAY");
  for(const [opts,expected] of [[{limit:1},"NOT_CONVERGED"],[{corruptResidual:true},"SOURCE_INTEGRITY"],[{badCandidate:true},"BOUND"],...(adapterFactory?[[{tamperCipher:true},"AUTH_FAIL"]]:[])]) {
    let caught=false;try{run(opts);}catch(e){if(e.message!==expected)throw e;caught=true;}if(!caught)throw Error("EXPECTED_TRAP:"+expected);
  }
  for(const [x,y,z] of [[0,0,0],[8191,8191,8191],[13,47,7000]]) {
    const a=z*2**26+y*8192+x;if(a%8192!==x||Math.floor(a/8192)%8192!==y||Math.floor(a/2**26)!==z)throw Error("ADDRESS");
  }
  return {iterations:a.iterations,executedWords:a.trace.length,outputBytes:a.output.length,finalDelta:a.deltas.at(-1),omega:a.omega,cryptoTested:!!adapterFactory,tests:"PASS"};
}

const crypto=require("node:crypto");
function adapterFactory() {
  const key=crypto.randomBytes(32),prefix=crypto.randomBytes(8);
  let next=0;
  return {
    hash: s=>crypto.createHash("sha3-256").update(s).digest("hex"),
    seal: (s,i)=>{
      if(i!==next++||i>=2**32)throw Error("NONCE_SEQUENCE");
      const nonce=Buffer.alloc(12);prefix.copy(nonce);nonce.writeUInt32BE(i,8);
      const cipher=crypto.createCipheriv("aes-256-gcm",key,nonce);
      cipher.setAAD(Buffer.from("S3DBCX-v1:"+i));
      const data=Buffer.concat([cipher.update(s,"utf8"),cipher.final()]);
      return {nonce,data,tag:cipher.getAuthTag()};
    },
    open: (p,i,tamper)=>{
      try{
        const data=Buffer.from(p.data);if(tamper)data[0]^=1;
        const d=crypto.createDecipheriv("aes-256-gcm",key,p.nonce);
        d.setAAD(Buffer.from("S3DBCX-v1:"+i));d.setAuthTag(p.tag);
        return Buffer.concat([d.update(data),d.final()]).toString("utf8");
      }catch(e){throw Error("AUTH_FAIL");}
    }
  };
}
if(require.main===module)console.log(JSON.stringify(selftest(adapterFactory),null,2));
module.exports={reference,selftest,adapterFactory};
