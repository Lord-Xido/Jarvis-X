// JARVIS-X experimental bounded 64-bit volumetric ISA, 2026-11-26.
// Not the canonical JX3DVM1 authority or a trained ANN. Run: node isa64.mjs
function isa64(){
  const L=10000,Q=65536n, MASK=0xfffffffn;
  const pack=(op,mode,rd,s1,s2,imm)=>{for(const [v,max] of [[op,255],[mode,15],[rd,255],[s1,255],[s2,255]])if(!Number.isInteger(v)||v<0||v>max)throw Error("field range");if(!Number.isInteger(imm)||imm<-(2**27)||imm>2**28-1)throw Error("immediate range");return BigInt(op)<<56n|BigInt(mode)<<52n|BigInt(rd)<<44n|BigInt(s1)<<36n|BigInt(s2)<<28n|(BigInt(imm)&MASK)};
  const unpack=w=>{if(typeof w!=="bigint"||w<0n||w>=1n<<64n)throw Error("word range");const i=Number(w&MASK);return [Number(w>>56n),Number(w>>52n&15n),Number(w>>44n&255n),Number(w>>36n&255n),Number(w>>28n&255n),i,i>=2**27?i-2**28:i]};
  const mort=(x,y,z)=>{let m=0n;for(const v of [x,y,z])if(!Number.isInteger(v)||v<0||v>=L)throw Error("coordinate");for(let i=0n;i<14n;i++)for(const [j,v] of [x,y,z].entries())m|=(BigInt(v)>>i&1n)<<(3n*i+BigInt(j));return m};
  const unmort=(m,axis)=>{if(!Number.isInteger(axis)||axis<0||axis>2||m<0n||m>=1n<<42n)throw Error("Morton range");let c=0n;for(let i=0n;i<14n;i++)c|=(m>>(3n*i+BigInt(axis))&1n)<<i;if(c>=BigInt(L))throw Error("coordinate");return Number(c)};
  const idx=(x,y,z)=>{for(const v of [x,y,z])if(!Number.isInteger(v)||v<0||v>=L)throw Error("coordinate");return BigInt(x)+10000n*BigInt(y)+100000000n*BigInt(z)};
  const mul=(a,b)=>a*b>>16n;
  const codebook=[0n,Q/4n,Q/2n,3n*Q/4n,Q];
  const encode=v=>{let token=0;const abs=x=>x<0n?-x:x;for(let i=1;i<codebook.length;i++)if(abs(v-codebook[i])<abs(v-codebook[token]))token=i;return {token,residual:v-codebook[token]}};
  const decode=({token,residual})=>{if(!Number.isInteger(token)||token<0||token>=codebook.length||typeof residual!=="bigint")throw Error("invalid token");return codebook[token]+residual};
  // PDF profile: Z0 implicit in MORT3D/LINIDX. Q_FMA is scaled multiply, not fused add.
  const program=[
    [1,0,0x40,0,0x10,0],[0x10,3,0x50,0x40,0,3],[0x11,0,0,0x50,0,11],
    [2,0,0x30,0,0x10,L],[0x20,1,0x61,0x80,0,0],[0x20,1,0x62,0x80,0x10,1],[0x20,1,0x63,0x80,0x20,2],
    [0x22,1,0x60,0x61,0x62,16],[0x22,1,0x60,0x60,0x63,16],[0x30,1,0x64,0x40,0x60,63],[0x22,1,0x60,0x60,0x64,16],
    [0x21,0,0x31,0x40,0x60,1],[0x40,1,0,0x60,0x31,0],[0x41,0,0,0x10,0x20,-13],[0xff,0,0,0,0,0]
  ].map(a=>pack(...a));
  // Reference profile: bounded origin brick, periodic brick boundaries, read-only previous state.
  // Occupancy is depth-limited within the brick, with empty regions treated as zero.
  function sweep(side,previous,factors,occupied,trace=false){
    if(!Number.isInteger(side)||side<1||side>32)throw Error("physical side bound 1..32");
    if(previous.size>side**3||occupied.size>side**3)throw Error("active bound");
    if(factors.length!==3||factors.some(a=>a.length!==side||a.some(v=>typeof v!=="bigint"||v<0n||v>Q)))throw Error("factor domain");
    const local=(m)=>[0,1,2].map(a=>unmort(m,a));
    for(const m of occupied)if(local(m).some(c=>c>=side))throw Error("occupancy coordinate");
    for(const [m,v] of previous)if(local(m).some(c=>c>=side)||typeof v!=="bigint"||v<0n||v>Q)throw Error("previous state");
    const r=Array(256).fill(0n), next=new Map(), tokens=new Map(), log=[];
    const bits=Math.ceil(Math.log2(side)), depth=Math.min(3,bits), shift=BigInt(3*(bits-depth));
    const prefixes=new Set([...occupied].map(m=>m>>shift));
    let pc=0,steps=0,active=0,pruned=0;
    const budget=14*side**3+1;
    while(true){
      if(++steps>budget||pc<0||pc>=program.length)throw Error("execution bound");
      const [op,mode,d,a,b,i,si]=unpack(program[pc]);const oldPC=pc;let jump=pc+1;
      switch(op){
        case 1:r[d]=mort(Number(r[a]),Number(r[b]),Number(r[0x20]));break;
        case 0x10:r[d]=prefixes.has(r[a]>>shift)?0n:1n;break;
        case 0x11:if(r[a]!==0n){pruned++;jump=pc+si}break;
        case 2:if(i!==L)throw Error("stride");r[d]=idx(Number(r[a]),Number(r[b]),Number(r[0x20]));break;
        case 0x20:r[d]=factors[i][Number(r[b])];break;
        case 0x22:if(mode!==1||i!==16)throw Error("precision");r[d]=mul(r[a],r[b]);break;
        case 0x30:{
          const p=local(r[a]);let sum=0n,n=0;
          for(let j=0;j<6;j++)if(i&(1<<j)){const q=p.slice();const ax=j>>1;q[ax]=(q[ax]+(j%2?1:-1)+side)%side;sum+=previous.get(mort(...q))??0n;n++}
          r[d]=n?sum/BigInt(n):0n;break;
        }
        case 0x21:if(i!==1)throw Error("codebook");r[d]=BigInt(encode(r[b]).token);break;
        case 0x40:{
          const m=r[0x40],v=r[a],packet=encode(v);
          if(BigInt(packet.token)!==r[b]||decode(packet)!==v||r[0x30]!==idx(...local(m)))throw Error("verification failure");
          if(next.has(m))throw Error("duplicate commit");
          next.set(m,v);tokens.set(m,packet);active++;break;
        }
        case 0x41:
          if(++r[d]===BigInt(side)){r[d]=0n;if(++r[a]===BigInt(side)){r[a]=0n;if(++r[b]===BigInt(side)){r[b]=0n;break}}}
          jump=pc+si;break;
        case 0xff:
          if(active+pruned!==side**3||steps!==4*side**3+10*active+1)throw Error("coverage");
          return {next,tokens,steps,active,pruned,log};
        default:throw Error("unsupported opcode");
      }
      if(trace&&log.length<128)log.push({pc:oldPC,op,nextPC:jump,xyz:[r[0],r[0x10],r[0x20]].map(Number),a0:r[0x60].toString(),m:r[0x40].toString()});
      pc=jump;
    }
  }
  return {pack,unpack,mort,unmort,idx,program,sweep,encode,decode,Q};
}
function selftest(){
  const m=isa64(),check=(p,msg)=>{if(!p)throw Error(msg)};
  const expected=["0104000100000000","1035040000000003","110005000000000b","0203000100002710","2016180000000000","2016280100000001","2016380200000002","2216061620000010","2216060630000010","301644060000003f","2216060640000010","2103140600000001","4010060310000000","410001020ffffff3","ff00000000000000"];
  check(m.program.every((w,i)=>w.toString(16).padStart(16,"0")===expected[i]),"golden bytecode");
  for(const w of m.program)check(m.pack(...m.unpack(w).slice(0,6))===w,"word roundtrip");
  check(m.idx(9999,9999,9999)===999999999999n,"last index");
  for(let i=0;i<1000;i++){const p=[i*997%10000,i*313%10000,i*761%10000],k=m.mort(...p);check(p.every((v,a)=>v===m.unmort(k,a)),"Morton inverse");const v=BigInt(i*71);check(m.decode(m.encode(v))===v,"residual inverse")}
  const side=2,keys=[];for(let z=0;z<side;z++)for(let y=0;y<side;y++)for(let x=0;x<side;x++)keys.push(m.mort(x,y,z));
  const previous=new Map(keys.map(k=>[k,3n*m.Q/4n])),full=new Set(keys),factors=[[m.Q/2n,m.Q/2n],[3n*m.Q/4n,3n*m.Q/4n],[m.Q/2n,m.Q/2n]];
  const first=m.sweep(side,previous,factors,full,true);
  check(first.active===8&&first.steps===113&&[...first.next.values()].every(v=>v===9216n),"full trace");
  const sparse=m.sweep(side,previous,factors,new Set([keys[0]]));
  check(sparse.active===1&&sparse.pruned===7&&sparse.steps===43&&sparse.next.size===1,"prune no stale write");
  const empty=m.sweep(side,previous,factors,new Set());
  check(empty.steps===33&&empty.next.size===0,"empty traversal");
  let state=previous,total=0;
  for(let i=0;i<1000;i++){const a=m.sweep(side,state,factors,full),b=m.sweep(side,state,factors,full);check(a.steps===b.steps&&[...a.next].every(([k,v])=>b.next.get(k)===v&&m.decode(a.tokens.get(k))===v),"deterministic sweep");state=a.next;total+=a.steps}
  check([...state.values()].every(v=>v===0n),"fixture convergence");
  return {wordRoundtrips:15,mortonRoundtrips:1000,residualRoundtrips:1000,verifiedSweeps:1000,replaySweeps:1000,primaryInstructions:total,firstCellRawQ16:first.next.get(keys[0]).toString(),sparseInstructions:sparse.steps,emptyInstructions:empty.steps,finalState:"zero fixed point (fixture only)"};
}
console.log(JSON.stringify(selftest(),null,2));
export {isa64,selftest};
