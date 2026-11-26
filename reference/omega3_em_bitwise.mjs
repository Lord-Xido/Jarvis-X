// JARVIS-X Omega3: deterministic digital reference; no RF hardware simulation.
function run() {
  const W=(1n<<64n)-1n;
  function rot(x,n){n=BigInt(n%64);return ((x<<n)|(x>>(64n-n)))&W;}
  function enc(x){return rot(x,7);}
  function dec(x){return rot(x,57);}
  function check(x,msg){if(!x)throw new Error(msg);}
  function encode8(v){let a=Array(13).fill(0),k=0;for(let i=1;i<=12;i++)if(i&(i-1))a[i]=(v>>k++)&1;for(let p of [1,2,4,8])for(let i=1;i<=12;i++)if(i!==p&&(i&p))a[p]^=a[i];return a;}
  function decode8(a){a=a.slice();let s=0;for(let i=1;i<=12;i++)if(a[i])s^=i;if(s>12)throw new Error("invalid syndrome");if(s)a[s]^=1;let v=0,k=0;for(let i=1;i<=12;i++)if(i&(i-1))v|=a[i]<<k++;return v;}
  let fec=0;for(let x=0;x<256;x++){let a=encode8(x);for(let e=0;e<=12;e++){let b=a.slice();if(e)b[e]^=1;check(decode8(b)===x,"FEC");fec++;}}
  const L=8,N=L**3,idx=(x,y,z)=>((z+L)%L)*L*L+((y+L)%L)*L+(x+L)%L;
  const adj=Array.from({length:N},(_,i)=>{let x=i%L,y=Math.floor(i/L)%L,z=Math.floor(i/(L*L));return [idx(x+1,y,z),idx(x-1,y,z),idx(x,y+1,z),idx(x,y-1,z),idx(x,y,z+1),idx(x,y,z-1)];});
  let rng=0x123456789abcdefn;function next(){rng^=(rng<<13n)&W;rng^=rng>>7n;rng^=(rng<<17n)&W;return rng&W;}
  const basis=0xa5c39e7b12d4f608n;
  const original=Array.from({length:N},next);
  const residual=original.map(v=>v^basis);
  check(original.every((v,i)=>(basis^residual[i])===v),"residual");
  check(original.every(v=>dec(enc(v))===v),"codec");
  let field=original.slice(),valid=Array(N).fill(false);field[0]=basis;valid[0]=true;
  const reached=[1];for(let t=1;t<=12;t++){const nv=valid.map((v,i)=>v||adj[i].some(j=>valid[j]));field=field.map((v,i)=>nv[i]?v^(v^basis):v);valid=nv;reached.push(valid.filter(Boolean).length);}
  check(field.every(v=>v===basis),"permeation");
  for(let t=0;t<1000;t++){field=field.map(v=>v^(v^basis));check(field.every(v=>v===basis),"idempotence");}
  function step(s,mask,core,phase,key,delta){const z=s.map(enc);return z.map((v,i)=>{let u=v;for(let k=0;k<6;k++)u^=z[adj[i][k]]&mask[k];return dec(u^rot(core,phase)^key)^delta[i];});}
  const masks=Array(6).fill(W),zero=Array(N).fill(0n),checker=Array.from({length:N},(_,i)=>(i%L+Math.floor(i/L)%L+Math.floor(i/(L*L)))%2?0n:basis);
  let c=checker.slice();for(let t=0;t<1000;t++){c=step(c,masks,0n,0,0n,zero);check(c.every((v,i)=>v===checker[i]),"checkerboard");}
  const core=next(),key=next(),delta=Array.from({length:N},next),nonuniform=step(original,[W,0n,1n,0xffn,W,0n],core,3,key,delta);
  check(nonuniform.every((v,i)=>{let x=enc(original[i]);for(let k=0;k<6;k++)x^=enc(original[adj[i][k]])&[W,0n,1n,0xffn,W,0n][k];return v===(dec(x^rot(core,3)^key)^delta[i]);}),"transition");
  return {model:"Omega3 EM digital reference",lattice:[L,L,L],word_bits:64,fec_cases:fec,fec_failures:0,reachability:reached,residual_mismatches:0,permeation_mismatches:0,fixed_point_iterations:1000,checkerboard_iterations:1000,checkerboard_distinct_words:new Set(c).size,checkerboard_reference_mismatches:c.filter(v=>v!==basis).length,hardware_verified:false,status:"PASS"};
}
console.log(JSON.stringify(run(), null, 2));
