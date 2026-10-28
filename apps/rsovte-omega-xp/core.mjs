// RSOVTE-Ω/XP: bounded, deterministic, sparse 3D geometric emulator.
// Virtual G^3 denotes address capacity, NOT allocated cells or learned intelligence.
export const G = 1_000_000;
export const BLOCK = 1024;
export const CELLS = Math.ceil(G / BLOCK); // 977 including the partial terminal cell
export const EPS = 1e-7;
export const MAX_POINTS = 200_000;
const clamp = (x,a,b) => Math.max(a,Math.min(b,x));

function randomGenerator(seed) {
  if (!Number.isInteger(seed) || seed < 0 || seed > 0xffffffff) throw new RangeError('invalid seed');
  let s = (seed >>> 0) || 1;
  return () => {
    s ^= s << 13; s ^= s >>> 17; s ^= s << 5;
    return (s >>> 0) / 4294967296;
  };
}
function assertCoords(coords, allowEmpty=false) {
  if (!(coords instanceof Uint32Array) || coords.length % 3 ||
      coords.length / 3 > MAX_POINTS || (!allowEmpty && coords.length === 0)) {
    throw new RangeError('coordinates must be Uint32Array of 1..200000 xyz triples');
  }
  for (let i=0;i<coords.length;i++) if (coords[i] >= G) throw new RangeError('coordinate outside 3D domain');
}
export function generateCoordinates(count=20_000, seed=42) {
  if (!Number.isInteger(count) || count < 1 || count > MAX_POINTS) throw new RangeError('count outside bound');
  const rng=randomGenerator(seed), coords=new Uint32Array(3*count);
  for (let j=0;j<coords.length;j+=3) {
    coords[j]=Math.floor(clamp((.2+rng()*.6+(rng()-.5)*.1)*G,0,G-1));
    coords[j+1]=Math.floor(clamp((.2+rng()*.6+(rng()-.5)*.1)*G,0,G-1));
    coords[j+2]=Math.floor((.15+rng()*.7)*G);
  }
  return coords;
}
export function hashBytes(bytes) {
  let h=2166136261>>>0;
  for (const byte of bytes) h=Math.imul(h^byte,16777619)>>>0;
  return h>>>0; // non-cryptographic FNV-1a, only accidental-corruption detection
}
export function encodeCoordinates(coords) {
  assertCoords(coords);
  const n=coords.length/3, b=new Uint8Array(24+8*n), dv=new DataView(b.buffer);
  b.set([82,83,51,68]); // RS3D
  dv.setUint16(4,1,true); dv.setUint16(6,24,true);
  dv.setUint32(8,n,true); dv.setUint32(12,G,true); dv.setUint32(16,BLOCK,true);
  for(let i=0,j=0;i<n;i++,j+=3) {
    const x=coords[j],y=coords[j+1],z=coords[j+2];
    dv.setUint32(24+8*i,(x | (y<<20))>>>0,true);
    dv.setUint32(28+8*i,((y>>>12) | (z<<8))>>>0,true);
  }
  dv.setUint32(20,hashBytes(b.subarray(24)),true);
  return b;
}
export function decodeCoordinates(input) {
  const b=input instanceof ArrayBuffer ? new Uint8Array(input) : input;
  if (!(b instanceof Uint8Array) || b.byteLength < 24) throw new Error('truncated RS3D header');
  const dv=new DataView(b.buffer,b.byteOffset,b.byteLength);
  if (b[0]!==82 || b[1]!==83 || b[2]!==51 || b[3]!==68) throw new Error('invalid RS3D magic');
  if (dv.getUint16(4,true)!==1 || dv.getUint16(6,true)!==24) throw new Error('unsupported codec version/header');
  const n=dv.getUint32(8,true);
  if (!n || n > MAX_POINTS || b.byteLength!==24+8*n) throw new Error('invalid count or packet length');
  if (dv.getUint32(12,true)!==G || dv.getUint32(16,true)!==BLOCK) throw new Error('domain/block mismatch');
  if (dv.getUint32(20,true)!==hashBytes(b.subarray(24))) throw new Error('payload checksum mismatch');
  const coords=new Uint32Array(3*n);
  for (let i=0,j=0;i<n;i++,j+=3) {
    const lo=dv.getUint32(24+8*i,true), hi=dv.getUint32(28+8*i,true);
    if ((hi>>>28)!==0) throw new Error('nonzero reserved bits');
    const x=lo&0xfffff,y=(lo>>>20)|((hi&255)<<12),z=(hi>>>8)&0xfffff;
    if (x>=G||y>=G||z>=G) throw new Error('decoded coordinate outside domain');
    coords[j]=x;coords[j+1]=y;coords[j+2]=z;
  }
  return coords;
}
export function verifyCodec(coords) {
  const b=encodeCoordinates(coords), decoded=decodeCoordinates(b);
  let mismatches=0;
  for(let i=0;i<coords.length;i++) if(coords[i]!==decoded[i]) mismatches++;
  return {pass:mismatches===0,mismatches,points:coords.length/3,packetBytes:b.byteLength,
    payloadFNV:new DataView(b.buffer).getUint32(20,true).toString(16).padStart(8,'0').toUpperCase()};
}
export function initialState(coords) {
  assertCoords(coords);
  const count=coords.length/3,positions=new Float64Array(coords),
    target=new Float64Array(coords.length),velocity=new Float64Array(coords.length),
    active=new Uint8Array(count),occupied=new Set();
  active.fill(1);
  for(let j=0;j<coords.length;j+=3){
    const cx=Math.floor(coords[j]/BLOCK),cy=Math.floor(coords[j+1]/BLOCK),cz=Math.floor(coords[j+2]/BLOCK);
    occupied.add((cx*CELLS+cy)*CELLS+cz);
    for(let a=0;a<3;a++) target[j+a]=Math.min(G-1,(Math.floor(coords[j+a]/BLOCK)+.5)*BLOCK);
  }
  return {count,source:coords.slice(),positions,target,velocity,active,occupiedCells:occupied.size,
    activeCount:count,iterations:0,updates:0,maxSnapError:0,history:[]};
}
export function spectralRadius(q,beta,weights) {
  let rho=0;
  for(const w of weights) {
    const trace=1-q*w+beta, disc=trace*trace-4*beta;
    const radius=disc>=0?Math.max(Math.abs((trace+Math.sqrt(disc))/2),
      Math.abs((trace-Math.sqrt(disc))/2)):Math.sqrt(beta);
    rho=Math.max(rho,radius);
  }
  return rho;
}
function validWeights(w) {
  return Array.isArray(w)&&w.length===3&&w.every(x=>Number.isFinite(x)&&x>=.1&&x<=2);
}
export function optimizeParameters(weights=[1,.8,1.25]) {
  if(!validWeights(weights)) throw new RangeError('weights must contain three numbers in [0.1, 2]');
  let best={q:.65,beta:.12,rho:spectralRadius(.65,.12,weights),candidates:0};
  for(let qi=20;qi<=155;qi++) for(let bi=0;bi<=40;bi++) {
    const q=qi/100,beta=bi/200,rho=spectralRadius(q,beta,weights);
    best.candidates++;
    if(rho<1 && rho<best.rho-1e-12) best={...best,q,beta,rho};
  }
  return best;
}
export function step(state, weights, parameters, epsilon=EPS) {
  if(!validWeights(weights)) throw new RangeError('invalid weights');
  if(!(epsilon>0 && Number.isFinite(epsilon))) throw new RangeError('invalid tolerance');
  const {q,beta}=parameters;
  if(!(Number.isFinite(q)&&Number.isFinite(beta)&&beta>=0&&beta<1 &&
      spectralRadius(q,beta,weights)<1)) throw new RangeError('unstable or invalid parameters');
  let updated=0,settled=0,maxResidual=0;
  const {positions:p,target:t,velocity:v,active, count}=state;
  for(let i=0,j=0;i<count;i++,j+=3) {
    if(!active[i]) continue;
    let localResidual=0,maxVelocity=0;
    for(let a=0;a<3;a++) {
      const k=j+a,dx=q*weights[a]*(t[k]-p[k])+beta*v[k];
      p[k]+=dx;v[k]=dx;
      localResidual=Math.max(localResidual,Math.abs(t[k]-p[k]));
      maxVelocity=Math.max(maxVelocity,Math.abs(dx));
    }
    updated++;
    if(localResidual<=epsilon && maxVelocity<=epsilon) {
      state.maxSnapError=Math.max(state.maxSnapError,localResidual);
      for(let a=0;a<3;a++){p[j+a]=t[j+a];v[j+a]=0;}
      active[i]=0;state.activeCount--;settled++;localResidual=0;
    }
    maxResidual=Math.max(maxResidual,localResidual);
  }
  state.iterations++;state.updates+=updated;
  return {iteration:state.iterations,updated,settled,active:state.activeCount,maxResidual};
}
export function verify(state, tolerance=1e-5) {
  let meanError=0,maxError=0,nonFinite=0,wrongCell=0,outOfBounds=0;
  for(let i=0,j=0;i<state.count;i++,j+=3) {
    let localError=0;
    for(let a=0;a<3;a++) {
      const k=j+a,p=state.positions[k];
      if(!Number.isFinite(p)){nonFinite++;continue;}
      localError=Math.max(localError,Math.abs(p-state.target[k]));
      if(Math.floor(p/BLOCK)!==Math.floor(state.source[k]/BLOCK))wrongCell++;
      if(p<0 || p>=G)outOfBounds++;
    }
    meanError+=localError;maxError=Math.max(maxError,localError);
  }
  meanError/=state.count;
  return {pass:nonFinite===0&&wrongCell===0&&outOfBounds===0&&maxError<=tolerance,
    meanError,maxError,nonFinite,wrongCell,outOfBounds,active:state.activeCount,
    exactFixedPoint:state.activeCount===0&&maxError===0,maxSnapError:state.maxSnapError};
}
export function digest(state) {
  let h=2166136261>>>0;
  for(const p of state.positions) {
    const q=Math.round(p*1e5),lo=q>>>0,hi=Math.floor(q/4294967296)>>>0;
    h=Math.imul(h^lo,16777619)>>>0;
    h=Math.imul(h^hi,16777619)>>>0;
  }
  return h.toString(16).padStart(8,'0').toUpperCase(); // noncryptographic
}
export function simulate(coords,{iterations=1000,weights=[1,.8,1.25],tuned=true}={}) {
  if(!Number.isInteger(iterations)||iterations<1||iterations>10000) throw new RangeError('iterations 1..10000');
  if(!validWeights(weights)) throw new RangeError('invalid weights');
  const packet=encodeCoordinates(coords),reconstructed=decodeCoordinates(packet);
  const state=initialState(reconstructed);
  const tuning=tuned?optimizeParameters(weights):
    {q:.65,beta:.12,rho:spectralRadius(.65,.12,weights),candidates:0};
  if(tuning.rho>=1) throw new Error('default dynamics unstable for selected weights');
  for(let i=0;i<iterations;i++) {
    const receipt=step(state,weights,tuning);
    if(i===0 || (i+1)%10===0 || i===iterations-1)state.history.push(receipt);
  }
  const codec=verifyCodec(coords),geometry=verify(state);
  return {state,report:{
    engine:'RSOVTE-Ω/XP',domain:G,foldBlock:BLOCK,foldCellsPerAxis:CELLS,
    particleCount:coords.length/3,iterations,stateDigest:digest(state),
    actualPointUpdates:state.updates,prunedUpdates:iterations*state.count-state.updates,
    codec,geometry,tuning,
    history:state.history,passed:codec.pass&&geometry.pass}};
}
