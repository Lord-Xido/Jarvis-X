/* DM3D Septillion³ portable codec kernel: browser/Node, no dependencies.
   Structured virtual 3D field, 3D 2x average pooling, periodic latent folding,
   modular exact residual decoding. This is NOT a pretrained neural network. */
const DM3D = (() => {
  const SIDE = 10n ** 24n;
  const TILE = 8, LATENT = 4, N_TILES = SIDE / 8n;
  const MASK = (1n << 64n) - 1n;
  function hash64(text) {
    let h = 0xcbf29ce484222325n;
    for (let i=0;i<text.length;i++) h = ((h ^ BigInt(text.charCodeAt(i))) * 0x100000001b3n) & MASK;
    return h;
  }
  function splitmix64(v) {
    v = (v + 0x9e3779b97f4a7c15n) & MASK;
    v = ((v ^ (v >> 30n)) * 0xbf58476d1ce4e5b9n) & MASK;
    v = ((v ^ (v >> 27n)) * 0x94d049bb133111ebn) & MASK;
    return (v ^ (v >> 31n)) & MASK;
  }
  function readPosition(values) {
    const xyz = values.map(v => BigInt(String(v).trim()));
    if (xyz.some(v => v<0n || v>=SIDE)) throw new RangeError('Coordinates must satisfy 0 ≤ x,y,z < 10²⁴');
    return xyz;
  }
  function getTile(q, seed=7) {
    if (q.some(v => v<0n || v>=N_TILES)) throw new RangeError('Tile index out of bounds');
    const h=hash64(`${seed}|${q[0]}|${q[1]}|${q[2]}`);
    const bases=[8n,16n,24n].map(s => 64+Number((h>>s)&127n));
    const buf=new Uint8Array(8*8*8*3);
    for (let z=0;z<8;z++) for (let y=0;y<8;y++) for (let x=0;x<8;x++) {
      const index=x+8*(y+8*z);
      const g=[Math.floor((3*x+2*y-2*z)/2)-5,
               Math.floor((-2*x+3*y+2*z)/2)-5,
               Math.floor((2*x-2*y+3*z)/2)-5];
      for(let c=0;c<3;c++) {
        const jitter=Number(splitmix64(h ^ BigInt(index*17+c*1337)) % 5n)-2;
        buf[index*3+c]=Math.max(0,Math.min(255,bases[c]+g[c]+jitter));
      }
    }
    return buf;
  }
  const srcIdx=(z,y,x,c)=>((z*8+y)*8+x)*3+c;
  const latIdx=(z,y,x,c)=>((z*4+y)*4+x)*3+c;
  function encode(input) {
    if(input.length !== 1536) throw new RangeError('Expected 1536 RGB8 bytes');
    const out = new Uint8Array(192);
    for(let z=0;z<4;z++)for(let y=0;y<4;y++)for(let x=0;x<4;x++)for(let c=0;c<3;c++){
      let s=0;
      for(let dz=0;dz<2;dz++)for(let dy=0;dy<2;dy++)for(let dx=0;dx<2;dx++)
        s+=input[srcIdx(z*2+dz,y*2+dy,x*2+dx,c)];
      out[latIdx(z,y,x,c)]=Math.floor(s/8+0.5);
    }
    return out;
  }
  function fold(latent, alpha=0.12, steps=2) {
    if(!(alpha>=0&&alpha<=1 && steps>=0 && steps<=100 && Number.isInteger(steps))) throw new RangeError('Bad fold parameters');
    let a=new Uint8Array(latent);
    for(let k=0;k<steps;k++) {
      const b = new Uint8Array(192);
      for(let z=0;z<4;z++)for(let y=0;y<4;y++)for(let x=0;x<4;x++)for(let c=0;c<3;c++) {
        const n=a[latIdx((z+3)%4,y,x,c)] + a[latIdx((z+1)%4,y,x,c)]
          +a[latIdx(z,(y+3)%4,x,c)] + a[latIdx(z,(y+1)%4,x,c)]
          +a[latIdx(z,y,(x+3)%4,c)] + a[latIdx(z,y,(x+1)%4,c)];
        const v=(1-alpha)*a[latIdx(z,y,x,c)]+(alpha/6)*n;
        b[latIdx(z,y,x,c)]=Math.max(0,Math.min(255,Math.floor(v+0.5)));
      }
      a=b;
    }
    return a;
  }
  function decodeApprox(latent) {
    const out=new Uint8Array(1536);
    for(let z=0;z<8;z++)for(let y=0;y<8;y++)for(let x=0;x<8;x++)for(let c=0;c<3;c++)
      out[srcIdx(z,y,x,c)]=latent[latIdx(z>>1,y>>1,x>>1,c)];
    return out;
  }
  function reconstruct(orig,approx) {
    const residual=new Uint8Array(1536),exact=new Uint8Array(1536);
    let mse=0,identical=true;
    for(let i=0;i<1536;i++) {
      residual[i]=(orig[i]-approx[i]+256)&255;
      exact[i]=(approx[i]+residual[i])&255;
      mse+=(orig[i]-approx[i])**2;
      if(exact[i] !== orig[i]) identical=false;
    }
    mse/=1536;
    return {residual,exact,identical,mse,psnr:mse===0?Infinity:10*Math.log10(255*255/mse)};
  }
  function processTile(q,alpha=.12,steps=2,seed=7) {
    const orig=getTile(q,seed),z=encode(orig),zf=fold(z,alpha,steps),approx=decodeApprox(zf);
    return {q,orig,z,zf,approx,...reconstruct(orig,approx)};
  }
  function loss(q,alpha,steps=2,seed=7) {return processTile(q,alpha,steps,seed).mse;}
  return {SIDE,TILE,LATENT,N_TILES,hash64,splitmix64,readPosition,getTile,
          encode,fold,decodeApprox,reconstruct,processTile,loss};
})();
if (typeof module !== 'undefined' && module.exports) module.exports = DM3D;
