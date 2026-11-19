// Clinic3D 0.3 -- synthetic-only, unvalidated 3D bitwise clinical workflow.
// Reversible lattice transform is obfuscation, NOT cryptographic protection.
import { createHash, createCipheriv, createDecipheriv, hkdfSync, randomBytes } from 'node:crypto';

export const SIDE = 8;
export const OPCODES = Object.freeze({
  INGEST: 1, ENCODE_3D: 2, PERCEIVE: 3, SEAL: 4, OPEN: 5,
  REASON: 6, RETRIEVE: 7, GUARD: 8, EMIT: 9, HALT: 255
});
export const PROGRAM = Object.freeze([
  [1,0],[2,0],[3,0],[4,0],[5,0],[6,0],[4,1],
  [5,1],[7,0],[4,2],[5,2],[8,0],[9,0],[255,0]
]);
export const BYTECODE = Buffer.from(PROGRAM.flatMap(([op,arg]) => [op,arg,0,0]));
export const PROGRAM_SHA256 = sha(BYTECODE);
const SOURCE_INDEX = Object.freeze([
  {title:'South African Standard Treatment Guidelines and Essential Medicines List',
   url:'https://www.health.gov.za/nhi-edp-stgs-eml/',type:'source-index'},
  {title:'WHO SMART Guidelines',url:'https://smart.who.int/',type:'source-index'}
]);
const RULE_VERSION = 'synthetic-rules-0.3';
const CLINICAL_NOTICE = 'UNVALIDATED SOFTWARE SIMULATION. Not a diagnosis, triage tool, treatment plan, or medical service. Do not enter patient data.';
let enabled = true;
export function setEnabled(v) { enabled = !!v; return enabled; }
export function isEnabled() { return enabled; }
export function sha(data) { return createHash('sha256').update(data).digest('hex'); }
export function index3(x,y,z,n) { return x+n*y+n*n*z; }
export function encodeCube(bytes) {
  const b = Buffer.from(bytes);
  const n = Math.max(1,Math.ceil(Math.cbrt(b.length)));
  if (n > 64) throw new Error('Cube exceeds simulation limit');
  const cube = Buffer.alloc(n*n*n);
  for(let i=0;i<b.length;i++) {
    const x=i%n, y=Math.floor(i/n)%n, z=Math.floor(i/(n*n));
    const xx=(z+1)%n, yy=(x+2)%n, zz=(y+3)%n;
    cube[index3(xx,yy,zz,n)]=b[i];
  }
  return {side:n,length:b.length,cube};
}
export function decodeCube({side:n,length,cube}) {
  if (!Number.isInteger(n)||n<1||n>64||!Number.isInteger(length)||
      length<0||length>n*n*n||cube.length!==n*n*n) throw new Error('Invalid cube');
  const b=Buffer.alloc(length);
  for(let i=0;i<length;i++) {
    const x=i%n,y=Math.floor(i/n)%n,z=Math.floor(i/(n*n));
    b[i]=cube[index3((z+1)%n,(x+2)%n,(y+3)%n,n)];
  }
  return b;
}
export function voxelize(text) {
  const v=new Uint8Array(SIDE**3);
  const data=Buffer.from(String(text),'utf8');
  for(let i=0;i<data.length;i++) {
    const x=i%SIDE,y=Math.floor(i/SIDE)%SIDE,z=Math.floor(i/(SIDE*SIDE))%SIDE;
    v[index3(x,y,z,SIDE)]^=data[i];
  }
  // Discrete toroidal 3D stencil. Numerical feature demo; not a trained ANN.
  const out=new Uint8Array(v.length);
  for(let z=0;z<SIDE;z++) for(let y=0;y<SIDE;y++) for(let x=0;x<SIDE;x++) {
    const f=(a,b,c)=>v[index3((a+SIDE)%SIDE,(b+SIDE)%SIDE,(c+SIDE)%SIDE,SIDE)];
    const total=2*f(x,y,z)+f(x-1,y,z)+f(x+1,y,z)+f(x,y-1,z)+
        f(x,y+1,z)+f(x,y,z-1)+f(x,y,z+1);
    out[index3(x,y,z,SIDE)]=Math.floor(total/8);
  }
  return out;
}
export function deriveStageKey(root, stage) {
  if(!Buffer.isBuffer(root)||root.length!==32||![0,1,2].includes(stage))
    throw new Error('Bad stage key inputs');
  return Buffer.from(hkdfSync('sha256',root,Buffer.from('Clinic3D research v0.3'),
                    Buffer.from('stage:'+stage),32));
}
function aadFor(stage,previous,runId,programHash,side,length) {
  return Buffer.from(JSON.stringify({protocol:3,stage,previous,runId,programHash,side,length}));
}
export function seal(value,stage,previous,runId,root) {
  const plain=Buffer.from(JSON.stringify(value),'utf8');
  const folded=encodeCube(plain);
  const nonce=randomBytes(12), key=deriveStageKey(root,stage);
  const aad=aadFor(stage,previous,runId,PROGRAM_SHA256,folded.side,folded.length);
  const cipher=createCipheriv('aes-256-gcm',key,nonce,{authTagLength:16});
  cipher.setAAD(aad);
  const body=Buffer.concat([cipher.update(folded.cube),cipher.final()]);
  const packet={stage,previous,runId,side:folded.side,length:folded.length,
    nonce:nonce.toString('hex'),data:body.toString('base64'),
    tag:cipher.getAuthTag().toString('hex'),program:PROGRAM_SHA256};
  return packet;
}
export function openPacket(packet,stage,previous,runId,root) {
  if(packet.stage!==stage||packet.previous!==previous||packet.runId!==runId||
     packet.program!==PROGRAM_SHA256) throw new Error('Authenticated context mismatch');
  const nonce=Buffer.from(packet.nonce,'hex'),tag=Buffer.from(packet.tag,'hex');
  if(nonce.length!==12||tag.length!==16) throw new Error('Invalid nonce/tag');
  const decipher=createDecipheriv('aes-256-gcm',deriveStageKey(root,stage),
                                   nonce,{authTagLength:16});
  decipher.setAAD(aadFor(stage,previous,runId,PROGRAM_SHA256,packet.side,packet.length));
  decipher.setAuthTag(tag);
  const raw=Buffer.concat([decipher.update(Buffer.from(packet.data,'base64')),decipher.final()]);
  const decoded=decodeCube({side:packet.side,length:packet.length,cube:raw});
  return JSON.parse(decoded.toString('utf8'));
}
export function evaluateSynthetic(text,flags) {
  const s=text.toLowerCase();
  const signal=flags.some(x=>x==='synthetic_red_flag') ||
    /trouble swallowing|difficulty breathing|can't breathe/i.test(s);
  // Keyword fixture only; NEVER a validated red-flag or clearance model.
  return {classification:signal?'SIMULATION_RED_FLAG':'UNASSESSED_NOT_CLEARED',
    ruleVersion:RULE_VERSION,unvalidated:true};
}
export function validateInput(input) {
  if (!input||typeof input!=='object'||Array.isArray(input)) throw new Error('Object required');
  if(typeof input.text!=='string'||input.text.length>2000) throw new Error('Invalid text');
  if(!Array.isArray(input.flags)||input.flags.length>16||
      !input.flags.every(x=>typeof x==='string'&&x.length<80))
    throw new Error('Invalid flags');
  if(!['en','zu','st','af'].includes(input.language)) throw new Error('Invalid language');
  return {text:input.text,flags:input.flags,language:input.language};
}
export function run(input, options={}) {
  if(!enabled) throw new Error('Simulation disabled by administrator');
  const clean=validateInput(input);
  const instructions=Buffer.from(options.bytecode||BYTECODE);
  if(instructions.length!==BYTECODE.length||sha(instructions)!==PROGRAM_SHA256)
    throw new Error('Bytecode integrity failure');
  const root=options.root||randomBytes(32),runId=randomBytes(12).toString('hex');
  const trace=[],packets=[];
  let state=null,previous='GENESIS',pending=null;
  const issued=new Set();
  for(let pc=0;pc<instructions.length;pc+=4) {
    const op=instructions[pc],arg=instructions[pc+1];
    if(instructions[pc+2]!==0||instructions[pc+3]!==0||
       op!==BYTECODE[pc]||arg!==BYTECODE[pc+1]) throw new Error('Instruction mismatch');
    trace.push({pc:pc/4,opcode:Object.keys(OPCODES).find(k=>OPCODES[k]===op),argument:arg});
    switch(op) {
      case 1: state={input:clean};break;
      case 2: state.volume=Array.from(voxelize(clean.text));break;
      case 3: state.perception={reportedText:clean.text,flags:clean.flags,
                  provenance:'user-supplied-unverified',cubeSide:SIDE};break;
      case 4:
        if(pending) throw new Error('Unopened handoff');
        pending=seal(state,arg,previous,runId,root);break;
      case 5:
        if(!pending||issued.has(arg)) throw new Error('Replay or absent handoff');
        state=openPacket(pending,arg,previous,runId,root);
        issued.add(arg);previous=sha(Buffer.from(JSON.stringify(pending)));
        packets.push({stage:arg,digest:previous,tagBytes:16,integrity:true});
        pending=null;break;
      case 6: state.reasoning=evaluateSynthetic(clean.text,clean.flags);break;
      case 7: state.evidence=SOURCE_INDEX;break;
      case 8:
        if(issued.size!==3||pending||!state.reasoning?.unvalidated)
          throw new Error('Clinical safety gate blocked');
        state.notice=CLINICAL_NOTICE;state.orders=[];break;
      case 9:
        return {simulation:true,runId,classification:state.reasoning.classification,
          notice:state.notice,evidence:state.evidence,orders:state.orders,
          cube:{side:SIDE,voxels:state.volume},stageProofs:packets,
          bytecode:{sha256:PROGRAM_SHA256,wordBits:32,instructionCount:PROGRAM.length},
          trace};
      case 255: return state;
      default: throw new Error('Unknown operation');
    }
  }
  throw new Error('No EMIT instruction');
}
