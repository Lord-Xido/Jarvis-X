import http from 'node:http';
import fs from 'node:fs';
import fsp from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import crypto from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { spawn } from 'node:child_process';

const ROOT = path.dirname(fileURLToPath(import.meta.url));
const PUBLIC = path.join(ROOT, 'public');
const DATA = path.resolve(process.env.DATA_DIR || path.join(ROOT, 'data'));
const PORT = Number(process.env.PORT || 3000);
const KEY = process.env.OPENAI_API_KEY || '';
const API_BASE = (process.env.OPENAI_BASE_URL || 'https://api.openai.com/v1').replace(/\/$/, '');
const ACCESS = process.env.APP_ACCESS_TOKEN || '';
const TEXT_MODEL = process.env.TEXT_MODEL || 'gpt-4.1-mini';
const IMAGE_MODEL = process.env.IMAGE_MODEL || 'gpt-image-1.5';
const TTS_MODEL = process.env.TTS_MODEL || 'gpt-4o-mini-tts';
const AUDIO_MODEL = process.env.TRANSCRIBE_MODEL || 'gpt-4o-mini-transcribe';
const LIMIT = 16 * 1024 * 1024;
let requests = 0, errors = 0;
const counters = new Map();
let sessionWrite = Promise.resolve();

function respond(res, status, data) {
  res.writeHead(status, {'Content-Type':'application/json; charset=utf-8','Cache-Control':'no-store','X-Content-Type-Options':'nosniff'});
  res.end(JSON.stringify(data));
}
function fail(res, status, message) { respond(res, status, {error:message}); }
function clampText(v, max=8000) {return typeof v==='string' ? v.slice(0,max) : '';}
function auth(req, res) {
  if (!ACCESS) return true;
  const token = req.headers['x-app-token'];
  const a = Buffer.from(ACCESS), b = Buffer.from(typeof token === 'string' ? token : '');
  if (a.length !== b.length || !crypto.timingSafeEqual(a,b)) {fail(res,401,'Access token required.');return false;}
  return true;
}
function rate(req,res) {
  const ip=req.socket.remoteAddress || 'unknown', now=Date.now();
  const cur=counters.get(ip);
  const next=(!cur || now-cur.start>60000)?{start:now,n:1}:{start:cur.start,n:cur.n+1};
  counters.set(ip,next);
  if (counters.size>4096) for(const [k,v] of counters) {if(now-v.start>60000)counters.delete(k);}
  if(next.n>80){fail(res,429,'Rate limit exceeded.');return false;}
  return true;
}
async function jsonBody(req) {
  const items=[];let bytes=0;
  for await(const item of req) {
    bytes+=item.length;
    if(bytes>LIMIT)throw Object.assign(new Error('Request exceeds 16 MiB'),{status:413});
    items.push(item);
  }
  try{return JSON.parse(Buffer.concat(items).toString('utf8')||'{}');}
  catch {throw Object.assign(new Error('Malformed JSON'),{status:400});}
}
async function cloud(endpoint, payload, options={}) {
  if(!KEY)throw Object.assign(new Error('OPENAI_API_KEY is not configured on the server.'),{status:503});
  const headers={Authorization:'Bearer '+KEY};
  if(!options.form)headers['Content-Type']='application/json';
  let r;
  try {
    r=await fetch(API_BASE+endpoint,{method:'POST',headers,body:options.form?payload:JSON.stringify(payload),signal:AbortSignal.timeout(120000)});
  } catch(e) {throw Object.assign(new Error('Cloud transport error: '+e.message),{status:502});}
  if(!r.ok) {
    let detail='';try{const body=await r.json();detail=body.error?.message||'Upstream error';}
    catch{detail=(await r.text()).slice(0,250);}
    throw Object.assign(new Error('Cloud API '+r.status+': '+detail),{status:r.status>=500?502:r.status});
  }
  return options.binary?Buffer.from(await r.arrayBuffer()):r.json();
}
export function parseMessage(d) {
  const out=[];
  for(const m of (Array.isArray(d.history)?d.history:[]).slice(-18)) {
    if(!['user','assistant'].includes(m?.role))continue;
    const text=clampText(m.text,4000);
    if(text)out.push({role:m.role,content:[{type:m.role==='user'?'input_text':'output_text',text}]});
  }
  const content=[];
  if(clampText(d.prompt).trim())content.push({type:'input_text',text:clampText(d.prompt)});
  for(const a of (Array.isArray(d.attachments)?d.attachments:[]).slice(0,4)) {
    if(typeof a?.data!=='string' || typeof a?.type!=='string')continue;
    if(/^image\/(png|jpeg|webp|gif)$/.test(a.type) && a.data.startsWith('data:'+a.type+';base64,')) {
      content.push({type:'input_image',image_url:a.data,detail:'auto'});
    } else if(a.type==='application/pdf' && a.data.startsWith('data:application/pdf;base64,')) {
      content.push({type:'input_file',filename:clampText(a.name||'attachment.pdf',80),file_data:a.data});
    } else if(a.type==='text/plain' && a.data.startsWith('data:text/plain;base64,')) {
      const bytes=Buffer.from(a.data.split(',')[1],'base64');
      content.push({type:'input_text',text:'File '+clampText(a.name||'document.txt',80)+':\n'+bytes.toString('utf8').slice(0,18000)});
    }
  }
  if(!content.length)throw Object.assign(new Error('Send text or a supported attachment.'),{status:400});
  out.push({role:'user',content});return out;
}
export function flattenText(result) {
  if(typeof result.output_text==='string' && result.output_text)return result.output_text;
  return (result.output||[]).filter(x=>x.type==='message').flatMap(x=>x.content||[])
    .filter(x=>x.type==='output_text').map(x=>x.text||'').join('\n') || '[Model returned no text]';
}
const sessionsFile=path.join(DATA,'sessions.json');
async function sessions() {try{return JSON.parse(await fsp.readFile(sessionsFile,'utf8'));}catch{return {};}}
async function saveSessions(all) {
  const op=async()=> {
    await fsp.mkdir(DATA,{recursive:true});
    const tmp=sessionsFile+'.'+crypto.randomUUID()+'.tmp';
    await fsp.writeFile(tmp,JSON.stringify(all));
    await fsp.rename(tmp,sessionsFile);
  };
  sessionWrite=sessionWrite.then(op,op);
  return sessionWrite;
}
function safeID(id){return typeof id==='string' && /^[a-zA-Z0-9_-]{8,80}$/.test(id);}
export async function makeVideo(prompt, image) {
  const tmp=await fsp.mkdtemp(path.join(os.tmpdir(),'annos-'));
  try {
    const caption=path.join(tmp,'caption.txt');
    await fsp.writeFile(caption,clampText(prompt,90).replace(/[\r\n]/g,' '));
    const dest=path.join(tmp,'clip.mp4');
    const args=['-hide_banner','-loglevel','error'];
    const font='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf';
    const captionFilter='drawtext=fontfile='+font+':textfile='+caption+':fontcolor=white:fontsize=28:borderw=2:bordercolor=black:x=(w-text_w)/2:y=h-75';
    if(typeof image==='string' && /^data:image\/(png|jpeg|webp);base64,/.test(image)) {
      if(image.length>12*1024*1024)throw Object.assign(new Error('Source image too large'),{status:413});
      const match=image.match(/^data:image\/(png|jpeg|webp);base64,(.+)$/s);
      if(!match)throw Object.assign(new Error('Invalid source image'),{status:400});
      const file=path.join(tmp,match[1]==='jpeg'?'still.jpg':'still.'+match[1]);
      await fsp.writeFile(file,Buffer.from(match[2],'base64'));
      args.push('-loop','1','-framerate','24','-i',file,'-vf','scale=1100:620:force_original_aspect_ratio=increase,crop=960:540,zoompan=z=min(zoom+0.001\\,1.12):d=1:s=960x540:fps=24,'+captionFilter);
    } else {
      args.push('-f','lavfi','-i','mandelbrot=s=960x540:r=24:maxiter=64','-vf','hue=h=25*t,eq=contrast=1.1:brightness=-0.04,'+captionFilter);
    }
    args.push('-t','6','-an','-c:v','libx264','-preset','ultrafast','-crf','27','-pix_fmt','yuv420p','-movflags','+faststart','-y',dest);
    await new Promise((resolve,reject)=>{
      const p=spawn('ffmpeg',args,{stdio:['ignore','ignore','pipe']});
      let detail='';
      const timer=setTimeout(()=>p.kill('SIGKILL'),55000);
      p.stderr.on('data',c=>detail=(detail+c.toString()).slice(-1000));
      p.on('error',e=>{clearTimeout(timer);reject(e);});
      p.on('close',code=>{clearTimeout(timer);code===0?resolve():reject(new Error('FFmpeg render failure: '+(detail||code)));});
    });
    return await fsp.readFile(dest);
  } finally {await fsp.rm(tmp,{recursive:true,force:true});}
}
const MIME={'.html':'text/html; charset=utf-8','.css':'text/css; charset=utf-8','.js':'text/javascript; charset=utf-8','.png':'image/png','.json':'application/json'};
async function route(req,res) {
  const url=new URL(req.url,'http://localhost');
  if(url.pathname==='/api/status' && req.method==='GET')
    return respond(res,200,{online:!!KEY,videoRenderer:true,accessProtected:!!ACCESS,models:{text:TEXT_MODEL,image:IMAGE_MODEL,tts:TTS_MODEL,transcription:AUDIO_MODEL},requests,errors});
  if(url.pathname.startsWith('/api/')) {
    if(!auth(req,res) || !rate(req,res))return;
    requests++;
    try {
      if(url.pathname==='/api/chat' && req.method==='POST') {
        const d=await jsonBody(req),input=parseMessage(d),start=Date.now();
        const result=await cloud('/responses',{model:TEXT_MODEL,instructions:'You are ANN OS multimodal assistant. Distinguish simulations from actual results. Do not invent tool execution or performance.',input,max_output_tokens:1800});
        return respond(res,200,{text:flattenText(result),model:result.model||TEXT_MODEL,tokens:result.usage||null,latencyMs:Date.now()-start});
      }
      if(url.pathname==='/api/image' && req.method==='POST') {
        const d=await jsonBody(req),prompt=clampText(d.prompt,3500).trim();
        if(!prompt)throw Object.assign(new Error('Prompt required'),{status:400});
        const result=await cloud('/images/generations',{model:IMAGE_MODEL,prompt,size:'1024x1024',n:1,quality:'medium'});
        const item=result.data?.[0];
        if(!item?.b64_json && !item?.url)throw new Error('Image provider returned no image');
        return respond(res,200,{image:item.b64_json?'data:image/png;base64,'+item.b64_json:item.url,model:IMAGE_MODEL});
      }
      if(url.pathname==='/api/speech' && req.method==='POST') {
        const d=await jsonBody(req),input=clampText(d.text,4096).trim();
        if(!input)throw Object.assign(new Error('Text required'),{status:400});
        const voice=['alloy','ash','coral','echo','fable','nova','onyx','sage','shimmer'].includes(d.voice)?d.voice:'alloy';
        const buf=await cloud('/audio/speech',{model:TTS_MODEL,input,voice,response_format:'mp3'},{binary:true});
        res.writeHead(200,{'Content-Type':'audio/mpeg','Cache-Control':'no-store','Content-Disposition':'attachment; filename="ann-os-speech.mp3"'});
        return res.end(buf);
      }
      if(url.pathname==='/api/transcribe' && req.method==='POST') {
        const d=await jsonBody(req),data=clampText(d.data,13_000_000);
        const match=data.match(/^data:(audio\/[a-z0-9+.-]+)(?:;[^,;]+)*;base64,(.+)$/s);
        if(!match)throw Object.assign(new Error('Expected base64 audio data URL'),{status:400});
        const bytes=Buffer.from(match[2],'base64');
        if(bytes.length>8*1024*1024)throw Object.assign(new Error('Recording exceeds 8 MiB'),{status:413});
        const ext=match[1].includes('webm')?'webm':match[1].includes('mp4')?'mp4':match[1].includes('ogg')?'ogg':'wav';
        const form=new FormData();form.set('model',AUDIO_MODEL);form.set('file',new Blob([bytes],{type:match[1]}),'recording.'+ext);
        const result=await cloud('/audio/transcriptions',form,{form:true});
        return respond(res,200,{text:result.text||''});
      }
      if(url.pathname==='/api/video' && req.method==='POST') {
        const d=await jsonBody(req),prompt=clampText(d.prompt,300).trim();
        if(!prompt)throw Object.assign(new Error('Video prompt required'),{status:400});
        const buf=await makeVideo(prompt,d.image);
        res.writeHead(200,{'Content-Type':'video/mp4','Cache-Control':'no-store','Content-Disposition':'attachment; filename="ann-os-motion.mp4"'});
        return res.end(buf);
      }
      if(url.pathname==='/api/sessions') {
        if(req.method==='GET')return respond(res,200,{sessions:Object.values(await sessions())});
        if(req.method==='POST') {
          const d=await jsonBody(req);
          if(!safeID(d.id))throw Object.assign(new Error('Invalid session ID'),{status:400});
          const all=await sessions();
          all[d.id]={id:d.id,title:clampText(d.title,80)||'New conversation',updated:Date.now(),
            messages:(Array.isArray(d.messages)?d.messages:[]).slice(-50).map(m=>({role:m.role==='assistant'?'assistant':'user',text:clampText(m.text,12000),time:clampText(m.time,50),kind:clampText(m.kind,30)}))};
          const latest=Object.values(all).sort((a,b)=>b.updated-a.updated).slice(0,80);
          await saveSessions(Object.fromEntries(latest.map(x=>[x.id,x])));
          return respond(res,200,{saved:true});
        }
        if(req.method==='DELETE') {
          const id=url.searchParams.get('id');
          if(!safeID(id))throw Object.assign(new Error('Invalid session ID'),{status:400});
          const all=await sessions();delete all[id];await saveSessions(all);return respond(res,200,{deleted:true});
        }
      }
      return fail(res,404,'Unknown API endpoint');
    } catch(e) {errors++;console.error('[ann-os]',e.message);return fail(res,e.status||500,e.message||'Server error');}
  }
  if(!['GET','HEAD'].includes(req.method))return fail(res,405,'Method not allowed');
  const rel=url.pathname==='/'?'/index.html':decodeURIComponent(url.pathname);
  const target=path.resolve(PUBLIC,'.'+rel);
  if(!target.startsWith(PUBLIC+path.sep))return fail(res,403,'Forbidden');
  try {
    const stat=await fsp.stat(target);if(!stat.isFile())return fail(res,404,'Not found');
    res.writeHead(200,{'Content-Type':MIME[path.extname(target)]||'application/octet-stream','Content-Length':stat.size,
      'X-Content-Type-Options':'nosniff','X-Frame-Options':'SAMEORIGIN','Referrer-Policy':'no-referrer'});
    if(req.method==='HEAD')res.end();else fs.createReadStream(target).pipe(res);
  } catch {return fail(res,404,'Not found');}
}
export const server=http.createServer((req,res)=>route(req,res).catch(e=>{
  console.error(e);if(!res.headersSent)fail(res,500,'Internal server error');else res.end();
}));
if(process.env.NODE_ENV==='production' && !ACCESS)throw Error('APP_ACCESS_TOKEN must be set in production.');
if(process.env.NODE_ENV!=='test')server.listen(PORT,'0.0.0.0',()=>console.log('ANN OS Cloud Studio on port '+PORT));
