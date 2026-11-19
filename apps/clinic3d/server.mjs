// Loopback-only default. Public exposure is prohibited for this unvalidated prototype.
import http from 'node:http';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { dirname,join } from 'node:path';
import { randomBytes,scryptSync,timingSafeEqual } from 'node:crypto';
import { run,setEnabled,isEnabled,PROGRAM_SHA256 } from './engine.mjs';
const base=dirname(fileURLToPath(import.meta.url));
const securityHeaders={
  'x-content-type-options':'nosniff','referrer-policy':'no-referrer',
  'x-frame-options':'DENY','cache-control':'no-store',
  'content-security-policy':"default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
};
const secret=process.env.CLINIC_ADMIN_PASSWORD||'';
const salt=randomBytes(16);
const passwordHash=secret.length>=16?scryptSync(secret,salt,32):null;
const sessions=new Map();
const attempts=new Map();
const audit=[];
function auditEvent(action){audit.push({time:new Date().toISOString(),action});if(audit.length>200)audit.shift();}
function json(res,status,payload,headers={}){
  res.writeHead(status,{...securityHeaders,'content-type':'application/json; charset=utf-8',...headers});
  res.end(JSON.stringify(payload));
}
function parseCookie(req){const m=/\bclinic3d_admin=([0-9a-f]{64})\b/.exec(req.headers.cookie||'');return m?m[1]:null;}
function session(req){const token=parseCookie(req);const item=token&&sessions.get(token);
  return item&&item.expires>Date.now()?item:null;}
async function body(req){
  const chunks=[];let size=0;
  for await(const chunk of req){size+=chunk.length;if(size>8192) throw new Error('Request too large');chunks.push(chunk);}
  return JSON.parse(Buffer.concat(chunks).toString('utf8'));
}
const buckets=new Map();
function limited(req){const ip=req.socket.remoteAddress||'unknown';let b=buckets.get(ip);
  const now=Date.now();if(!b||b.until<now){b={hits:0,until:now+60000};buckets.set(ip,b);}
  return ++b.hits>90;
}
export function handler(req,res){
  Promise.resolve().then(async()=>{
    if(limited(req))return json(res,429,{error:'Rate limit'});
    const url=new URL(req.url,'http://localhost');
    if(req.method==='GET'&&url.pathname==='/health')
      return json(res,200,{ok:true,mode:'synthetic-only',bytecode:PROGRAM_SHA256});
    if(req.method==='GET'&&url.pathname==='/app.js'){
      const js=await readFile(join(base,'public/app.js'));
      res.writeHead(200,{...securityHeaders,'content-type':'text/javascript; charset=utf-8'});return res.end(js);
    }
    if(req.method==='GET'&&url.pathname==='/'){
      const page=await readFile(join(base,'public/index.html'));
      res.writeHead(200,{...securityHeaders,'content-type':'text/html; charset=utf-8'});return res.end(page);
    }
    if(req.method==='POST'&&url.pathname==='/api/simulate'){
      const input=await body(req);const result=run(input);auditEvent('simulation-completed');
      return json(res,200,result);
    }
    if(req.method==='POST'&&url.pathname==='/api/admin/login'){
      const {password}=await body(req);const who=req.socket.remoteAddress||'unknown';
      const failures=attempts.get(who)||{n:0,until:0};
      if(failures.until>Date.now())return json(res,429,{error:'Login temporarily locked'});
      if(!passwordHash||typeof password!=='string'||password.length>1024||
         !timingSafeEqual(passwordHash,scryptSync(password,salt,32))){
        failures.n++;if(failures.n>=5){failures.until=Date.now()+300000;failures.n=0;}
        attempts.set(who,failures);return json(res,403,{error:'Authentication failed'});
      }
      attempts.delete(who);const token=randomBytes(32).toString('hex'),csrf=randomBytes(24).toString('hex');
      sessions.set(token,{csrf,expires:Date.now()+900000});auditEvent('admin-login');
      return json(res,200,{csrf},{'set-cookie':'clinic3d_admin='+token+'; HttpOnly; SameSite=Strict; Path=/; Max-Age=900'});
    }
    if(url.pathname.startsWith('/api/admin/')){
      const s=session(req);if(!s)return json(res,403,{error:'Admin authentication required'});
      if(req.method==='GET'&&url.pathname==='/api/admin/state')
        return json(res,200,{enabled:isEnabled(),audit:audit.slice(-50)});
      if(req.method==='POST'&&url.pathname==='/api/admin/toggle'){
        if(req.headers['x-csrf-token']!==s.csrf)return json(res,403,{error:'CSRF denied'});
        const x=await body(req);if(typeof x.enabled!=='boolean')return json(res,400,{error:'Invalid state'});
        setEnabled(x.enabled);auditEvent(x.enabled?'enabled':'disabled');
        return json(res,200,{enabled:isEnabled()});
      }
    }
    return json(res,404,{error:'Not found'});
  }).catch(e=>json(res,400,{error:e.message==='Simulation disabled by administrator'?e.message:'Invalid request or simulation failure'}));
}
export function startServer(port=8090,host='127.0.0.1'){
  return new Promise((resolve,reject)=>{
    const server=http.createServer(handler);
    server.once('error',reject);
    server.listen(port,host,()=>resolve(server));
  });
}
if(process.argv[1]&&fileURLToPath(import.meta.url)===process.argv[1]){
  const port=Number(process.env.PORT||8090),host=process.env.HOST||'127.0.0.1';
  startServer(port,host).then(()=>console.log('Clinic3D synthetic-only on '+host+':'+port));
}
