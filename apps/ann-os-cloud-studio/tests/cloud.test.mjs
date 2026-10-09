import test from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import fsp from 'node:fs/promises';
import crypto from 'node:crypto';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const upstream=http.createServer(async(req,res)=>{
  const parts=[];for await(const c of req)parts.push(c);
  const raw=Buffer.concat(parts).toString('utf8');
  assert.equal(req.headers.authorization,'Bearer mock_key');
  let body;
  if(req.url==='/responses') {
    body=JSON.parse(raw);assert.equal(body.input.at(-1).role,'user');
    res.writeHead(200,{'content-type':'application/json'});
    return res.end(JSON.stringify({model:'mock-model',output:[{type:'message',content:[{type:'output_text',text:'Verified mock answer'}]}],usage:{output_tokens:9}}));
  }
  if(req.url==='/images/generations'){
    res.writeHead(200,{'content-type':'application/json'});
    return res.end(JSON.stringify({data:[{b64_json:'aGVsbG8='}]}));
  }
  if(req.url==='/audio/speech'){res.writeHead(200,{'content-type':'audio/mpeg'});return res.end('mock-audio');}
  if(req.url==='/audio/transcriptions'){
    assert.match(req.headers['content-type'],/multipart\/form-data/);
    res.writeHead(200,{'content-type':'application/json'});
    return res.end(JSON.stringify({text:'Verified voice transcript'}));
  }
  res.writeHead(404);res.end();
});
await new Promise(resolve=>upstream.listen(0,'127.0.0.1',resolve));
const DATA=await fsp.mkdtemp(path.join(os.tmpdir(),'annos-ci-'));
Object.assign(process.env,{NODE_ENV:'test',OPENAI_API_KEY:'mock_key',APP_ACCESS_TOKEN:'mock_access',
  OPENAI_BASE_URL:'http://127.0.0.1:'+upstream.address().port,DATA_DIR:DATA});
const {server,parseMessage,flattenText,makeVideo}=await import('../server.mjs');
await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
const BASE='http://127.0.0.1:'+server.address().port;
async function post(route,data,authorized=true){
  return fetch(BASE+route,{method:'POST',headers:{'content-type':'application/json',...(authorized?{'x-app-token':'mock_access'}:{})},body:JSON.stringify(data)});
}
test.after(async()=>{
  await new Promise(resolve=>server.close(resolve));
  await new Promise(resolve=>upstream.close(resolve));
  await fsp.rm(DATA,{recursive:true,force:true});
});
test('preserved React emulator is a byte-identical Git blob',async()=>{
  const file=path.join(path.dirname(fileURLToPath(import.meta.url)),'..','public','legacy','ann3d.html');
  const b=await fsp.readFile(file);
  const git=crypto.createHash('sha1').update('blob '+b.length+'\0').update(b).digest('hex');
  assert.equal(git,'d60c19c1c4dcc454ee63ddd4b00bfe054b1e0a4a');
});
test('frontend and status are served without cloud credentials',async()=>{
  const r=await fetch(BASE+'/');assert.equal(r.status,200);
  const html=await r.text();
  assert.match(html,/ANN OS Cloud Studio/);assert.match(html,/\/api\/chat/);
  const s=await (await fetch(BASE+'/api/status')).json();
  assert.equal(s.online,true);assert.equal(s.accessProtected,true);assert.equal(s.videoRenderer,true);
});
test('malformed turns are rejected and multimodal content is mapped',()=>{
  assert.throws(()=>parseMessage({}),/supported attachment/);
  const input=parseMessage({prompt:'describe',attachments:[{type:'image/png',data:'data:image/png;base64,aGVsbG8='}]});
  assert.equal(input.at(-1).content[1].type,'input_image');
  assert.equal(flattenText({output:[{type:'message',content:[{type:'output_text',text:'ok'}]}]}),'ok');
});
test('token gate is enforced',async()=>{
  const bad=await post('/api/chat',{prompt:'ping'},false);assert.equal(bad.status,401);
});
test('cloud inference uses a server-side credential with mocked upstream',async()=>{
  let r=await post('/api/chat',{prompt:'ping'});assert.equal(r.status,200);
  let data=await r.json();assert.equal(data.text,'Verified mock answer');
  assert.equal(data.tokens.output_tokens,9);
  r=await post('/api/image',{prompt:'draw an array'});assert.equal(r.status,200);
  data=await r.json();assert.equal(data.image,'data:image/png;base64,aGVsbG8=');
  r=await post('/api/speech',{text:'hello'});assert.equal(r.status,200);assert.equal(await r.text(),'mock-audio');
  r=await post('/api/transcribe',{data:'data:audio/webm;base64,'+Buffer.from('voice').toString('base64')});
  assert.equal(r.status,200);assert.equal((await r.json()).text,'Verified voice transcript');
});
test('session write/read/delete',async()=>{
  let r=await post('/api/sessions',{id:'session_abc1234',title:'Conversation',messages:[{role:'user',text:'hello'}]});
  assert.equal(r.status,200);
  r=await fetch(BASE+'/api/sessions',{headers:{'x-app-token':'mock_access'}});
  assert.equal((await r.json()).sessions[0].title,'Conversation');
  r=await fetch(BASE+'/api/sessions?id=session_abc1234',{method:'DELETE',headers:{'x-app-token':'mock_access'}});
  assert.equal(r.status,200);
  r=await fetch(BASE+'/api/sessions',{headers:{'x-app-token':'mock_access'}});
  assert.equal((await r.json()).sessions.length,0);
});
test('procedural video is a real MP4',async()=>{
  const b=await makeVideo('3D toroidal codec field');
  assert.ok(b.length>1000);
  assert.equal(b.toString('latin1',4,8),'ftyp');
});
