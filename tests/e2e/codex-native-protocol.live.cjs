// Opt-in real Codex CLI protocol check against a local HTTP 401 fixture.
// Proves native integration/error/session mechanics, not real model success.
const fs=require('node:fs'),path=require('node:path'),os=require('node:os'),http=require('node:http')
if(process.env.MILKIE_LIVE_CODEX_PROTOCOL!=='1')throw new Error('Set MILKIE_LIVE_CODEX_PROTOCOL=1; requires Codex 0.166.0 on PATH. Uses only a loopback provider, not real credentials.')
const {ExecutionClient}=require(path.resolve('dist/execution/ExecutionClient.js'))
const evidence=path.resolve(process.env.MILKIE_EVIDENCE_DIR||path.join('test-output/verify-milkie',`codex-native-protocol-${Date.now()}`))
fs.mkdirSync(evidence,{recursive:true})
const root=fs.mkdtempSync(path.join(os.tmpdir(),'milkie-codex-loopback-'))
const configDir=path.join(root,'config'),sessionDir=path.join(root,'sessions'),cwd=path.join(root,'work')
for(const d of [configDir,sessionDir,cwd])fs.mkdirSync(d)
let requests=0,safeToRemove=true
const server=http.createServer((req,res)=>{requests++;req.resume();res.writeHead(401,{'content-type':'application/json'});res.end(JSON.stringify({error:{message:'Offline fixture authentication rejection',type:'authentication_error',code:'invalid_api_key'}}))})
;(async()=>{
 await new Promise(r=>server.listen(0,'127.0.0.1',r))
 const port=server.address().port
 fs.writeFileSync(path.join(configDir,'auth.json'),JSON.stringify({OPENAI_API_KEY:'fixture-not-real'}))
 fs.writeFileSync(path.join(configDir,'config.toml'),`model_provider = "fixture"\n[model_providers.fixture]\nname = "fixture"\nbase_url = "http://127.0.0.1:${port}/v1"\nwire_api = "responses"\nrequires_openai_auth = false\nrequest_max_retries = 0\nstream_max_retries = 0\n`)
 const client=new ExecutionClient({dataDir:path.join(root,'data'),connection:{contractVersion:1,fields:{transport:'agent-cli',runtime:'codex',model:'gpt-6-astra'}},env:{...process.env,PATH:process.env.PATH,NO_PROXY:'127.0.0.1,localhost',no_proxy:'127.0.0.1,localhost'},lifetime:'host'})
 const c=client.createContext(cwd,{configDir,sessionDir});safeToRemove=false
 const id=client.start(c.contextId,'Local fixture protocol probe. No real credentials.',{timeoutMs:15000})
 let r
 const end=Date.now()+25000
 while(Date.now()<end){r=client.query(id);if(r?.finishedAt)break;await new Promise(r=>setTimeout(r,50))}
 if(!r?.finishedAt){await client.cancel(id);throw new Error('Native fixture did not stop')}
 safeToRemove=r.stopped===true
 const context=client.getContext(c.contextId),file=context.nativeSessionFile
 const header=file&&fs.existsSync(file)?JSON.parse(fs.readFileSync(file,'utf8').split('\n')[0]):undefined
 const summary={status:r.status,code:r.code,stopped:r.stopped,providerRequests:requests,provider:'loopback fixture only',realCredentials:false,sessionFileExists:!!header,headerType:header?.type,sessionIdMatches:header?.payload?.id===context.nativeSessionId,cwdMatches:header?.payload?.cwd===context.cwd}
 fs.writeFileSync(path.join(evidence,'summary.json'),JSON.stringify(summary,null,2))
 console.log(JSON.stringify(summary))
 if(r.status!=='failed'||r.code!=='auth_failed'||!r.stopped||requests!==1||header?.type!=='session_meta'||header.payload.id!==context.nativeSessionId||header.payload.cwd!==context.cwd)process.exitCode=1
})().catch(e=>{console.error(e.message);process.exitCode=1}).finally(()=>{server.close();if(safeToRemove)fs.rmSync(root,{recursive:true,force:true});else console.error('Preserved unresolved fixture execution: '+root)})
