import assert from 'node:assert/strict'
import { mkdtempSync, rmSync, readFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { pathToFileURL } from 'node:url'
import { createRequire } from 'node:module'
const require = createRequire(import.meta.url)
const { writePiExtension } = require('../../dist/execution/hostTools.js')
const piRoot = process.argv[2]
if (!piRoot) throw new Error('Pass the installed Pi package directory (validated with Pi 0.85.1).')
assert.equal(JSON.parse(readFileSync(join(piRoot, 'package.json'), 'utf8')).version, '0.85.1')
const { AgentSession } = await import(pathToFileURL(join(piRoot, 'dist/core/agent-session.js')).href)
const { ModelRuntime } = await import(pathToFileURL(join(piRoot, 'dist/core/model-runtime.js')).href)
const root = mkdtempSync(join(tmpdir(), 'milkie-budget-compaction-'))
try {
 for (const hosted of [false, true]) {
  const file = join(root, `extension-${hosted}.mjs`)
  writePiExtension(file, hosted ? [{name:'note',description:'test',inputSchema:{type:'object',properties:{text:{type:'string'}},required:['text']}}] : [], undefined, 'serial', {limit:2,markerFile:join(root,'marker')})
  const hooks = new Map()
  const extension = await import(pathToFileURL(file).href)
  const runtime = await ModelRuntime.create({authPath:join(root,'fixture-auth.json'),modelsPath:null,refreshOnCreate:false})
  extension.default({on:(name,fn)=>hooks.set(name,fn),registerTool:()=>{},registerProvider:p=>runtime.registerNativeProvider(p)})
  const originalProvider = runtime.getProvider('openai')
  hooks.get('session_start')({}, {model:{provider:'openai'},modelRegistry:runtime})
  assert.equal(runtime.getProvider('openai').auth,originalProvider.auth)
  let requests = 0
  const model = {id:'fixture',name:'fixture',api:'openai-responses',provider:'openai',baseUrl:'https://fixture.invalid/v1',reasoning:false,input:['text'],contextWindow:10000,maxTokens:100,cost:{input:0,output:0,cacheRead:0,cacheWrite:0}}
  for (let attempt=0;attempt<3;attempt++) {
   const abort = new AbortController()
   const response = await runtime.streamSimple(model,{messages:[{role:'user',content:'test',timestamp:Date.now()}]},{apiKey:'fixture',maxRetries:3,signal:abort.signal,onPayload:payload=>{hooks.get('before_provider_request')({}, {abort:()=>abort.abort()});return payload},fetch:async()=>{requests++;return new Response(JSON.stringify({error:{message:'fixture retry',type:'rate_limit_error'}}),{status:429,headers:{'content-type':'application/json','retry-after-ms':'1'}})}}).result()
   assert.equal(requests,Math.min(attempt+1,2))
   assert.equal(response.stopReason,attempt===2?'aborted':'error')
  }
  console.log(JSON.stringify({pi:'0.85.1',hosted,budget:2,requestedInternalRetries:3,transportRequests:requests,remoteRequests:0}))
  for (const reason of ['threshold','overflow']) {
   let calls = 0
   const events = []
   const session = Object.create(AgentSession.prototype)
   session.agent = {state:{model:{id:'fixture'}}}
   session.settingsManager = {getCompactionSettings:()=>({enabled:true,reserveTokens:1,keepRecentTokens:1})}
   session.sessionManager = {getBranch:()=>Array.from({length:4},(_,i)=>({type:'message',id:String(i),parentId:i?String(i-1):null,timestamp:new Date().toISOString(),message:{role:'user',content:`${i} ${'history '.repeat(100)}`,timestamp:Date.now()}}))}
   session._getSummarizationRequestAuth = async()=>({model:{id:'fixture'}})
   session._extensionRunner = {hasHandlers:name=>hooks.has(name),emit:async event=>hooks.get(event.type)?.(event,{})}
   session._emit = event=>events.push(event)
   session._emitSessionCompactFailed = async()=>{}
   session._runDefaultCompaction = async()=>{calls++;throw new Error('Unexpected provider request')}
   const retry = await session._runAutoCompaction(reason,reason==='overflow')
   assert.equal(calls,0)
   assert.equal(retry,false)
   assert.ok(events.some(e=>e.type==='compaction_end'&&e.aborted===true))
   console.log(JSON.stringify({pi:'0.85.1',hosted,reason,providerRequests: calls,retry,aborted:true}))
  }
 }
} finally {rmSync(root,{recursive:true,force:true})}
