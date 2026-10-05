import { randomUUID } from 'node:crypto'
import { mkdtempSync, mkdirSync, symlinkSync, chmodSync, readFileSync, existsSync, rmSync, writeFileSync, realpathSync, readdirSync, unlinkSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { spawn } from 'node:child_process'
import { connect } from 'node:net'
import { setTimeout as delay } from 'node:timers/promises'
import { CliEvents, cliCommand, nativeFile } from '../execution/adapters'
import { openToolBridge } from '../execution/hostTools'
import { ExecutionStore } from '../execution/store'
import type { CliStorage, ExecutionContext } from '../execution/types'
// The public distribution launches worker.js; build before running this suite.
const { ExecutionClient } = require('../../dist/execution/ExecutionClient') as typeof import('../execution/ExecutionClient')

let root: string, env: NodeJS.ProcessEnv
beforeEach(() => {
  root = mkdtempSync(join(tmpdir(), 'milkie-execution-'))
  mkdirSync(join(root, 'bin')); mkdirSync(join(root, 'cwd')); mkdirSync(join(root, 'home'))
  const fixture = resolve('tests/fixtures/agent-cli.cjs')
  chmodSync(fixture, 0o755)
  for (const name of ['grok', 'pi']) symlinkSync(fixture, join(root, 'bin', name))
  env = { ...process.env, HOME: join(root, 'home'), GROK_HOME: undefined, PATH: `${join(root,'bin')}:${process.env.PATH}` }
})
afterEach(() => rmSync(root, { recursive: true, force: true }))
function client(runtime = 'pi') {
  return new ExecutionClient({ dataDir: join(root, 'data'), connection: { contractVersion: 1, fields: { transport: 'agent-cli', runtime } }, env })
}
let storageSeq = 0
function storage(): CliStorage {
  const id = String(++storageSeq)
  const configDir = join(root, `config-${id}`), sessionDir = join(root, `sessions-${id}`)
  mkdirSync(configDir); mkdirSync(sessionDir)
  writeFileSync(join(configDir, 'auth.json'), '{"fixture":true}\n')
  return { configDir, sessionDir }
}
function cliContext(owner: { createContext(cwd: string, storage?: CliStorage): ExecutionContext }, cwd = join(root, 'cwd')) {
  return owner.createContext(cwd, storage())
}
async function waitFile(file: string) { for (let i=0;i<100;i++) { if (existsSync(file)) return; await delay(25) } throw new Error('fixture did not start') }

describe.each(['grok-cli','pi'])('%s SDK with deterministic child protocol', runtime => {
  test('persists exact native session across clients, isolates two contexts, rejects concurrent execution', async () => {
    const a = client(runtime), c1 = cliContext(a), c2 = cliContext(a)
    const first = await a.wait(a.start(c1.contextId,'random-marker-one'))
    expect(first.status).toBe('succeeded'); expect(first.stopped).toBe(true)
    const invocation = JSON.parse(readFileSync(join(root, 'cwd', 'cli-invocation.json'), 'utf8'))
    expect(invocation.credentialPresent).toBe(false)
    if (runtime === 'pi') { expect(invocation.piConfig).toBe(c1.configDir); expect(invocation.sessionDir).toBe(c1.sessionDir); expect(invocation.session).toBe(c1.nativeSessionFile) }
    else { expect(invocation.grokHome).toBe(c1.configDir); expect(invocation.leaderSocket).toBe(join(c1.configDir!, 'leader.sock')) }
    expect((await a.wait(a.start(c2.contextId,'random-marker-two'))).status).toBe('succeeded')
    const b = client(runtime)
    for (let round=0;round<2;round++) {
      const next = await b.wait(b.start(c1.contextId,'fixture:recall'))
      expect(next.output).toBe('random-marker-one'); expect(next.nativeSessionId).toBe(first.nativeSessionId); expect(next.runId).not.toBe(first.runId)
    }
    expect((await b.wait(b.start(c2.contextId,'fixture:recall'))).output).toBe('random-marker-two')
    const c3 = cliContext(b)
    expect((await b.wait(b.start(c3.contextId,'fixture:recall'))).output).toBe('')
    const slow = a.start(c1.contextId,'fixture:sleep')
    expect(()=>b.start(c1.contextId,'must-not-run')).toThrow('context_busy')
    expect((await b.cancel(slow)).status).toBe('cancelled')
  })
  test('missing session cannot silently start fresh; auth errors are redacted', async () => {
    const a=client(runtime), c=cliContext(a)
    expect((await a.wait(a.start(c.contextId,'marker'))).status).toBe('succeeded')
    const actual=a.getContext(c.contextId)
    const file=nativeFile(actual)
    rmSync(file)
    expect(()=>a.start(c.contextId,'fixture:recall')).toThrow('session_missing')
    expect(existsSync(file)).toBe(false)
    const other=cliContext(a)
    const auth=await a.wait(a.start(other.contextId,'fixture:auth'))
    expect(auth.status).toBe('failed');expect(auth.code).toBe('auth_failed')
    expect(JSON.stringify(auth)).not.toContain('DO-NOT-LEAK')
  })
  test('deadline and cancellation confirm task process-group termination', async () => {
    const a=client(runtime), c=cliContext(a)
    const timed=await a.wait(a.start(c.contextId,'fixture:sleep',{timeoutMs:500}))
    expect(timed.status).toBe('timed_out');expect(timed.stopped).toBe(true)
    const c2=cliContext(a), id=a.start(c2.contextId,'fixture:child')
    await waitFile(join(root,'cwd','child.pid'))
    const pid=Number(readFileSync(join(root,'cwd','child.pid'),'utf8')), start=Date.now()
    const result=await client(runtime).cancel(id)
    expect(result.status).toBe('cancelled');expect(result.stopped).toBe(true);expect(Date.now()-start).toBeLessThan(10000)
    expect(()=>process.kill(pid,0)).toThrow()
  })
  test('malformed protocol fails and unknown constraints never start', async () => {
    const a=client(runtime), c=cliContext(a)
    expect(()=>a.start(c.contextId,'x',{tokenLimit:1} as any)).toThrow('unsupported_constraint')
    const bad=await a.wait(a.start(c.contextId,'fixture:malformed'))
    expect(bad.status).toBe('failed');expect(bad.code).toBe('protocol_error')
  })
})
test('unsupported runtime remains parseable but cannot execute', () => {
  const a=client('claude-code');expect(a.capabilities()).toMatchObject({supported:false,code:'unsupported_runtime'})
  expect(()=>a.createContext(join(root,'cwd'))).toThrow('unsupported_runtime')
})
test('stale supervision is unknown and cannot be replayed', () => {
  const a=client(),c=cliContext(a),store=new ExecutionStore(join(root,'data'))
  const id='f7471111-1111-4111-8111-111111111111'
  store.claim(c.contextId,id)
  store.write('runs',id,{version:1,runId:id,contextId:c.contextId,status:'running',heartbeatAt:0,startedAt:0,stopped:false})
  expect(a.query(id)?.status).toBe('unknown');expect(()=>a.start(c.contextId,'retry')).toThrow('context_busy')
  expect(()=>a.query('../other')).toThrow('invalid_request')
})
test('Pi error in JSON stream is not success despite exit code zero', () => {
  const events=new CliEvents('pi')
  events.push(JSON.stringify({type:'message_end',message:{role:'assistant',content:[],stopReason:'error',errorMessage:'Authentication failed'}})+'\n')
  events.push('{"type":"agent_end"}\n');expect(events.code).toBe('auth_failed')
})
test('Grok max-turns stop is an iteration budget, not a generic process failure', () => {
  const events = new CliEvents('grok-cli')
  events.push(JSON.stringify({ type: 'end', stopReason: 'max_turns', sessionId: 'sid' }) + '\n')
  expect(events.code).toBe('iteration_budget_exhausted')
  expect(events.ended).toBe(false)
  expect(events.sessionId).toBe('sid')
  const reached = new CliEvents('grok-cli')
  reached.push(JSON.stringify({ type: 'max_turns_reached', sessionId: 'sid-2' }) + '\n')
  expect(reached.code).toBe('iteration_budget_exhausted')
  expect(reached.sessionId).toBe('sid-2')
})
test('Grok keeps an observed iteration budget when a later cancelled end arrives', () => {
  const events = new CliEvents('grok-cli')
  events.push('{"type":"max_turns_reached"}\n')
  events.push('{"type":"end","stopReason":"cancelled","sessionId":"same-session"}\n')
  events.push('{"type":"error","message":"cancelled"}\n')
  expect(events.code).toBe('iteration_budget_exhausted')
  expect(events.ended).toBe(false)
  expect(events.sessionId).toBe('same-session')
  const cancelled = new CliEvents('grok-cli')
  cancelled.push('{"type":"end","stopReason":"cancelled","sessionId":"same-session"}\n')
  expect(cancelled.code).toBe('native_cancelled')
  expect(cancelled.code).not.toBe('iteration_budget_exhausted')
})
test('Pi auto-retry success replaces an earlier assistant error', () => {
  const events=new CliEvents('pi')
  events.push(JSON.stringify({type:'session',id:'sid'})+'\n')
  events.push(JSON.stringify({type:'message_end',message:{role:'assistant',content:[],stopReason:'error',errorMessage:'overloaded'}})+'\n')
  events.push('{"type":"agent_end"}\n')
  events.push(JSON.stringify({type:'message_end',message:{role:'assistant',content:[{type:'text',text:'ACK'}],stopReason:'stop'}})+'\n')
  events.push('{"type":"agent_end"}\n')
  expect(events.code).toBeUndefined();expect(events.ended).toBe(true);expect(events.output).toBe('ACK');expect(events.sessionId).toBe('sid')
})
test('CLI argv contains exact session selection and enforceable tool flags, never user prompt', () => {
  const c: ExecutionContext={version:1,contextId:'id',connection:{contractVersion:1,transport:'agent-cli',runtime:'pi',hasApiKey:false,hasBaseUrl:false,source:'canonical'},cwd:root,hasExecuted:true,configDir:'/owned/config',sessionDir:'/owned/sessions',nativeSessionFile:'/owned/sessions/session.jsonl',nativeSessionId:'native'}
  const command=cliCommand(c,'private prompt',{toolPolicy:'read-only',timeoutMs:10},'/private/prompt')
  expect(command.args).toContain('--session-dir');expect(command.args).toContain('/owned/sessions');expect(command.args).toContain('read,grep,find,ls');expect(command.args).toContain('--no-extensions');expect(command.args).not.toContain('private prompt');expect(command.args).not.toContain('--continue')
  const grok: ExecutionContext={...c,connection:{...c.connection,runtime:'grok-cli'},configDir:'/owned/grok-config'}
  const grokCommand=cliCommand(grok,'private prompt',{toolPolicy:'read-only',timeoutMs:10},'/private/prompt')
  expect(grokCommand.args).toContain('--leader-socket');expect(grokCommand.args).toContain('/owned/grok-config/leader.sock');expect(grokCommand.args).not.toContain('private prompt')
  expect(grokCommand.args[grokCommand.args.indexOf('--disallowed-tools') + 1]).not.toContain('read_file')
  expect(grokCommand.args).not.toContain('--tools')
})
test('a context lock stops a second release from deleting the newer claim', async () => {
  const store = new ExecutionStore(join(root, 'store-lock'))
  const contextId = 'f7471111-1111-4111-8111-111111111112'
  const oldRun = randomUUID()
  const newRun = randomUUID()
  store.claim(contextId, oldRun)
  const ready = join(root, 'lock-ready')
  const gate = join(root, 'lock-gate')
  const holderCode = `const fs=require('node:fs');const {ExecutionStore}=require(${JSON.stringify(resolve('dist/execution/store.js'))});const store=new ExecutionStore(process.argv[1]);store.exclusive(process.argv[2],()=>{fs.writeFileSync(process.argv[5],'held');const deadline=Date.now()+5000;while(!fs.existsSync(process.argv[6])){if(Date.now()>deadline)throw new Error('timed out');Atomics.wait(new Int32Array(new SharedArrayBuffer(4)),0,0,20)}if(fs.readFileSync(store.path('active',process.argv[2]),'utf8')===process.argv[3])store.release(process.argv[2],process.argv[3]);store.claim(process.argv[2],process.argv[4])});console.log('done')`
  const holder = spawn(process.execPath, ['-e', holderCode, store.root, contextId, oldRun, newRun, ready, gate])
  let holderOut = ''
  holder.stdout.on('data', chunk => { holderOut += chunk })
  try {
    for (let i = 0; i < 50 && !existsSync(ready); i++) await delay(20)
    expect(existsSync(ready)).toBe(true)
    const releaseCode = `const {ExecutionStore}=require(${JSON.stringify(resolve('dist/execution/store.js'))});try{new ExecutionStore(process.argv[1]).release(process.argv[2],process.argv[3]);console.log('released')}catch(e){console.log(e.code)}`
    const released = await new Promise<string>((resolveRelease, reject) => {
      const child = spawn(process.execPath, ['-e', releaseCode, store.root, contextId, oldRun])
      let out = ''
      child.stdout.on('data', chunk => { out += chunk })
      child.on('error', reject)
      child.on('exit', () => resolveRelease(out.trim()))
    })
    expect(released).toBe('context_busy')
    expect(readFileSync(store.path('active', contextId), 'utf8')).toBe(oldRun)
    writeFileSync(gate, 'go')
    await new Promise<void>((resolveExit, reject) => { holder.on('error', reject); holder.on('exit', code => code === 0 ? resolveExit() : reject(new Error(holderOut || `holder exited ${code}`))) })
    expect(readFileSync(store.path('active', contextId), 'utf8')).toBe(newRun)
  } finally { if (holder.exitCode === null) holder.kill('SIGKILL') }
})
test('two processes cannot both hold one context lock', async () => {
  const store = new ExecutionStore(join(root, 'store-two-lock'))
  const contextId = 'f7471111-1111-4111-8111-111111111114'
  const log = join(root, 'lock-log')
  const code = `const fs=require('node:fs');const {ExecutionStore}=require(${JSON.stringify(resolve('dist/execution/store.js'))});try{new ExecutionStore(process.argv[1]).exclusive(process.argv[2],()=>{fs.appendFileSync(process.argv[3],'enter\\n');Atomics.wait(new Int32Array(new SharedArrayBuffer(4)),0,0,200);fs.appendFileSync(process.argv[3],'exit\\n')});console.log('entered')}catch(e){console.log(e.code)}`
  const run = () => new Promise<string>((resolveRun, reject) => {
    const child = spawn(process.execPath, ['-e', code, store.root, contextId, log])
    let out = ''
    child.stdout.on('data', chunk => { out += chunk })
    child.on('error', reject)
    child.on('exit', () => resolveRun(out.trim()))
  })
  const results = (await Promise.all([run(), run()])).sort()
  expect(results).toEqual(['context_busy', 'entered'])
  const text = readFileSync(log, 'utf8')
  expect(text).toBe('enter\nexit\n')
})
test('a killed holder releases the context lock', async () => {
  const store = new ExecutionStore(join(root, 'store-dead-lock'))
  const contextId = 'f7471111-1111-4111-8111-111111111115'
  const ready = join(root, 'dead-ready')
  const holderCode = `const fs=require('node:fs');const {ExecutionStore}=require(${JSON.stringify(resolve('dist/execution/store.js'))});new ExecutionStore(process.argv[1]).exclusive(process.argv[2],()=>{fs.writeFileSync(process.argv[3],'held');Atomics.wait(new Int32Array(new SharedArrayBuffer(4)),0,0,10000)});`
  const holder = spawn(process.execPath, ['-e', holderCode, store.root, contextId, ready])
  try {
    for (let i = 0; i < 50 && !existsSync(ready); i++) await delay(20)
    expect(existsSync(ready)).toBe(true)
    expect(() => store.exclusive(contextId, () => undefined)).toThrow('context_busy')
    holder.kill('SIGKILL')
    await new Promise<void>(resolveExit => holder.once('exit', () => resolveExit()))
    let entered = false
    for (let i = 0; i < 50 && !entered; i++) {
      try { store.exclusive(contextId, () => { entered = true }) }
      catch (error) { if (!(error instanceof Error) || !error.message.includes('context_busy')) throw error; await delay(20) }
    }
    expect(entered).toBe(true)
  } finally { if (holder.exitCode === null) holder.kill('SIGKILL') }
})
test('reconcile does not overwrite a result while the context lock is held', async () => {
  const a = client('grok-cli'), c = cliContext(a)
  const store = new ExecutionStore(join(root, 'data'))
  const runId = randomUUID()
  const callId = randomUUID()
  store.claim(c.contextId, runId)
  store.write('runs', runId, { version: 1, runId, contextId: c.contextId, status: 'unknown', startedAt: 1, heartbeatAt: 1, stopped: true })
  store.write('calls', callId, { version: 1, callId, name: 'alpha', input: { n: 1 }, runId, contextId: c.contextId, status: 'pending' })
  const ready = join(root, 'reconcile-ready')
  const gate = join(root, 'reconcile-gate')
  const holderCode = `const fs=require('node:fs');const {ExecutionStore}=require(${JSON.stringify(resolve('dist/execution/store.js'))});new ExecutionStore(process.argv[1]).exclusive(process.argv[2],()=>{fs.writeFileSync(process.argv[3],'held');const deadline=Date.now()+5000;while(!fs.existsSync(process.argv[4])){if(Date.now()>deadline)throw new Error('timed out');Atomics.wait(new Int32Array(new SharedArrayBuffer(4)),0,0,20)}});console.log('done')`
  const holder = spawn(process.execPath, ['-e', holderCode, store.root, c.contextId, ready, gate])
  try {
    for (let i = 0; i < 50 && !existsSync(ready); i++) await delay(20)
    expect(existsSync(ready)).toBe(true)
    const reconcileCode = `const {ExecutionClient}=require(${JSON.stringify(resolve('dist/execution/ExecutionClient.js'))});try{new ExecutionClient({dataDir:process.argv[1],connection:{contractVersion:1,fields:{transport:'agent-cli',runtime:'grok-cli'}}}).reconcile(process.argv[2],process.argv[3]);console.log('reconciled')}catch(e){console.log(e.code)}`
    const blocked = await new Promise<string>((resolveBlocked, reject) => {
      const child = spawn(process.execPath, ['-e', reconcileCode, store.root, callId, 'other'])
      let out = ''
      child.stdout.on('data', chunk => { out += chunk })
      child.on('error', reject)
      child.on('exit', () => resolveBlocked(out.trim()))
    })
    expect(blocked).toBe('context_busy')
    expect(store.read('calls', callId)).toMatchObject({ status: 'pending' })
  } finally {
    writeFileSync(gate, 'go')
    await new Promise<void>(resolveExit => {
      const timer = setTimeout(() => { if (holder.exitCode === null) holder.kill('SIGKILL') }, 2000)
      holder.once('exit', () => { clearTimeout(timer); resolveExit() })
    })
  }
  expect(a.reconcile(callId, 'done')).toMatchObject({ status: 'reconciled', output: 'done' })
  expect(() => a.reconcile(callId, 'other')).toThrow('invalid_request')
  expect(store.read('calls', callId)).toMatchObject({ output: 'done' })
})
test('two host processes cannot both claim an execution context', async () => {
  const store=new ExecutionStore(join(root,'store')),contextId='f7471111-1111-4111-8111-111111111111'
  const code=`const {ExecutionStore}=require(${JSON.stringify(resolve('dist/execution/store.js'))}); try {new ExecutionStore(process.argv[1]).claim(process.argv[2],process.argv[3]);console.log('claimed')}catch(e){console.log(e.code)}`
  const launch=(id:string)=>new Promise<string>((res,rej)=>{const p=spawn(process.execPath,['-e',code,store.root,contextId,id]);let out='';p.stdout.on('data',c=>out+=c);p.on('error',rej);p.on('exit',()=>res(out.trim()))})
  expect((await Promise.all([launch('one'),launch('two')])).sort()).toEqual(['claimed','context_busy'])
})
test('API uses the same public lifecycle without persisting credentials', async () => {
  const a=new ExecutionClient({dataDir:join(root,'data'),connection:{contractVersion:1,fields:{transport:'api',protocol:'openai-chat-completions',model:'fixture',apiKey:'SECRET-API-KEY',baseUrl:'https://secret.invalid/v1'}},env:{...env,NODE_OPTIONS:`--require=${resolve('tests/fixtures/execution-api.cjs')}`}})
  const c=a.createContext(join(root,'cwd'))
  expect(a.capabilities()).toMatchObject({supported:true,resume:false})
  const result=await a.wait(a.start(c.contextId,'hello'))
  expect(result.status).toBe('succeeded');expect(result.output).toBe('api:hello')
  const data=readFileSync(join(root,'data','contexts',`${c.contextId}.json`),'utf8')+readFileSync(join(root,'data','runs',`${result.runId}.json`),'utf8')
  expect(data).not.toMatch(/SECRET-API-KEY|secret.invalid/)
  expect((await a.cancel(a.start(c.contextId,'fixture:sleep'))).status).toBe('cancelled')
})
test.each(['grok-cli','pi'])('%s execution survives host loss as a queryable owned run', async runtime => {
  const a=client(runtime),c=cliContext(a)
  const code=`const {ExecutionClient}=require(${JSON.stringify(resolve('dist/execution/ExecutionClient.js'))});const c=new ExecutionClient({dataDir:process.argv[1],connection:{contractVersion:1,fields:{transport:'agent-cli',runtime:process.argv[3]}}});console.log(c.start(process.argv[2],'fixture:child'));setInterval(()=>{},1000)`
  const host=spawn(process.execPath,['-e',code,join(root,'data'),c.contextId,runtime],{env})
  const id=await new Promise<string>((res,rej)=>{host.stdout.once('data',b=>res(String(b).trim()));host.on('error',rej)})
  await waitFile(join(root,'cwd','child.pid'))
  const exit=new Promise<void>(res=>host.once('exit',()=>res()));host.kill('SIGKILL');await exit
  const b=client(runtime)
  expect(b.query(id)?.status).toBe('running')
  expect(()=>b.start(c.contextId,'replay')).toThrow('context_busy')
  const result=await b.cancel(id);expect(result.status).toBe('cancelled');expect(result.stopped).toBe(true)
})
test('Pi end event without a completed assistant message cannot claim success', () => {
  const events=new CliEvents('pi')
  events.push('{"type":"agent_end"}\n')
  expect(events.ended).toBe(false)
})
test('connection is captured at construction, not mutated by the caller', () => {
  const connection={contractVersion:1,fields:{transport:'agent-cli',runtime:'pi'}}
  const a=new ExecutionClient({dataDir:join(root,'data'),connection,env})
  connection.fields.runtime='claude-code'
  expect(cliContext(a).connection.runtime).toBe('pi')
})

test('API completion cannot report success after a deadline hidden by a blocked event loop', async () => {
  const client = new ExecutionClient({ dataDir:join(root,'data'), connection:{contractVersion:1,fields:{transport:'api',protocol:'openai-chat-completions',model:'fixture',apiKey:'fixture-key'}},env:{...env,NODE_OPTIONS:`--require=${resolve('tests/fixtures/execution-api.cjs')}`}})
  const context=client.createContext(join(root,'cwd'))
  const result=await client.wait(client.start(context.contextId,'fixture:busy',{timeoutMs:300}))
  expect(result.status).toBe('timed_out')
  expect(result.stopped).toBe(true)
})

test('mutating a returned context does not change the client connection', () => {
  const a=client('pi')
  const first=cliContext(a)
  first.connection.runtime='claude-code'
  expect(a.capabilities().supported).toBe(true)
  expect(cliContext(a).connection.runtime).toBe('pi')
})

test('signal denial persists unknown within the cancellation deadline without waiting forever for exit', async () => {
  const a=new ExecutionClient({dataDir:join(root,'data'),connection:{contractVersion:1,fields:{transport:'agent-cli',runtime:'pi'}},env:{...env,NODE_OPTIONS:`--require=${resolve('tests/fixtures/execution-deny-signals.cjs')}`}})
  const c=cliContext(a)
  const id=a.start(c.contextId,'fixture:sleep')
  await waitFile(join(root,'cwd','runner.pid'))
  const pid=Number(readFileSync(join(root,'cwd','runner.pid'),'utf8'))
  try {
    const started=Date.now(), result=await a.cancel(id)
    expect(Date.now()-started).toBeLessThan(10000)
    expect(result.status).toBe('unknown');expect(result.stopped).toBe(false)
    expect(new ExecutionStore(join(root,'data')).run(id)?.status).toBe('unknown')
    expect(()=>a.start(c.contextId,'must not replay')).toThrow('context_busy')
  } finally {
    try { process.kill(-pid,'SIGKILL') } catch { /* fixture may already have exited */ }
  }
})
test.each(['grok-cli','pi'])('%s cancellation tracks a task that leaves the original process group', async runtime => {
  const a=client(runtime), c=cliContext(a)
  const id=a.start(c.contextId,'fixture:escaped-child')
  await waitFile(join(root,'cwd','child.pid'))
  const pid=Number(readFileSync(join(root,'cwd','child.pid'),'utf8'))
  try {
    const result=await a.cancel(id)
    expect(result.status).toBe('cancelled');expect(result.stopped).toBe(true)
    expect(result.resources?.some(resource=>resource.pid===pid)).toBe(true)
    expect(()=>process.kill(pid,0)).toThrow()
  } finally { try { process.kill(pid,'SIGKILL') } catch { /* already stopped */ } }
})
test('dedicated storage is required and credential env does not reach the CLI', async () => {
  env.XAI_API_KEY = 'should-not-pass'
  const a = client('grok-cli')
  expect(() => a.createContext(join(root, 'cwd'))).toThrow('config_missing')
  const configOnly = storage()
  rmSync(configOnly.sessionDir, { recursive: true })
  expect(() => a.createContext(join(root, 'cwd'), configOnly)).toThrow('session_missing')
  const c = cliContext(a)
  expect(existsSync(join(env.HOME!, '.grok'))).toBe(false)
  await a.wait(a.start(c.contextId, 'marker'))
  const invocation = JSON.parse(readFileSync(join(root, 'cwd', 'cli-invocation.json'), 'utf8'))
  expect(invocation.grokHome).toBe(c.configDir)
  expect(invocation.leaderSocket).toBe(join(c.configDir!, 'leader.sock'))
  expect(invocation.credentialPresent).toBe(false)
  expect(existsSync(nativeFile(a.getContext(c.contextId)))).toBe(true)
  const shared = storage()
  const first = a.createContext(join(root, 'cwd'), shared)
  expect(() => a.createContext(join(root, 'cwd'), { configDir: shared.configDir, sessionDir: storage().sessionDir })).toThrow('invalid_request')
  expect(first.configDir).toBe(realpathSync(shared.configDir))
})
test.each(['grok-cli','pi'])('%s CLI gets HOME on the config directory and only allowlisted host variables', async runtime => {
  Object.assign(env, { HOST_ONLY_MARKER: 'x', NODE_OPTIONS: '--no-warnings', GROK_CONFIG: 'x', GROK_CONFIG_PATH: '/host/grok.toml', GROK_DEPLOYMENT_KEY: 'x', GROK_AUTH_PROVIDER_COMMAND: 'x', XDG_CONFIG_HOME: '/host/xdg', PI_OFFLINE: '1', HTTPS_PROXY: 'http://proxy.invalid:3128', LANG: 'C.UTF-8' })
  const a = client(runtime), c = cliContext(a)
  expect((await a.wait(a.start(c.contextId, 'marker'))).status).toBe('succeeded')
  const invocation = JSON.parse(readFileSync(join(root, 'cwd', 'cli-invocation.json'), 'utf8'))
  expect(invocation.home).toBe(c.configDir)
  for (const key of ['HOST_ONLY_MARKER', 'NODE_OPTIONS', 'GROK_CONFIG', 'GROK_CONFIG_PATH', 'GROK_DEPLOYMENT_KEY', 'GROK_AUTH_PROVIDER_COMMAND', 'XDG_CONFIG_HOME', 'PI_OFFLINE']) expect(invocation.envKeys).not.toContain(key)
  for (const key of ['PATH', 'HTTPS_PROXY', 'LANG']) expect(invocation.envKeys).toContain(key)
  const own = runtime === 'grok-cli' ? ['GROK_HOME', 'GROK_LEADER_SOCKET'] : ['PI_CODING_AGENT_DIR', 'PI_CODING_AGENT_SESSION_DIR']
  for (const key of own) expect(invocation.envKeys).toContain(key)
})
test('a Grok sessions entry created concurrently is validated instead of leaking a raw error', () => {
  const fs = require('node:fs') as typeof import('node:fs')
  const { resolveCliStorage } = require('../execution/adapters') as typeof import('../execution/adapters')
  const own = storage(), other = storage()
  const realSymlink = fs.symlinkSync
  const race = (target: string) => jest.spyOn(fs, 'symlinkSync').mockImplementationOnce((_from, path) => {
    realSymlink(target, path)
    throw Object.assign(new Error(`EEXIST: file already exists, symlink '${String(path)}'`), { code: 'EEXIST' })
  })
  try {
    race(realpathSync(own.sessionDir))
    expect(resolveCliStorage('grok-cli', own).sessionDir).toBe(realpathSync(own.sessionDir))
    race(realpathSync(own.sessionDir))
    let error: any
    try { resolveCliStorage('grok-cli', other) } catch (e) { error = e }
    expect(error?.code).toBe('invalid_request')
    expect(String(error?.message)).not.toContain(root)
  } finally { jest.restoreAllMocks() }
})
test('native cancellation and missing CLI login are separate structured failures', () => {
  const events=new CliEvents('grok-cli')
  events.push('{"type":"end","stopReason":"cancelled","sessionId":"native"}\n')
  expect(events.code).toBe('native_cancelled')
  const {classifyFailure}=require('../execution/adapters')
  expect(classifyFailure('Not signed in. To authenticate, run grok login.')).toBe('auth_failed')
  expect(classifyFailure('No API key found for the selected model.')).toBe('auth_failed')
})
test('a short wait returns the known active state without cancelling execution', async () => {
  const a=client('pi'), c=cliContext(a), id=a.start(c.contextId,'fixture:sleep')
  try { expect(['starting','running']).toContain((await a.wait(id,10)).status) }
  finally { await a.cancel(id) }
})
const alpha = { name: 'alpha', description: 'Alpha', inputSchema: { type: 'object' as const, properties: { n: { type: 'integer' as const } }, required: ['n'], additionalProperties: false } }
const beta = { name: 'beta', description: 'Beta', inputSchema: alpha.inputSchema }
function calls() {
  const dir = join(root, 'data', 'calls')
  if (!existsSync(dir)) return []
  return readdirSync(dir).filter(name => name.endsWith('.json')).map(name => JSON.parse(readFileSync(join(dir, name), 'utf8')))
}
describe.each(['grok-cli', 'pi'])('%s host tools', runtime => {
  test('capabilities, serial forwarding, rejection, and invalid input stay distinct', async () => {
    const a = client(runtime)
    expect(a.capabilities()).toMatchObject({ hostTools: true, nativeCallId: runtime === 'pi', forwarding: ['serial', 'parallel'] })
    const c = cliContext(a)
    if (runtime === 'pi') {
      const settingsFile = join(c.configDir!, 'settings.json')
      writeFileSync(settingsFile, JSON.stringify({ packages: ['npm:pi-web-access'] }))
      const blocked = await a.wait(a.start(c.contextId, 'x', { tools: [alpha], timeoutMs: 10000 }, async () => ({ ok: true, output: 'x' })))
      expect(blocked).toMatchObject({ status: 'failed', code: 'policy_mismatch' })
      unlinkSync(settingsFile)
    }
    expect(() => a.start(c.contextId, 'x', { tools: [alpha], toolPolicy: 'standard' }, async () => ({ ok: true, output: 'x' }))).toThrow('unsupported_constraint')
    expect(() => a.start(c.contextId, 'x', { forwarding: 'serial' })).toThrow('unsupported_constraint')
    expect(() => a.start(c.contextId, 'x', { tools: [alpha] })).toThrow('invalid_request')
    expect(() => a.start(c.contextId, 'x', { tools: [{ name: 'bash', description: 'shell', inputSchema: alpha.inputSchema }] }, async () => ({ ok: true, output: 'x' }))).toThrow('invalid_request')
    let called = false
    const invalid = await a.wait(a.start(c.contextId, 'fixture:invalid', { tools: [alpha], timeoutMs: 10000 }, () => { called = true; return { ok: true, output: 'x' } }))
    expect(invalid.status).toBe('succeeded'); expect(called).toBe(false)
    expect(calls()[0]).toMatchObject({ status: 'invalid_input', name: 'alpha', runId: invalid.runId, contextId: c.contextId })
    const rejected = await a.wait(a.start(cliContext(a).contextId, 'fixture:reject', { tools: [alpha], timeoutMs: 10000 }, () => ({ ok: false, code: 'rejected', message: 'no' })))
    expect(rejected.status).toBe('succeeded')
    expect(calls().find((call: { runId: string }) => call.runId === rejected.runId)).toMatchObject({ status: 'rejected', message: 'no' })
    let foreign = 0
    const denied = await a.wait(a.start(cliContext(a).contextId, 'fixture:foreign', { tools: [alpha], timeoutMs: 10000 }, () => { foreign += 1; return { ok: true, output: 'x' } }))
    expect(denied.status).toBe('succeeded'); expect(foreign).toBe(0)
    expect(calls().find((call: { runId: string }) => call.runId === denied.runId)).toMatchObject({ status: 'rejected', name: 'bash' })
    let active = 0, max = 0
    const order: string[] = []
    const serialContext = cliContext(a)
    const serial = await a.wait(a.start(serialContext.contextId, 'fixture:tools', { tools: [alpha, beta], timeoutMs: 10000 }, async call => {
      active += 1; max = Math.max(max, active); order.push(call.name); await delay(80); active -= 1
      return { ok: true, output: call.name }
    }))
    expect(serial.status).toBe('succeeded'); expect(serial.output).toBe('alpha|beta'); expect(order).toEqual(['alpha', 'beta']); expect(max).toBe(1)
    const recorded = calls().filter((call: { runId: string }) => call.runId === serial.runId).sort((left: { name: string }, right: { name: string }) => left.name.localeCompare(right.name))
    expect(recorded.map((call: { status: string }) => call.status)).toEqual(['succeeded', 'succeeded'])
    const alphaCall = recorded.find((call: { name: string }) => call.name === 'alpha')
    expect(a.toolCall(alphaCall.callId)).toMatchObject({ runId: serial.runId, contextId: serialContext.contextId, status: 'succeeded' })
    expect(JSON.stringify(alphaCall)).not.toMatch(/API_KEY|auth\.json|GROK_AUTH/)
    if (runtime === 'pi') expect(alphaCall.nativeCallId).toBe('native-alpha')
    else expect(alphaCall.nativeCallId).toBeUndefined()
    const invocation = JSON.parse(readFileSync(join(root, 'cwd', 'cli-invocation.json'), 'utf8'))
    expect(invocation.home).toBe(serialContext.configDir)
    expect(invocation.credentialPresent).toBe(false)
    if (runtime === 'pi') {
      expect(invocation.args).toEqual(expect.arrayContaining(['--no-extensions', '--no-builtin-tools', '--extension', '--tools', 'alpha,beta']))
      expect(invocation.args.join(' ')).not.toContain('read,bash')
    } else {
      expect(invocation.args).toEqual(expect.arrayContaining(['--disallowed-tools', '--deny', 'Bash(*)', 'Read(**)', 'Edit(**)', 'Grep']))
      const grokArgs = invocation.args.join(',')
      expect(grokArgs).toContain('x_search,web_search,web_fetch')
      expect(grokArgs).not.toContain('search_tool')
      expect(grokArgs).not.toContain('use_tool')
      for (const key of ['MILKIE_TOOL_SOCKET', 'GROK_MANAGED_CONFIG', 'GROK_CURSOR_MCPS_ENABLED', 'GROK_CLAUDE_HOOKS_ENABLED', 'GROK_CODEX_SKILLS_ENABLED']) expect(invocation.envKeys).toContain(key)
    }
    active = 0; max = 0
    await a.wait(a.start(cliContext(a).contextId, 'fixture:tools', { tools: [alpha, beta], forwarding: 'parallel', timeoutMs: 10000 }, async call => {
      active += 1; max = Math.max(max, active); await delay(80); active -= 1
      return { ok: true, output: call.name }
    }))
    expect(max).toBe(2)
  })
  test('host death leaves the run unknown and the unanswered call recorded', async () => {
    const a = client(runtime), c = cliContext(a)
    const host = spawn(process.execPath, [resolve('tests/fixtures/execution-tool-host.cjs')], { env })
    let output = '', pid = 0
    host.stdout.on('data', chunk => { output += chunk })
    try {
      host.stdin.end(JSON.stringify({ dataDir: join(root, 'data'), connection: { contractVersion: 1, fields: { transport: 'agent-cli', runtime } }, contextId: c.contextId, input: 'fixture:hold', constraints: { tools: [alpha], timeoutMs: 30000 } }))
      const runId = await new Promise<string>((resolveId, reject) => {
        const timer = setTimeout(() => reject(new Error(`host did not start: ${output}`)), 5000)
        const finish = () => { if (!output.trim()) return; clearTimeout(timer); resolveId(output.trim()) }
        host.stdout.on('data', finish)
        finish()
      })
      let pending: { status?: string } | undefined
      for (let i = 0; i < 200 && pending?.status !== 'pending'; i++) { pending = calls().find((call: { runId: string }) => call.runId === runId); await delay(25) }
      expect(pending?.status).toBe('pending')
      await waitFile(join(root, 'cwd', 'runner.pid'))
      pid = Number(readFileSync(join(root, 'cwd', 'runner.pid'), 'utf8'))
      const exited = new Promise<void>(resolveExit => host.once('exit', () => resolveExit()))
      host.kill('SIGKILL'); await exited
      let stored: { status?: string } | undefined
      for (let i = 0; i < 200 && stored?.status !== 'unknown'; i++) { stored = new ExecutionStore(join(root, 'data')).run(runId); await delay(25) }
      expect(stored?.status).toBe('unknown')
      expect(calls().find((call: { runId: string }) => call.runId === runId)?.status).toBe('pending')
      expect(() => process.kill(pid, 0)).toThrow()
      expect(() => a.start(c.contextId, 'replay')).toThrow('context_busy')
    } finally {
      if (host.exitCode === null) host.kill('SIGKILL')
      if (pid) { try { process.kill(pid, 'SIGKILL') } catch { /* already stopped */ } }
    }
  })
})
test('grok host tools fail before the model when the loaded servers do not match', async () => {
  const a = client('grok-cli'), c = cliContext(a)
  writeFileSync(join(root, 'cwd', '.mcp.json'), '{"mcpServers":{"evil":{"command":"node"}}}\n')
  const mismatch = await a.wait(a.start(c.contextId, 'fixture:tools', { tools: [alpha, beta], timeoutMs: 10000 }, async () => ({ ok: true, output: 'x' })))
  expect(mismatch.status).toBe('failed'); expect(mismatch.code).toBe('policy_mismatch')
  expect(existsSync(join(root, 'cwd', 'runner.pid'))).toBe(false)
  const owned = cliContext(a)
  writeFileSync(join(owned.configDir!, 'config.toml'), 'enabled = true\n')
  const foreign = await a.wait(a.start(owned.contextId, 'fixture:tools', { tools: [alpha], timeoutMs: 10000 }, async () => ({ ok: true, output: 'x' })))
  expect(foreign.code).toBe('policy_mismatch')
  expect(readFileSync(join(owned.configDir!, 'config.toml'), 'utf8')).toBe('enabled = true\n')
})
test('generated Pi extension keeps newline framing', () => {
  const { writePiExtension } = require('../execution/hostTools') as typeof import('../execution/hostTools')
  const file = join(root, 'extension.mjs')
  writePiExtension(file, [alpha], '/tmp/milkie-test.sock', 'serial')
  const source = readFileSync(file, 'utf8')
  expect(source).toContain("indexOf('\\n')")
  expect(source).toContain('export default function')
  expect(source).toContain('nativeCallId: toolCallId')
})
test.each([false, true])('budgeted Pi extension cancels uncounted compaction (host tools: %s)', hosted => {
  const { writePiExtension } = require('../execution/hostTools') as typeof import('../execution/hostTools')
  const file = join(root, 'budget.mjs'), marker = join(root, 'iteration.marker')
  writePiExtension(file, hosted ? [alpha] : [], hosted ? '/tmp/test.sock' : undefined, 'serial', { limit: 2, markerFile: marker })
  const source = readFileSync(file, 'utf8').replace(/^import .*$/gm, '').replace('export default function', 'return function')
  const hooks = new Map<string, (...args: any[]) => any>()
  const type = { String: () => ({}), Integer: () => ({}), Object: () => ({}) }
  const initialize = new Function('writeFileSync', 'Type', source)(writeFileSync, type)
  initialize({ on: (name: string, fn: (...args: any[]) => any) => hooks.set(name, fn), registerTool: () => {} })
  for (const reason of ['threshold', 'overflow', 'manual']) {
    expect(hooks.get('session_before_compact')?.({ reason })).toEqual({ cancel: true })
  }
  let requests = 0, aborted = false
  const ctx = { abort: () => { aborted = true } }
  for (let i = 0; i < 3; i++) {
    hooks.get('before_provider_request')!({}, ctx)
    if (!aborted) requests++
  }
  expect(requests).toBe(2)
  expect(readFileSync(marker, 'utf8')).toBe('exhausted\n')
})
test('quoted project mcp table cannot replace the host launch arguments', async () => {
  const { assertGrokMcpLaunch, assertProjectGrokConfig } = require('../execution/hostTools') as typeof import('../execution/hostTools')
  const dir = join(root, 'quoted-cwd')
  mkdirSync(join(dir, '.grok'), { recursive: true })
  writeFileSync(join(dir, '.grok', 'config.toml'), `["mcp_servers".milkie]\ncommand = ${JSON.stringify(process.execPath)}\nargs = ["project-evil.js"]\n`)
  expect(() => assertProjectGrokConfig(dir)).toThrow('policy_mismatch')
  writeFileSync(join(dir, '.grok', 'config.toml'), 'mcp_servers.milkie.args = ["project-evil.js"]\n')
  expect(() => assertProjectGrokConfig(dir)).toThrow('policy_mismatch')
  writeFileSync(join(dir, '.grok', 'config.toml'), '[ui]\nscreen_mode = "minimal"\n')
  expect(() => assertProjectGrokConfig(dir)).not.toThrow()
  expect(() => assertGrokMcpLaunch([{ name: 'milkie', command: process.execPath, args: ['project-evil.js'], enabled: true }], process.execPath, ['host.js', 'sock', 'tools.json'])).toThrow('policy_mismatch')
  const a = client('grok-cli'), c = cliContext(a)
  mkdirSync(join(root, 'cwd', '.grok'), { recursive: true })
  writeFileSync(join(root, 'cwd', '.grok', 'fixture-mcp-args.json'), JSON.stringify({ args: ['project-evil.js'] }))
  const blocked = await a.wait(a.start(c.contextId, 'fixture:tools', { tools: [alpha], timeoutMs: 10000 }, async () => ({ ok: true, output: 'x' })))
  expect(blocked).toMatchObject({ status: 'failed', code: 'policy_mismatch' })
  expect(existsSync(join(root, 'cwd', 'runner.pid'))).toBe(false)
})
test('project grok config cannot replace the host MCP server', async () => {
  const { assertGrokInventory } = require('../execution/hostTools') as typeof import('../execution/hostTools')
  const report = {
    mcpServers: [{ name: 'milkie', target: '/usr/bin/false' }],
    hooks: [], skills: [], plugins: [], lspServers: [], marketplaces: [],
    agents: [{ name: 'general-purpose', source: { type: 'builtin' } }],
    externalCompat: { cells: [{ enabled: false }] },
    permissions: { managedSettingsActive: false },
  }
  expect(() => assertGrokInventory(report, process.execPath)).toThrow('policy_mismatch')
  const a = client('grok-cli'), c = cliContext(a)
  mkdirSync(join(root, 'cwd', '.grok'))
  writeFileSync(join(root, 'cwd', '.grok', 'config.toml'), '[mcp_servers.milkie]\ncommand = "/usr/bin/false"\nargs = ["evil"]\n')
  const blocked = await a.wait(a.start(c.contextId, 'fixture:tools', { tools: [alpha], timeoutMs: 10000 }, async () => ({ ok: true, output: 'x' })))
  expect(blocked).toMatchObject({ status: 'failed', code: 'policy_mismatch' })
  expect(existsSync(join(root, 'cwd', 'runner.pid'))).toBe(false)
})
test('two host-tool runs cannot share one grok config directory', async () => {
  const a = client('grok-cli')
  const shared = storage()
  const first = a.createContext(join(root, 'cwd'), shared)
  const second = a.createContext(join(root, 'cwd'), shared)
  const runId = a.start(first.contextId, 'fixture:hold', { tools: [alpha], timeoutMs: 30000 }, async () => ({ ok: true, output: 'x' }))
  const file = join(shared.configDir, 'config.toml')
  try {
    for (let i = 0; i < 100 && !existsSync(file); i++) await delay(25)
    const before = readFileSync(file, 'utf8')
    const blocked = await a.wait(a.start(second.contextId, 'fixture:tools', { tools: [alpha], timeoutMs: 10000 }, async () => ({ ok: true, output: 'x' })))
    expect(blocked).toMatchObject({ status: 'failed', code: 'policy_mismatch' })
    expect(readFileSync(file, 'utf8')).toBe(before)
  } finally { await a.cancel(runId) }
})
test('a dead grok config lock stays closed until that run is confirmed stopped', async () => {
  const holder = spawn(process.execPath, ['-e', 'setInterval(() => {}, 1000)'], { stdio: 'ignore' })
  await new Promise<void>(resolve => holder.once('spawn', resolve))
  const pid = holder.pid!
  holder.kill('SIGKILL')
  await new Promise<void>(resolve => holder.once('exit', resolve))
  const a = client('grok-cli'), c = cliContext(a)
  writeFileSync(join(c.configDir!, 'milkie-host-tools.lock'), `${pid} ${randomUUID()}\n`)
  const blocked = await a.wait(a.start(c.contextId, 'fixture:tools', { tools: [alpha], timeoutMs: 10000 }, async () => ({ ok: true, output: 'x' })))
  expect(blocked).toMatchObject({ status: 'failed', code: 'policy_mismatch' })
  expect(existsSync(join(root, 'cwd', 'runner.pid'))).toBe(false)
})
test('an unreadable process inventory does not report the run stopped', async () => {
  const { ProcessTracker } = require('../execution/processes') as typeof import('../execution/processes')
  const child = spawn(process.execPath, ['-e', 'setInterval(() => {}, 1000)'], { stdio: 'ignore' })
  await new Promise<void>(resolve => child.once('spawn', resolve))
  let phase = 'boot'
  const self = { pid: process.pid, parent: 1, state: 'Ss', startedAt: 'boot', tagged: false }
  const tracker = new ProcessTracker(randomUUID(), () => {
    if (phase === 'boot') return [self]
    if (phase === 'seen') return [self, { pid: child.pid!, parent: process.pid, state: 'S', startedAt: 'child', tagged: true }]
    throw new Error('inventory unavailable')
  })
  phase = 'seen'
  tracker.start()
  await delay(250)
  phase = 'fail'
  try {
    expect(await tracker.stop()).toBe(false)
    expect(() => process.kill(child.pid!, 0)).not.toThrow()
  } finally { child.kill('SIGKILL') }
})
test.each(['grok-cli', 'pi'])('%s resume rejects a tool removed from the next registration', async runtime => {
  const a = client(runtime), c = cliContext(a)
  const first = await a.wait(a.start(c.contextId, 'fixture:tools', { tools: [alpha, beta], timeoutMs: 10000 }, async call => {
    expect(a.toolCall(call.callId)?.status).toBe('pending')
    if (call.name === 'alpha') {
      let betaPending = false
      for (let i = 0; i < 20 && !betaPending; i++) { betaPending = calls().some(item => item.runId === call.runId && item.name === 'beta' && item.status === 'pending'); if (!betaPending) await delay(10) }
      expect(betaPending).toBe(true)
    }
    return { ok: true, output: call.name }
  }))
  expect(first.status).toBe('succeeded')
  const before = a.getContext(c.contextId)
  const firstIds = calls().filter(call => call.runId === first.runId).map(call => call.callId)
  const b = client(runtime)
  let revokedHandler = false
  const second = await b.wait(b.start(c.contextId, 'fixture:revoked', { tools: [beta], timeoutMs: 10000 }, () => { revokedHandler = true; return { ok: true, output: 'x' } }))
  expect(second.status).toBe('succeeded')
  expect(revokedHandler).toBe(false)
  const revoked = calls().find(call => call.runId === second.runId && call.name === 'alpha')
  expect(revoked?.status).toBe('rejected')
  expect(firstIds).not.toContain(revoked?.callId)
  const listed = JSON.parse(readFileSync(join(root, 'data', 'runs', `${second.runId}.tools.json`), 'utf8')) as Array<{ name: string }>
  expect(listed.map(tool => tool.name)).toEqual(['beta', 'alpha'])
  const after = b.getContext(c.contextId)
  expect(after.nativeSessionId).toBe(before.nativeSessionId)
  expect(after.nativeSessionFile).toBe(before.nativeSessionFile)
})
test.each(['grok-cli', 'pi'])('%s lost tool reply blocks resume until the host reconciles it', async runtime => {
  const a = client(runtime), c = cliContext(a)
  const effect = join(root, `effect-${runtime}`)
  const host = spawn(process.execPath, [resolve('tests/fixtures/execution-tool-host.cjs')], { env })
  let output = ''
  host.stdout.on('data', chunk => { output += chunk })
  try {
    host.stdin.end(JSON.stringify({ dataDir: join(root, 'data'), connection: { contractVersion: 1, fields: { transport: 'agent-cli', runtime } }, contextId: c.contextId, input: 'fixture:hold', constraints: { tools: [alpha], timeoutMs: 30000 }, effect }))
    const runId = await new Promise<string>((resolveId, reject) => {
      const timer = setTimeout(() => reject(new Error(`host did not start: ${output}`)), 5000)
      const finish = () => { if (!output.trim()) return; clearTimeout(timer); resolveId(output.trim()) }
      host.stdout.on('data', finish)
      finish()
    })
    let pending: { callId?: string; status?: string; name?: string; input?: unknown } | undefined
    for (let i = 0; i < 200 && (pending?.status !== 'pending' || !existsSync(effect)); i++) { pending = calls().find(call => call.runId === runId); await delay(25) }
    expect(pending).toMatchObject({ status: 'pending', name: 'alpha', input: { n: 1 } })
    expect(() => a.reconcile(pending!.callId!, 'too soon')).toThrow('context_busy')
    const exited = new Promise<void>(resolveExit => host.once('exit', () => resolveExit()))
    host.kill('SIGKILL'); await exited
    for (let i = 0; i < 200 && new ExecutionStore(join(root, 'data')).run(runId)?.status !== 'unknown'; i++) await delay(25)
    const b = client(runtime)
    expect(() => b.start(c.contextId, 'again', { tools: [alpha], timeoutMs: 10000 }, async () => ({ ok: true, output: 'x' }))).toThrow('context_busy')
    expect(() => b.reconcile(pending!.callId!, '')).toThrow('invalid_request')
    const sessionBefore = b.getContext(c.contextId)
    expect(b.reconcile(pending!.callId!, 'already done').status).toBe('reconciled')
    let repeated = false
    const resumed = await b.wait(b.start(c.contextId, 'already done', { tools: [alpha], timeoutMs: 10000 }, () => { repeated = true; return { ok: true, output: 'again' } }))
    expect(resumed.status).toBe('succeeded')
    expect(repeated).toBe(false)
    expect(readFileSync(effect, 'utf8')).toBe('once')
    const sessionAfter = b.getContext(c.contextId)
    expect(sessionAfter.nativeSessionId).toBe(sessionBefore.nativeSessionId)
    expect(sessionAfter.nativeSessionFile).toBe(sessionBefore.nativeSessionFile)
  } finally { if (host.exitCode === null) host.kill('SIGKILL') }
})
test('reconcile keeps the claim until the lost run is confirmed stopped', () => {
  const a = client('grok-cli'), c = cliContext(a)
  const store = new ExecutionStore(join(root, 'data'))
  const runId = randomUUID()
  store.claim(c.contextId, runId)
  store.write('runs', runId, { version: 1, runId, contextId: c.contextId, status: 'unknown', startedAt: 1, heartbeatAt: 1, stopped: false })
  const callId = randomUUID()
  store.write('calls', callId, { version: 1, callId, name: 'alpha', input: { n: 1 }, runId, contextId: c.contextId, status: 'pending' })
  expect(a.reconcile(callId, 'checked').status).toBe('reconciled')
  expect(readFileSync(store.path('active', c.contextId), 'utf8')).toBe(runId)
  expect(() => a.start(c.contextId, 'again', { tools: [alpha], timeoutMs: 10000 }, async () => ({ ok: true, output: 'x' }))).toThrow('context_busy')
})
test('reconcile can release a claim that survived the checked write', () => {
  const a = client('grok-cli'), c = cliContext(a)
  const store = new ExecutionStore(join(root, 'data'))
  const runId = randomUUID()
  const callId = randomUUID()
  store.claim(c.contextId, runId)
  store.write('runs', runId, { version: 1, runId, contextId: c.contextId, status: 'unknown', startedAt: 1, heartbeatAt: 1, stopped: true })
  store.write('calls', callId, { version: 1, callId, name: 'alpha', input: { n: 1 }, runId, contextId: c.contextId, status: 'pending' })
  const write = store.write.bind(store)
  store.write = (kind: string, id: string, value: unknown) => {
    write(kind, id, value)
    if (kind === 'calls') throw new Error('simulated host loss after durable write')
  }
  ;(a as unknown as { store: ExecutionStore }).store = store
  expect(() => a.reconcile(callId, 'done')).toThrow('simulated host loss after durable write')
  const restarted = client('grok-cli')
  expect(restarted.pendingToolCalls(c.contextId)).toEqual([])
  expect(existsSync(store.path('active', c.contextId))).toBe(true)
  expect(restarted.reconcile(callId, 'done')).toMatchObject({ status: 'reconciled', output: 'done' })
  expect(existsSync(store.path('active', c.contextId))).toBe(false)
  expect(() => restarted.reconcile(callId, 'other')).toThrow('invalid_request')
})
test('a busy context lock still stores the terminal run', async () => {
  const a = client('grok-cli'), c = cliContext(a)
  const store = new ExecutionStore(join(root, 'data'))
  const runId = a.start(c.contextId, 'fixture:sleep', { timeoutMs: 30000 })
  for (let i = 0; i < 50 && store.run(runId)?.status !== 'running'; i++) await delay(20)
  expect(store.run(runId)?.status).toBe('running')
  const ready = join(root, 'busy-ready'), gate = join(root, 'busy-gate')
  const holderCode = `const fs=require('node:fs');const {ExecutionStore}=require(${JSON.stringify(resolve('dist/execution/store.js'))});new ExecutionStore(process.argv[1]).exclusive(process.argv[2],()=>{fs.writeFileSync(process.argv[3],'held');const deadline=Date.now()+8000;while(!fs.existsSync(process.argv[4])){if(Date.now()>deadline)throw new Error('timed out');Atomics.wait(new Int32Array(new SharedArrayBuffer(4)),0,0,20)}});`
  const holder = spawn(process.execPath, ['-e', holderCode, store.root, c.contextId, ready, gate])
  try {
    for (let i = 0; i < 50 && !existsSync(ready); i++) await delay(20)
    expect(existsSync(ready)).toBe(true)
    const cancelled = await a.cancel(runId)
    expect(cancelled).toMatchObject({ status: 'cancelled', stopped: true })
    expect(readFileSync(store.path('active', c.contextId), 'utf8')).toBe(runId)
    writeFileSync(gate, 'go')
    await new Promise<void>((resolveExit, reject) => { holder.on('error', reject); holder.on('exit', code => code === 0 ? resolveExit() : reject(new Error(`holder exited ${code}`))) })
    const resumed = await a.wait(a.start(c.contextId, 'again', { timeoutMs: 10000 }))
    expect(resumed.status).toBe('succeeded')
    expect(resumed.runId).not.toBe(runId)
    expect(existsSync(store.path('active', c.contextId))).toBe(false)
  } finally { if (holder.exitCode === null) { try { writeFileSync(gate, 'go') } catch { /* absent */ } holder.kill('SIGKILL') } }
})
test('start releases a claim left after the checked write', async () => {
  const a = client('grok-cli'), c = cliContext(a)
  const store = new ExecutionStore(join(root, 'data'))
  const runId = randomUUID()
  const callId = randomUUID()
  store.claim(c.contextId, runId)
  store.write('runs', runId, { version: 1, runId, contextId: c.contextId, status: 'unknown', startedAt: 1, heartbeatAt: 1, stopped: true })
  store.write('calls', callId, { version: 1, callId, name: 'alpha', input: { n: 1 }, runId, contextId: c.contextId, status: 'reconciled', output: 'done' })
  const restarted = client('grok-cli')
  expect(restarted.pendingToolCalls(c.contextId)).toEqual([])
  const resumed = await restarted.wait(restarted.start(c.contextId, 'resume-after-checked-write', { timeoutMs: 10000 }))
  expect(resumed.status).toBe('succeeded')
  expect(resumed.runId).not.toBe(runId)
  expect(existsSync(store.path('active', c.contextId))).toBe(false)
})
test('startup failure retains its lock through rollback and allows a repaired retry', async () => {
  const Database = require('better-sqlite3') as typeof import('better-sqlite3')
  const a = client('pi'), c = cliContext(a)
  const store = new ExecutionStore(join(root, 'data'))
  ;(a as unknown as { store: ExecutionStore }).store = store
  const readContext = store.context.bind(store)
  const contender = new Database(join(store.root, 'locks', `${c.contextId}.sqlite`), { timeout: 0 })
  let reads = 0, competingError: string | undefined
  store.context = contextId => {
    const context = readContext(contextId)
    if (++reads === 2) {
      // Compete after claim but before startup validation. Keep any acquired lock
      // until start returns, reproducing the cleanup failure on the old code.
      try { contender.exec('BEGIN IMMEDIATE') }
      catch (error) { competingError = (error as { code?: string }).code }
      unlinkSync(join(c.configDir!, 'auth.json'))
    }
    return context
  }
  try {
    expect(() => a.start(c.contextId, 'first')).toThrow('config_missing')
    expect(competingError).toBe('SQLITE_BUSY')
    expect(existsSync(store.path('active', c.contextId))).toBe(false)
    expect(readdirSync(join(store.root, 'runs'))).toEqual([])
  } finally {
    contender.close()
    store.context = readContext
    writeFileSync(join(c.configDir!, 'auth.json'), '{"fixture":true}\n')
  }
  expect((await a.wait(a.start(c.contextId, 'retry'))).status).toBe('succeeded')
})
test('a pending call recorded during claim is rejected before the next worker starts', () => {
  const a = client('grok-cli'), c = cliContext(a)
  const store = new ExecutionStore(join(root, 'data'))
  const claim = store.claim.bind(store)
  store.claim = (contextId: string, runId: string) => {
    claim(contextId, runId)
    const callId = randomUUID()
    store.write('calls', callId, { version: 1, callId, name: 'alpha', input: { n: 1 }, runId: randomUUID(), contextId, status: 'pending' })
  }
  ;(a as unknown as { store: ExecutionStore }).store = store
  expect(() => a.start(c.contextId, 'fixture:tools', { tools: [alpha], timeoutMs: 10000 }, async () => ({ ok: true, output: 'x' }))).toThrow('context_busy')
  expect(existsSync(store.path('active', c.contextId))).toBe(false)
  expect(readdirSync(join(root, 'data', 'runs')).filter(name => name.endsWith('.json'))).toEqual([])
})
test('a queued call stays discoverable when the host exits before its handler', async () => {
  const a = client('grok-cli'), c = cliContext(a)
  const host = spawn(process.execPath, [resolve('tests/fixtures/execution-tool-host.cjs')], { env })
  let output = ''
  host.stdout.on('data', chunk => { output += chunk })
  try {
    host.stdin.end(JSON.stringify({ dataDir: join(root, 'data'), connection: { contractVersion: 1, fields: { transport: 'agent-cli', runtime: 'grok-cli' } }, contextId: c.contextId, input: 'fixture:tools', constraints: { tools: [alpha, beta], timeoutMs: 30000 } }))
    const runId = await new Promise<string>((resolveId, reject) => {
      const timer = setTimeout(() => reject(new Error(`host did not start: ${output}`)), 5000)
      const finish = () => { if (!output.trim()) return; clearTimeout(timer); resolveId(output.trim()) }
      host.stdout.on('data', finish)
      finish()
    })
    let pending: Array<{ name: string; status: string }> = []
    for (let i = 0; i < 200 && pending.length < 2; i++) { pending = calls().filter((call: { runId: string; status: string }) => call.runId === runId && call.status === 'pending'); await delay(25) }
    expect(pending.map(call => call.name).sort()).toEqual(['alpha', 'beta'])
    const exited = new Promise<void>(resolveExit => host.once('exit', () => resolveExit()))
    host.kill('SIGKILL'); await exited
    for (let i = 0; i < 200 && new ExecutionStore(join(root, 'data')).run(runId)?.status !== 'unknown'; i++) await delay(25)
    const found = client('grok-cli').pendingToolCalls(c.contextId)
    expect(found.map(call => call.name).sort()).toEqual(['alpha', 'beta'])
    expect(found.every(call => call.status === 'pending')).toBe(true)
    expect(() => client('grok-cli').pendingToolCalls(randomUUID())).toThrow('context_not_found')
  } finally { if (host.exitCode === null) host.kill('SIGKILL') }
})
test('a closed tool socket leaves the completed host call pending', async () => {
  const a = client('grok-cli'), c = cliContext(a)
  const store = new ExecutionStore(join(root, 'data'))
  const runId = randomUUID()
  let release: () => void = () => undefined
  const gate = new Promise<void>(resolve => { release = resolve })
  const bridge = await openToolBridge({
    tools: [alpha], forwarding: 'serial', runId, contextId: c.contextId, store, alive: () => true,
    onHost: async () => { await gate; return { ok: true, output: 'charged' } },
    onBroken: () => undefined,
  })
  try {
    const socket = connect(bridge.socketPath)
    await new Promise<void>(resolve => socket.once('connect', resolve))
    socket.write(JSON.stringify({ id: '1', name: 'alpha', input: { n: 1 } }) + '\n')
    for (let i = 0; i < 50 && a.pendingToolCalls(c.contextId).length === 0; i++) await delay(10)
    expect(a.pendingToolCalls(c.contextId)).toHaveLength(1)
    socket.destroy()
    release()
    await delay(50)
    const pending = a.pendingToolCalls(c.contextId)
    expect(pending.map(call => call.status)).toEqual(['pending'])
    const call = pending[0]
    if (!call) throw new Error('missing pending call')
    expect(call.output).toBeUndefined()
    store.write('runs', runId, { version: 1, runId, contextId: c.contextId, status: 'unknown', startedAt: 1, heartbeatAt: 1, stopped: true })
    expect(() => a.start(c.contextId, 'again', { timeoutMs: 10000 })).toThrow('context_busy')
    expect(a.reconcile(call.callId, 'already charged')).toMatchObject({ status: 'reconciled', output: 'already charged' })
  } finally { await bridge.close() }
})
test('a live tool socket records success after the reply is written', async () => {
  const store = new ExecutionStore(join(root, 'data'))
  const contextId = 'f7471111-1111-4111-8111-111111111116'
  const runId = randomUUID()
  const bridge = await openToolBridge({
    tools: [alpha], forwarding: 'serial', runId, contextId, store, alive: () => true,
    onHost: async () => ({ ok: true, output: 'charged' }),
    onBroken: () => undefined,
  })
  try {
    const socket = connect(bridge.socketPath)
    const received = new Promise<string>((resolveReply, reject) => {
      const timer = setTimeout(() => reject(new Error('reply was not written')), 1000)
      socket.on('data', chunk => { clearTimeout(timer); resolveReply(chunk.toString()) })
    })
    await new Promise<void>(resolve => socket.once('connect', resolve))
    socket.write(JSON.stringify({ id: '1', name: 'alpha', input: { n: 1 } }) + '\n')
    expect(await received).toContain('"output":"charged"')
    const saved = readdirSync(join(store.root, 'calls')).filter(name => name.endsWith('.json')).map(name => JSON.parse(readFileSync(join(store.root, 'calls', name), 'utf8')))
    expect(saved).toEqual([expect.objectContaining({ status: 'succeeded', output: 'charged', contextId })])
    socket.end()
  } finally { await bridge.close() }
})
const save = { name: 'save', description: 'Save', inputSchema: { type: 'object' as const, properties: { content: { type: 'string' as const }, parts: { type: 'array' as const, items: { type: 'string' as const } } }, additionalProperties: false } }
const MAX_TOOL_BYTES = 256 * 1024
const MAX_ENCODED_BYTES = 2 * 1024 * 1024
function savedCalls(): Array<{ status: string; output?: string; message?: string; name: string }> {
  const dir = join(root, 'data', 'calls')
  if (!existsSync(dir)) return []
  return readdirSync(dir).filter(name => name.endsWith('.json')).map(name => JSON.parse(readFileSync(join(dir, name), 'utf8')))
}
function writeFileCase(calls: Array<{ id: string; name: string; input: unknown }>): void {
  writeFileSync(join(root, 'cwd', 'file-case.json'), JSON.stringify({ calls }))
}
describe.each(['grok-cli', 'pi'])('%s model iteration budget', runtime => {
  test('rejects an illegal budget before any provider request and accepts 50', async () => {
    const a = client(runtime), c = cliContext(a)
    expect(a.capabilities().modelIterations).toBe(true)
    for (const value of [0, 1.5, 10001, -1]) expect(() => a.start(c.contextId, 'nope', { maxModelIterations: value })).toThrow('unsupported_constraint')
    expect(existsSync(join(root, 'cwd', 'runner.pid'))).toBe(false)
    const allowed = await a.wait(a.start(c.contextId, 'within-budget', { maxModelIterations: 50, timeoutMs: 10000 }))
    expect(allowed.status).toBe('succeeded')
    expect(allowed.iterationBudget).toEqual({ limit: 50, exhausted: false })
    const invocation = JSON.parse(readFileSync(join(root, 'cwd', 'cli-invocation.json'), 'utf8')) as { args: string[] }
    if (runtime === 'grok-cli') expect(invocation.args).toEqual(expect.arrayContaining(['--max-turns', '50']))
    else {
      expect(invocation.args).not.toContain('--max-turns')
      const extension = invocation.args[invocation.args.indexOf('--extension') + 1]
      if (!extension) throw new Error('missing pi extension')
      const source = readFileSync(extension, 'utf8')
      expect(source).toContain('modelIterationBudget = 50')
      expect(source).toContain('ctx.abort()')
      expect(source).not.toContain('typebox')
    }
  })
  test('a no-tool loop stops at the budget and the same session can run again', async () => {
    const a = client(runtime), c = cliContext(a)
    const exhausted = await a.wait(a.start(c.contextId, 'fixture:loop', { maxModelIterations: 2, timeoutMs: 10000 }))
    expect(exhausted.status).toBe('failed')
    expect(exhausted.code).toBe('iteration_budget_exhausted')
    expect(exhausted.iterationBudget).toEqual({ limit: 2, exhausted: true })
    expect(JSON.parse(readFileSync(join(root, 'cwd', 'provider-requests.json'), 'utf8'))).toHaveLength(2)
    const invocation = JSON.parse(readFileSync(join(root, 'cwd', 'cli-invocation.json'), 'utf8')) as { args: string[] }
    if (runtime === 'pi') {
      const extension = invocation.args[invocation.args.indexOf('--extension') + 1]
      if (!extension) throw new Error('missing pi extension')
      expect(readFileSync(extension, 'utf8')).toContain('modelIterations > modelIterationBudget')
    }
    const next = await a.wait(a.start(c.contextId, 'still-here', { timeoutMs: 10000 }))
    expect(next.status).toBe('succeeded')
    expect(next.output).toBe('still-here')
    expect(next.nativeSessionId).toBe(exhausted.nativeSessionId)
  })
  test('timeout, cancellation, and session mismatch stay distinct from an unused budget', async () => {
    const a = client(runtime), c = cliContext(a)
    const timed = await a.wait(a.start(c.contextId, 'fixture:sleep', { timeoutMs: 500, maxModelIterations: 2 }))
    expect(timed.status).toBe('timed_out')
    expect(timed.code).not.toBe('iteration_budget_exhausted')
    expect(timed.iterationBudget).toEqual({ limit: 2, exhausted: false })
    const running = a.start(c.contextId, 'fixture:sleep', { timeoutMs: 10000, maxModelIterations: 2 })
    const cancelled = await a.cancel(running)
    expect(cancelled.status).toBe('cancelled')
    expect(cancelled.code).not.toBe('iteration_budget_exhausted')
    expect(cancelled.iterationBudget?.exhausted).not.toBe(true)
    const primed = await a.wait(a.start(c.contextId, 'prime-session', { maxModelIterations: 2, timeoutMs: 10000 }))
    expect(primed.status).toBe('succeeded')
    expect(primed.iterationBudget).toEqual({ limit: 2, exhausted: false })
    const mismatched = await a.wait(a.start(c.contextId, 'fixture:wrong-session', { maxModelIterations: 2, timeoutMs: 10000 }))
    expect(mismatched.status).toBe('failed')
    expect(mismatched.code).toBe('session_mismatch')
    expect(mismatched.iterationBudget).toEqual({ limit: 2, exhausted: false })
  })
})
test('a foreign session after Grok budget events stays a session mismatch', async () => {
  const a = client('grok-cli'), c = cliContext(a)
  const mismatched = await a.wait(a.start(c.contextId, 'fixture:loop-foreign', { maxModelIterations: 2, timeoutMs: 10000 }))
  expect(mismatched.status).toBe('failed')
  expect(mismatched.code).toBe('session_mismatch')
  expect(mismatched.iterationBudget).toEqual({ limit: 2, exhausted: false })
})
describe.each(['grok-cli', 'pi'])('%s host tool payload bounds', runtime => {
  test('delivers a legal result once and rejects an oversized result without a second call', async () => {
    const a = client(runtime), c = cliContext(a)
    const exact = 'x'.repeat(MAX_TOOL_BYTES)
    const eighty = 'x'.repeat(80 * 1024)
    const chinese = '中'.repeat(87381)
    expect(Buffer.byteLength(chinese)).toBe(262143)
    let effects = 0
    writeFileCase([{ id: '1', name: 'save', input: { content: '80kib' } }])
    const small = await a.wait(a.start(c.contextId, 'fixture:file', { tools: [save], forwarding: 'serial', timeoutMs: 20000 }, async () => {
      effects += 1
      return { ok: true, output: eighty }
    }))
    expect(small.status).toBe('succeeded')
    expect(small.output).toBe(String(80 * 1024))
    expect(effects).toBe(1)
    expect(savedCalls()).toEqual([expect.objectContaining({ status: 'succeeded', output: eighty })])
    effects = 0
    writeFileCase([{ id: '1', name: 'save', input: { content: 'exact' } }])
    const legal = await a.wait(a.start(c.contextId, 'fixture:file', { tools: [save], forwarding: 'serial', timeoutMs: 20000 }, async () => {
      effects += 1
      return { ok: true, output: exact }
    }))
    expect(legal.status).toBe('succeeded')
    expect(legal.output).toBe(String(MAX_TOOL_BYTES))
    expect(effects).toBe(1)
    expect(savedCalls().filter(call => call.output === exact)).toHaveLength(1)
    effects = 0
    writeFileCase([{ id: '1', name: 'save', input: { content: 'chinese' } }])
    const cjk = await a.wait(a.start(c.contextId, 'fixture:file', { tools: [save], forwarding: 'serial', timeoutMs: 20000 }, async () => {
      effects += 1
      return { ok: true, output: chinese }
    }))
    expect(cjk.status).toBe('succeeded')
    expect(effects).toBe(1)
    expect(savedCalls().some(call => call.output === chinese)).toBe(true)
    effects = 0
    writeFileCase([{ id: '1', name: 'save', input: { content: 'over' } }])
    const rejected = await a.wait(a.start(c.contextId, 'fixture:file', { tools: [save], forwarding: 'serial', timeoutMs: 20000 }, async () => {
      effects += 1
      return { ok: true, output: `${exact}y` }
    }))
    expect(rejected.status).toBe('succeeded')
    expect(rejected.output).toContain('rejected:Tool result exceeds 262144 bytes.')
    expect(effects).toBe(1)
    const over = savedCalls().filter(call => call.message === 'Tool result exceeds 262144 bytes.')
    expect(over).toEqual([expect.objectContaining({ status: 'rejected' })])
    expect(over[0]?.output).toBeUndefined()
  })
  test('accepts a 256 KiB raw input and rejects a larger one before the handler', async () => {
    const a = client(runtime), c = cliContext(a)
    let effects = 0
    const handler = async () => { effects += 1; return { ok: true as const, output: 'saved' } }
    writeFileCase([{ id: '1', name: 'save', input: { content: 'a'.repeat(MAX_TOOL_BYTES) } }])
    const legal = await a.wait(a.start(c.contextId, 'fixture:file', { tools: [save], forwarding: 'serial', timeoutMs: 20000 }, handler))
    expect(legal.status).toBe('succeeded')
    expect(legal.output).toBe(String(Buffer.byteLength('saved')))
    expect(effects).toBe(1)
    effects = 0
    writeFileCase([{ id: '1', name: 'save', input: { content: '\0'.repeat(MAX_TOOL_BYTES) } }])
    const nul = await a.wait(a.start(c.contextId, 'fixture:file', { tools: [save], forwarding: 'serial', timeoutMs: 20000 }, handler))
    expect(nul.status).toBe('succeeded')
    expect(effects).toBe(1)
    effects = 0
    writeFileCase([{ id: '1', name: 'save', input: { content: '\n'.repeat(200 * 1024) } }])
    const newlines = await a.wait(a.start(c.contextId, 'fixture:file', { tools: [save], forwarding: 'serial', timeoutMs: 20000 }, handler))
    expect(newlines.status).toBe('succeeded')
    expect(effects).toBe(1)
    effects = 0
    writeFileCase([{ id: '1', name: 'save', input: { content: '中'.repeat(87381) } }])
    const cjk = await a.wait(a.start(c.contextId, 'fixture:file', { tools: [save], forwarding: 'serial', timeoutMs: 20000 }, handler))
    expect(cjk.status).toBe('succeeded')
    expect(effects).toBe(1)
    effects = 0
    writeFileCase([{ id: '1', name: 'save', input: { content: 'a'.repeat(MAX_TOOL_BYTES + 1) } }])
    const rejected = await a.wait(a.start(c.contextId, 'fixture:file', { tools: [save], forwarding: 'serial', timeoutMs: 20000 }, handler))
    expect(rejected.status).toBe('succeeded')
    expect(rejected.output).toBe('invalid_input:Tool input exceeds 262144 bytes.')
    expect(effects).toBe(0)
    expect(savedCalls().some(call => call.status === 'invalid_input' && call.message === 'Tool input exceeds 262144 bytes.')).toBe(true)
  })
  test('rejects an encoded request over 2 MiB when every raw string is legal', async () => {
    const a = client(runtime), c = cliContext(a)
    const first = '\0'.repeat(MAX_TOOL_BYTES)
    const grokPrefix = Buffer.byteLength(JSON.stringify({ id: '1', name: 'save', input: { parts: [first, ''] } }))
    const piPrefix = Buffer.byteLength(JSON.stringify({ id: '1', name: 'save', nativeCallId: 'native-save', input: { parts: [first, ''] } }))
    const extra = Math.floor((MAX_ENCODED_BYTES - grokPrefix) / 6) + 1
    const grokBytes = grokPrefix + extra * 6
    const piBytes = piPrefix + extra * 6
    expect(extra).toBeLessThanOrEqual(MAX_TOOL_BYTES)
    expect(grokBytes).toBeGreaterThan(MAX_ENCODED_BYTES)
    expect(piBytes).toBeLessThanOrEqual(MAX_ENCODED_BYTES + 64 * 1024)
    let effects = 0
    writeFileCase([{ id: '1', name: 'save', input: { parts: [first, '\0'.repeat(extra)] } }])
    const rejected = await a.wait(a.start(c.contextId, 'fixture:file', { tools: [save], forwarding: 'serial', timeoutMs: 20000 }, async () => {
      effects += 1
      return { ok: true, output: 'saved' }
    }))
    expect(rejected.status).toBe('succeeded')
    expect(rejected.output).toBe('invalid_input:Encoded tool request exceeds 2097152 bytes.')
    expect(effects).toBe(0)
  })
})
test('tool bridge rejects an unfinished frame past the encoded slack without calling the handler', async () => {
  const store = new ExecutionStore(join(root, 'data'))
  let effects = 0
  let broken = false
  const bridge = await openToolBridge({
    tools: [save], forwarding: 'serial', runId: randomUUID(), contextId: randomUUID(), store, alive: () => true,
    onHost: async () => { effects += 1; return { ok: true, output: 'saved' } },
    onBroken: () => { broken = true },
  })
  try {
    const socket = connect(bridge.socketPath)
    await new Promise<void>(resolve => socket.once('connect', resolve))
    socket.write('x'.repeat(MAX_ENCODED_BYTES + 64 * 1024 + 1))
    await new Promise<void>(resolve => socket.once('close', () => resolve()))
    expect(broken).toBe(true)
    expect(effects).toBe(0)
    expect(savedCalls()).toEqual([])
  } finally { await bridge.close() }
})
test('Grok MCP oversized request is rejected and persisted before any host effect', async () => {
  const store = new ExecutionStore(join(root, 'data'))
  const runId = randomUUID(), contextId = randomUUID()
  let effects = 0
  const bridge = await openToolBridge({
    tools: [save], forwarding: 'serial', runId, contextId, store, alive: () => true,
    onHost: async () => { effects++; return { ok: true, output: 'saved' } }, onBroken: () => {},
  })
  const toolsFile = join(root, 'tools.json')
  writeFileSync(toolsFile, JSON.stringify([save]))
  const child = spawn(process.execPath, [resolve('dist/execution/mcp-server.js'), bridge.socketPath, toolsFile], { stdio: ['pipe', 'pipe', 'pipe'] })
  const deadline = new AbortController()
  try {
    const request = JSON.stringify({ jsonrpc: '2.0', id: 1, method: 'tools/call', params: { name: 'save', arguments: { parts: Array(9).fill('\0'.repeat(39000)) } } })
    expect(Buffer.byteLength(request)).toBeGreaterThan(MAX_ENCODED_BYTES)
    expect(Buffer.byteLength(request)).toBeLessThan(MAX_ENCODED_BYTES + 64 * 1024)
    const response = new Promise<any>((resolveReply, reject) => {
      let output = ''
      child.stdout.on('data', chunk => {
        output += chunk.toString()
        if (output.includes('\n')) resolveReply(JSON.parse(output.slice(0, output.indexOf('\n'))))
      })
      child.once('error', reject)
      child.once('exit', code => reject(new Error(`MCP exited before reply: ${code}`)))
    })
    child.stdin.write(request + '\n')
    const reply = await Promise.race([response, delay(5000, undefined, { signal: deadline.signal }).then(() => { throw new Error('MCP reply timed out') })])
    expect(reply.result).toMatchObject({ isError: true, content: [{ text: 'invalid_input: Encoded tool request exceeds 2097152 bytes.' }] })
    const calls = savedCalls()
    expect(calls).toHaveLength(1)
    expect(calls[0]).toMatchObject({ runId, contextId, name: 'save', status: 'invalid_input' })
    expect(effects).toBe(0)
  } finally {
    deadline.abort()
    if (child.exitCode === null && child.signalCode === null) {
      const exited = new Promise<void>(resolveExit => child.once('exit', () => resolveExit()))
      child.kill()
      await exited
    }
    await bridge.close()
  }
})
test('reconcile accepts 256 KiB and rejects one more byte', () => {
  const a = client('grok-cli'), c = cliContext(a)
  const store = new ExecutionStore(join(root, 'data'))
  const runId = randomUUID()
  store.claim(c.contextId, runId)
  store.write('runs', runId, { version: 1, runId, contextId: c.contextId, status: 'unknown', startedAt: 1, heartbeatAt: 1, stopped: true })
  const callId = randomUUID()
  store.write('calls', callId, { version: 1, callId, name: 'save', input: { content: 'x' }, runId, contextId: c.contextId, status: 'pending' })
  const exact = 'y'.repeat(MAX_TOOL_BYTES)
  expect(a.reconcile(callId, exact)).toMatchObject({ status: 'reconciled', output: exact })
  const again = randomUUID()
  store.write('calls', again, { version: 1, callId: again, name: 'save', input: { content: 'x' }, runId, contextId: c.contextId, status: 'pending' })
  expect(() => a.reconcile(again, `${exact}z`)).toThrow('invalid_request')
  expect(() => a.reconcile(again, '')).toThrow('invalid_request')
})
test('API transport cannot register host tools', () => {
  const a = new ExecutionClient({ dataDir: join(root, 'data'), connection: { contractVersion: 1, fields: { transport: 'api', protocol: 'openai-chat-completions', model: 'fixture', apiKey: 'fixture-key' } }, env })
  const c = a.createContext(join(root, 'cwd'))
  expect(a.capabilities().hostTools).toBe(false)
  expect(a.capabilities().modelIterations).toBe(false)
  expect(() => a.start(c.contextId, 'hello', { tools: [alpha] }, async () => ({ ok: true, output: 'x' }))).toThrow('unsupported_constraint')
  expect(() => a.start(c.contextId, 'hello', { maxModelIterations: 2 })).toThrow('unsupported_constraint')
  expect(existsSync(join(root, 'cwd', 'runner.pid'))).toBe(false)
})
