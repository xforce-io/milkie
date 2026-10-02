import { mkdtempSync, mkdirSync, symlinkSync, chmodSync, readFileSync, existsSync, rmSync, writeFileSync, realpathSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { spawn } from 'node:child_process'
import { setTimeout as delay } from 'node:timers/promises'
import { CliEvents, cliCommand, nativeFile } from '../execution/adapters'
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
