import { randomUUID } from 'node:crypto'
import { mkdtempSync, mkdirSync, symlinkSync, chmodSync, readFileSync, existsSync, rmSync, writeFileSync, realpathSync, readdirSync, unlinkSync } from 'node:fs'
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
  expect(grokCommand.args[grokCommand.args.indexOf('--disallowed-tools') + 1]).not.toContain('read_file')
  expect(grokCommand.args).not.toContain('--tools')
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
test('API transport cannot register host tools', () => {
  const a = new ExecutionClient({ dataDir: join(root, 'data'), connection: { contractVersion: 1, fields: { transport: 'api', protocol: 'openai-chat-completions', model: 'fixture', apiKey: 'fixture-key' } }, env })
  const c = a.createContext(join(root, 'cwd'))
  expect(a.capabilities().hostTools).toBe(false)
  expect(() => a.start(c.contextId, 'hello', { tools: [alpha] }, async () => ({ ok: true, output: 'x' }))).toThrow('unsupported_constraint')
})
